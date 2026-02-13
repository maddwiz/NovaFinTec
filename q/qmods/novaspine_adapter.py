from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib import parse, request


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_slug(x: str) -> str:
    out = []
    for ch in str(x):
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        elif ch in (" ", "/", "\\", ":"):
            out.append("_")
    return "".join(out) or "default"


@dataclass
class PublishResult:
    backend: str
    published: int
    queued: int
    failed: int
    outbox_file: str | None = None
    error: str | None = None


def write_jsonl_outbox(events: Iterable[dict], outbox_dir: Path, prefix: str = "novaspine") -> Path:
    outbox_dir.mkdir(parents=True, exist_ok=True)
    stamp = _utc_now_iso().replace(":", "").replace("-", "")
    p = outbox_dir / f"{_safe_slug(prefix)}_{stamp}.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, separators=(",", ":"), ensure_ascii=True) + "\n")
    return p


def _json_request(
    url: str,
    method: str = "GET",
    payload: dict | None = None,
    token: str | None = None,
    timeout_sec: float = 6.0,
) -> tuple[int, dict | None]:
    headers = {"Accept": "application/json"}
    body = None
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = request.Request(str(url), method=method, data=body, headers=headers)
    with request.urlopen(req, timeout=float(timeout_sec)) as resp:
        code = int(getattr(resp, "status", 200))
        raw = resp.read()
        if not raw:
            return code, None
        try:
            return code, json.loads(raw.decode("utf-8", errors="replace"))
        except Exception:
            return code, None


def _base_url(x: str | None, default: str) -> str:
    s = str(x or default).strip()
    if not s.startswith(("http://", "https://")):
        s = "http://" + s
    return s.rstrip("/")


def _event_to_novaspine_ingest(
    ev: dict,
    namespace: str | None = None,
    source_prefix: str = "novafintec",
) -> dict:
    event_type = str(ev.get("event_type", "event"))
    trust = ev.get("trust", None)
    payload = ev.get("payload", {})
    ns = str(ev.get("namespace") or namespace or "private/nova/actions")
    compact = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    text = f"[{event_type}] trust={trust if trust is not None else 'na'} payload={compact}"
    # Guard overly large content; NovaSpine already chunks, but keep ingestion snappy.
    text = text[:16000]
    source_id = f"{source_prefix}:{event_type}"
    meta = {
        "namespace": ns,
        "event_type": event_type,
        "trust": trust,
        "ts_utc": ev.get("ts_utc", _utc_now_iso()),
        "origin": "novafintec",
        "payload": payload,
    }
    return {"text": text, "source_id": source_id, "metadata": meta}


def publish_events(
    events: list[dict],
    backend: str = "filesystem",
    namespace: str = "private/nova/actions",
    outbox_dir: Path | None = None,
    http_url: str | None = None,
    http_token: str | None = None,
    novaspine_base_url: str | None = None,
    novaspine_token: str | None = None,
    novaspine_source_prefix: str = "novafintec",
    timeout_sec: float = 6.0,
) -> PublishResult:
    backend = str(backend or "filesystem").strip().lower()
    outbox_dir = outbox_dir or Path("runs_plus") / "novaspine_outbox"
    if not events:
        return PublishResult(backend=backend, published=0, queued=0, failed=0)

    # Ensure every event is namespaced + timestamped.
    normalized = []
    for ev in events:
        x = dict(ev or {})
        x.setdefault("namespace", namespace)
        x.setdefault("ts_utc", _utc_now_iso())
        normalized.append(x)

    if backend in ("none", "off", "disabled"):
        return PublishResult(backend=backend, published=0, queued=len(normalized), failed=0)

    if backend in ("filesystem", "file", "local"):
        p = write_jsonl_outbox(normalized, outbox_dir=outbox_dir, prefix="novaspine_batch")
        return PublishResult(backend="filesystem", published=0, queued=len(normalized), failed=0, outbox_file=str(p))

    if backend in ("novaspine", "novaspine_api", "c3ae"):
        base = _base_url(novaspine_base_url, default="http://127.0.0.1:8420")
        tok = novaspine_token or http_token
        ingest_url = parse.urljoin(base + "/", "api/v1/memory/ingest")
        track_url = parse.urljoin(base + "/", "api/v1/events/track-batch")
        health_url = parse.urljoin(base + "/", "api/v1/health")
        try:
            code, _ = _json_request(health_url, method="GET", token=tok, timeout_sec=timeout_sec)
            if code < 200 or code >= 300:
                raise RuntimeError(f"health_status_{code}")
        except Exception as e:
            p = write_jsonl_outbox(normalized, outbox_dir=outbox_dir, prefix="novaspine_failed")
            return PublishResult(
                backend="novaspine_api",
                published=0,
                queued=len(normalized),
                failed=0,
                outbox_file=str(p),
                error=f"unreachable:{e}",
            )

        pub = 0
        fail = 0
        last_err = None
        for ev in normalized:
            payload = _event_to_novaspine_ingest(
                ev,
                namespace=str(ev.get("namespace") or namespace),
                source_prefix=novaspine_source_prefix,
            )
            try:
                code, _ = _json_request(ingest_url, method="POST", payload=payload, token=tok, timeout_sec=timeout_sec)
                if 200 <= code < 300:
                    pub += 1
                else:
                    fail += 1
                    last_err = f"ingest_status_{code}"
            except Exception as e:
                fail += 1
                last_err = str(e)

        # Best-effort motif tracking. This does not fail the batch if endpoint errors.
        try:
            _json_request(
                track_url,
                method="POST",
                payload={"events": [str(ev.get("event_type", "event")) for ev in normalized]},
                token=tok,
                timeout_sec=timeout_sec,
            )
        except Exception:
            pass

        if fail > 0:
            p = write_jsonl_outbox(normalized, outbox_dir=outbox_dir, prefix="novaspine_partial_failed")
            return PublishResult(
                backend="novaspine_api",
                published=pub,
                queued=max(0, len(normalized) - pub),
                failed=fail,
                outbox_file=str(p),
                error=last_err,
            )
        return PublishResult(backend="novaspine_api", published=pub, queued=0, failed=0)

    if backend in ("http", "https"):
        if not http_url:
            p = write_jsonl_outbox(normalized, outbox_dir=outbox_dir, prefix="novaspine_failed")
            return PublishResult(
                backend="http",
                published=0,
                queued=len(normalized),
                failed=0,
                outbox_file=str(p),
                error="missing_http_url",
            )

        try:
            payload = {"namespace": namespace, "events": normalized}
            code, _ = _json_request(str(http_url), method="POST", payload=payload, token=http_token, timeout_sec=timeout_sec)
            if 200 <= code < 300:
                return PublishResult(backend="http", published=len(normalized), queued=0, failed=0)
            p = write_jsonl_outbox(normalized, outbox_dir=outbox_dir, prefix="novaspine_failed")
            return PublishResult(
                backend="http",
                published=0,
                queued=len(normalized),
                failed=len(normalized),
                outbox_file=str(p),
                error=f"http_status_{code}",
            )
        except Exception as e:
            p = write_jsonl_outbox(normalized, outbox_dir=outbox_dir, prefix="novaspine_failed")
            return PublishResult(
                backend="http",
                published=0,
                queued=len(normalized),
                failed=len(normalized),
                outbox_file=str(p),
                error=str(e),
            )

    # Unknown backend: safe fallback to local queue.
    p = write_jsonl_outbox(normalized, outbox_dir=outbox_dir, prefix="novaspine_unknown_backend")
    return PublishResult(
        backend=backend,
        published=0,
        queued=len(normalized),
        failed=0,
        outbox_file=str(p),
        error="unknown_backend",
    )
