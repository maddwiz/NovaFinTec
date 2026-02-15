from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
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


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if (v == v) and (abs(v) != float("inf")) else default
    except Exception:
        return default


def write_jsonl_outbox(events: list[dict], outbox_dir: Path, prefix: str = "aion_novaspine") -> Path:
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


def _event_to_novaspine_ingest(ev: dict, namespace: str, source_prefix: str = "aion") -> dict:
    event_type = str(ev.get("event_type", "trade.event"))
    payload = ev.get("payload", {})
    compact = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    text = f"[{event_type}] payload={compact}"[:16000]
    return {
        "text": text,
        "source_id": f"{source_prefix}:{event_type}",
        "metadata": {
            "namespace": namespace,
            "event_type": event_type,
            "ts_utc": ev.get("ts_utc", _utc_now_iso()),
            "origin": "aion",
            "payload": payload,
        },
    }


def build_trade_event(
    event_type: str,
    symbol: str,
    side: str,
    qty: int,
    entry: float,
    exit: float,
    pnl: float,
    reason: str,
    confidence: float,
    regime: str,
    extra: dict | None = None,
) -> dict:
    payload = {
        "symbol": str(symbol).upper(),
        "side": str(side).upper(),
        "qty": int(max(0, int(qty))),
        "entry": float(_safe_float(entry, 0.0)),
        "exit": float(_safe_float(exit, 0.0)),
        "pnl": float(_safe_float(pnl, 0.0)),
        "reason": str(reason),
        "confidence": float(_safe_float(confidence, 0.0)),
        "regime": str(regime),
    }
    if isinstance(extra, dict):
        payload.update(extra)
    return {"event_type": str(event_type), "payload": payload, "ts_utc": _utc_now_iso()}


def emit_trade_event(event: dict, cfg) -> dict:
    enabled = bool(getattr(cfg, "MEMORY_ENABLE", False))
    if not enabled:
        return {"ok": True, "backend": "disabled", "published": 0, "queued": 0, "failed": 0}

    backend = str(getattr(cfg, "MEMORY_BACKEND", "filesystem")).strip().lower()
    namespace = str(getattr(cfg, "MEMORY_NAMESPACE", "private/nova/actions")).strip() or "private/nova/actions"
    outbox_raw = getattr(cfg, "MEMORY_OUTBOX_DIR", Path("logs") / "novaspine_outbox_aion")
    if outbox_raw is None or str(outbox_raw).strip() == "":
        outbox_raw = Path("logs") / "novaspine_outbox_aion"
    outbox_dir = Path(outbox_raw)
    timeout_sec = float(getattr(cfg, "MEMORY_TIMEOUT_SEC", 6.0))
    source_prefix = str(getattr(cfg, "MEMORY_SOURCE_PREFIX", "aion")).strip() or "aion"
    token = str(getattr(cfg, "MEMORY_TOKEN", "")).strip() or None

    events = [dict(event or {})]
    for e in events:
        e.setdefault("ts_utc", _utc_now_iso())

    if backend in {"filesystem", "file", "local"}:
        p = write_jsonl_outbox(events, outbox_dir=outbox_dir, prefix="aion_novaspine_batch")
        return {"ok": True, "backend": "filesystem", "published": 0, "queued": len(events), "failed": 0, "outbox_file": str(p)}

    if backend in {"novaspine", "novaspine_api", "c3ae"}:
        base = _base_url(getattr(cfg, "MEMORY_NOVASPINE_URL", "http://127.0.0.1:8420"), "http://127.0.0.1:8420")
        health_url = parse.urljoin(base + "/", "api/v1/health")
        ingest_url = parse.urljoin(base + "/", "api/v1/memory/ingest")
        try:
            code, _ = _json_request(health_url, method="GET", token=token, timeout_sec=timeout_sec)
            if not (200 <= code < 300):
                raise RuntimeError(f"health_status_{code}")
        except Exception as exc:
            p = write_jsonl_outbox(events, outbox_dir=outbox_dir, prefix="aion_novaspine_failed")
            return {
                "ok": False,
                "backend": "novaspine_api",
                "published": 0,
                "queued": len(events),
                "failed": 0,
                "outbox_file": str(p),
                "error": f"unreachable:{exc}",
            }

        pub = 0
        fail = 0
        last_err = None
        for ev in events:
            payload = _event_to_novaspine_ingest(ev, namespace=namespace, source_prefix=source_prefix)
            try:
                code, _ = _json_request(ingest_url, method="POST", payload=payload, token=token, timeout_sec=timeout_sec)
                if 200 <= code < 300:
                    pub += 1
                else:
                    fail += 1
                    last_err = f"ingest_status_{code}"
            except Exception as exc:
                fail += 1
                last_err = str(exc)

        if fail > 0:
            p = write_jsonl_outbox(events, outbox_dir=outbox_dir, prefix="aion_novaspine_partial_failed")
            return {
                "ok": False,
                "backend": "novaspine_api",
                "published": pub,
                "queued": max(0, len(events) - pub),
                "failed": fail,
                "outbox_file": str(p),
                "error": last_err,
            }
        return {"ok": True, "backend": "novaspine_api", "published": pub, "queued": 0, "failed": 0}

    # Unknown backend: fail-safe queue.
    p = write_jsonl_outbox(events, outbox_dir=outbox_dir, prefix="aion_novaspine_unknown_backend")
    return {
        "ok": False,
        "backend": backend,
        "published": 0,
        "queued": len(events),
        "failed": 0,
        "outbox_file": str(p),
        "error": "unknown_backend",
    }
