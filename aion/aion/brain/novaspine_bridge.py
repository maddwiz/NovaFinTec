from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import parse, request

_LAST_API_FAIL_TS = 0.0


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


def _stable_event_id(ev: dict) -> str:
    try:
        norm = {
            "event_type": str(ev.get("event_type", "")),
            "payload": ev.get("payload", {}),
            "ts_utc": str(ev.get("ts_utc", "")),
        }
        raw = json.dumps(norm, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except Exception:
        raw = str(ev)
    return hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()


def _normalize_event(ev: dict) -> dict:
    out = dict(ev or {})
    out.setdefault("ts_utc", _utc_now_iso())
    out.setdefault("event_type", "trade.event")
    payload = out.get("payload")
    if not isinstance(payload, dict):
        out["payload"] = {}
    out.setdefault("event_id", _stable_event_id(out))
    return out


def write_jsonl_outbox(events: list[dict], outbox_dir: Path, prefix: str = "aion_novaspine") -> Path:
    outbox_dir.mkdir(parents=True, exist_ok=True)
    stamp = _utc_now_iso().replace(":", "").replace("-", "")
    p = outbox_dir / f"{_safe_slug(prefix)}_{stamp}.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, separators=(",", ":"), ensure_ascii=True) + "\n")
    return p


def _load_jsonl_outbox(path: Path) -> list[dict]:
    out = []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []
    for ln in lines:
        s = str(ln).strip()
        if not s:
            continue
        try:
            row = json.loads(s)
        except Exception:
            continue
        if isinstance(row, dict):
            out.append(_normalize_event(row))
    return out


def _rewrite_jsonl_outbox(path: Path, events: list[dict]):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, separators=(",", ":"), ensure_ascii=True) + "\n")
    tmp.replace(path)


def _resolve_outbox_dir(cfg) -> Path:
    outbox_raw = getattr(cfg, "MEMORY_OUTBOX_DIR", Path("logs") / "novaspine_outbox_aion")
    if outbox_raw is None or str(outbox_raw).strip() == "":
        outbox_raw = Path("logs") / "novaspine_outbox_aion"
    return Path(outbox_raw)


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
    event_id = str(ev.get("event_id", _stable_event_id(ev)))
    compact = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    text = f"[{event_type}] payload={compact}"[:16000]
    rc_summary = _runtime_context_summary(payload if isinstance(payload, dict) else {})
    metadata = {
        "namespace": namespace,
        "event_id": event_id,
        "event_type": event_type,
        "ts_utc": ev.get("ts_utc", _utc_now_iso()),
        "origin": "aion",
        "payload": payload,
    }
    if rc_summary:
        metadata["runtime_context_summary"] = rc_summary
    return {
        "text": text,
        "source_id": f"{source_prefix}:{event_type}:{event_id[:16]}",
        "event_id": event_id,
        "metadata": metadata,
    }


def _runtime_context_summary(payload: dict | None) -> dict:
    if not isinstance(payload, dict):
        return {}
    rc = payload.get("runtime_context")
    if not isinstance(rc, dict):
        return {}

    flags = []
    raw_flags = rc.get("external_risk_flags", [])
    if isinstance(raw_flags, list):
        seen = set()
        for raw in raw_flags:
            key = str(raw).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            flags.append(key)

    src = str(rc.get("aion_feedback_source", rc.get("aion_feedback_source_selected", ""))).strip().lower() or "unknown"
    selected = str(rc.get("aion_feedback_source_selected", src)).strip().lower() or src
    pref = str(rc.get("aion_feedback_source_preference", "auto")).strip().lower() or "auto"

    return {
        "external_regime": str(rc.get("external_regime", "unknown")).strip().lower() or "unknown",
        "external_overlay_stale": bool(rc.get("external_overlay_stale", False)),
        "external_risk_flags": flags,
        "aion_feedback_status": str(rc.get("aion_feedback_status", "unknown")).strip().lower() or "unknown",
        "aion_feedback_source": src,
        "aion_feedback_source_selected": selected,
        "aion_feedback_source_preference": pref,
        "aion_feedback_stale": bool(rc.get("aion_feedback_stale", False)),
        "policy_block_new_entries": bool(rc.get("policy_block_new_entries", False)),
        "killswitch_block_new_entries": bool(rc.get("killswitch_block_new_entries", False)),
        "exec_governor_state": str(rc.get("exec_governor_state", "off")).strip().lower() or "off",
        "exec_governor_block_new_entries": bool(rc.get("exec_governor_block_new_entries", False)),
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
    global _LAST_API_FAIL_TS

    enabled = bool(getattr(cfg, "MEMORY_ENABLE", False))
    if not enabled:
        return {"ok": True, "backend": "disabled", "published": 0, "queued": 0, "failed": 0}

    backend = str(getattr(cfg, "MEMORY_BACKEND", "filesystem")).strip().lower()
    namespace = str(getattr(cfg, "MEMORY_NAMESPACE", "private/nova/actions")).strip() or "private/nova/actions"
    outbox_dir = _resolve_outbox_dir(cfg)
    timeout_sec = float(getattr(cfg, "MEMORY_TIMEOUT_SEC", 6.0))
    fail_cooldown_sec = max(0.0, float(getattr(cfg, "MEMORY_FAIL_COOLDOWN_SEC", 120.0)))
    source_prefix = str(getattr(cfg, "MEMORY_SOURCE_PREFIX", "aion")).strip() or "aion"
    token = str(getattr(cfg, "MEMORY_TOKEN", "")).strip() or None

    events = [_normalize_event(event if isinstance(event, dict) else {})]

    if backend in {"filesystem", "file", "local"}:
        p = write_jsonl_outbox(events, outbox_dir=outbox_dir, prefix="aion_novaspine_batch")
        return {"ok": True, "backend": "filesystem", "published": 0, "queued": len(events), "failed": 0, "outbox_file": str(p)}

    if backend in {"novaspine", "novaspine_api", "c3ae"}:
        if _LAST_API_FAIL_TS > 0 and fail_cooldown_sec > 0:
            dt_sec = time.monotonic() - float(_LAST_API_FAIL_TS)
            if dt_sec < fail_cooldown_sec:
                p = write_jsonl_outbox(events, outbox_dir=outbox_dir, prefix="aion_novaspine_cooldown")
                return {
                    "ok": False,
                    "backend": "novaspine_api",
                    "published": 0,
                    "queued": len(events),
                    "failed": 0,
                    "outbox_file": str(p),
                    "error": f"cooldown_active:{dt_sec:.2f}s<{fail_cooldown_sec:.2f}s",
                }

        base = _base_url(getattr(cfg, "MEMORY_NOVASPINE_URL", "http://127.0.0.1:8420"), "http://127.0.0.1:8420")
        health_url = parse.urljoin(base + "/", "api/v1/health")
        ingest_url = parse.urljoin(base + "/", "api/v1/memory/ingest")
        try:
            code, _ = _json_request(health_url, method="GET", token=token, timeout_sec=timeout_sec)
            if not (200 <= code < 300):
                raise RuntimeError(f"health_status_{code}")
        except Exception as exc:
            _LAST_API_FAIL_TS = time.monotonic()
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
            _LAST_API_FAIL_TS = time.monotonic()
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
        _LAST_API_FAIL_TS = 0.0
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


def replay_trade_outbox(cfg, max_files: int = 4, max_events: int = 200) -> dict:
    global _LAST_API_FAIL_TS

    enabled = bool(getattr(cfg, "MEMORY_ENABLE", False))
    if not enabled:
        return {
            "ok": True,
            "backend": "disabled",
            "replayed": 0,
            "failed": 0,
            "processed_files": 0,
            "moved_files": 0,
            "remaining_files": 0,
        }

    backend = str(getattr(cfg, "MEMORY_BACKEND", "filesystem")).strip().lower()
    outbox_dir = _resolve_outbox_dir(cfg)
    outbox_dir.mkdir(parents=True, exist_ok=True)
    sent_dir = outbox_dir / "sent"
    sent_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in outbox_dir.glob("*.jsonl") if p.is_file()], key=lambda p: p.name)
    queued_files = int(len(files))

    if backend in {"filesystem", "file", "local"}:
        return {
            "ok": True,
            "backend": "filesystem",
            "replayed": 0,
            "failed": 0,
            "processed_files": 0,
            "moved_files": 0,
            "queued_files": queued_files,
            "remaining_files": queued_files,
            "reason": "filesystem_backend",
        }

    if backend not in {"novaspine", "novaspine_api", "c3ae"}:
        return {
            "ok": False,
            "backend": backend,
            "replayed": 0,
            "failed": 0,
            "processed_files": 0,
            "moved_files": 0,
            "queued_files": queued_files,
            "remaining_files": queued_files,
            "error": "unknown_backend",
        }

    if queued_files <= 0:
        return {
            "ok": True,
            "backend": "novaspine_api",
            "replayed": 0,
            "failed": 0,
            "processed_files": 0,
            "moved_files": 0,
            "queued_files": 0,
            "remaining_files": 0,
        }

    timeout_sec = float(getattr(cfg, "MEMORY_TIMEOUT_SEC", 6.0))
    fail_cooldown_sec = max(0.0, float(getattr(cfg, "MEMORY_FAIL_COOLDOWN_SEC", 120.0)))
    namespace = str(getattr(cfg, "MEMORY_NAMESPACE", "private/nova/actions")).strip() or "private/nova/actions"
    source_prefix = str(getattr(cfg, "MEMORY_SOURCE_PREFIX", "aion")).strip() or "aion"
    token = str(getattr(cfg, "MEMORY_TOKEN", "")).strip() or None
    base = _base_url(getattr(cfg, "MEMORY_NOVASPINE_URL", "http://127.0.0.1:8420"), "http://127.0.0.1:8420")
    health_url = parse.urljoin(base + "/", "api/v1/health")
    ingest_url = parse.urljoin(base + "/", "api/v1/memory/ingest")

    if _LAST_API_FAIL_TS > 0 and fail_cooldown_sec > 0:
        dt_sec = time.monotonic() - float(_LAST_API_FAIL_TS)
        if dt_sec < fail_cooldown_sec:
            return {
                "ok": False,
                "backend": "novaspine_api",
                "replayed": 0,
                "failed": 0,
                "processed_files": 0,
                "moved_files": 0,
                "queued_files": queued_files,
                "remaining_files": queued_files,
                "error": f"cooldown_active:{dt_sec:.2f}s<{fail_cooldown_sec:.2f}s",
            }

    try:
        code, _ = _json_request(health_url, method="GET", token=token, timeout_sec=timeout_sec)
        if not (200 <= int(code) < 300):
            raise RuntimeError(f"health_status_{code}")
    except Exception as exc:
        _LAST_API_FAIL_TS = time.monotonic()
        return {
            "ok": False,
            "backend": "novaspine_api",
            "replayed": 0,
            "failed": 0,
            "processed_files": 0,
            "moved_files": 0,
            "queued_files": queued_files,
            "remaining_files": queued_files,
            "error": f"unreachable:{exc}",
        }

    processed_files = 0
    moved_files = 0
    replayed = 0
    failed = 0
    replay_budget = max(1, int(max_events))
    file_budget = max(1, int(max_files))
    last_err = None

    for p in files:
        if processed_files >= file_budget or replay_budget <= 0:
            break
        events = _load_jsonl_outbox(p)
        processed_files += 1
        if not events:
            try:
                p.rename(sent_dir / p.name)
                moved_files += 1
            except Exception:
                pass
            continue

        remaining: list[dict] = []
        for idx, ev in enumerate(events):
            if replay_budget <= 0:
                remaining.extend(events[idx:])
                break
            payload = _event_to_novaspine_ingest(ev, namespace=namespace, source_prefix=source_prefix)
            try:
                code, _ = _json_request(ingest_url, method="POST", payload=payload, token=token, timeout_sec=timeout_sec)
                if 200 <= int(code) < 300:
                    replayed += 1
                else:
                    failed += 1
                    remaining.append(ev)
                    last_err = f"ingest_status_{code}"
            except Exception as exc:
                failed += 1
                remaining.append(ev)
                last_err = str(exc)
            replay_budget -= 1

        if remaining:
            try:
                _rewrite_jsonl_outbox(p, remaining)
            except Exception:
                last_err = last_err or "outbox_rewrite_failed"
        else:
            try:
                p.rename(sent_dir / p.name)
                moved_files += 1
            except Exception:
                last_err = last_err or "outbox_move_failed"

    if failed > 0:
        _LAST_API_FAIL_TS = time.monotonic()
    else:
        _LAST_API_FAIL_TS = 0.0

    remaining_files = int(len([p for p in outbox_dir.glob("*.jsonl") if p.is_file()]))
    return {
        "ok": failed == 0 and last_err is None,
        "backend": "novaspine_api",
        "replayed": int(replayed),
        "failed": int(failed),
        "processed_files": int(processed_files),
        "moved_files": int(moved_files),
        "queued_files": int(queued_files),
        "remaining_files": int(remaining_files),
        "error": last_err,
    }
