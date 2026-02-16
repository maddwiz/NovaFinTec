from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_float(x, default: float | None = None):
    try:
        v = float(x)
    except Exception:
        return default
    if not math.isfinite(v):
        return default
    return float(v)


def _safe_list(x):
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    return []


def read_runtime_controls(path: Path) -> dict:
    if not path.exists() or path.is_dir():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data
    except Exception:
        return default


def runtime_controls_age_sec(path: Path, now_ts: float | None = None) -> float | None:
    if not path.exists():
        return None
    try:
        st = path.stat()
    except Exception:
        return None
    ts = time.time() if now_ts is None else float(now_ts)
    age = ts - float(st.st_mtime)
    if not math.isfinite(age):
        return None
    return float(max(0.0, age))


def runtime_controls_stale_threshold_sec(
    payload: dict | None,
    *,
    default_loop_seconds: int,
    base_stale_seconds: int,
) -> float:
    p = payload if isinstance(payload, dict) else {}
    loop_seconds = max(5, _safe_int(p.get("loop_seconds"), max(5, int(default_loop_seconds))))
    watchlist_size = max(0, _safe_int(p.get("watchlist_size"), 0))

    # Baseline expects at least ~6 loops worth of updates.
    base_dynamic = loop_seconds * 6
    # Larger watchlists can make a single loop substantially longer.
    watchlist_factor = min(18, watchlist_size // 12)
    with_watchlist = loop_seconds * (6 + watchlist_factor)
    dynamic = max(base_dynamic, with_watchlist)

    floor = max(60, int(base_stale_seconds))
    ceiling = loop_seconds * 24
    return float(max(floor, min(dynamic, ceiling)))


def runtime_controls_stale_info(
    path: Path,
    *,
    default_loop_seconds: int,
    base_stale_seconds: int,
    now_ts: float | None = None,
) -> dict:
    payload = read_runtime_controls(path)
    age = runtime_controls_age_sec(path, now_ts=now_ts)
    threshold = runtime_controls_stale_threshold_sec(
        payload,
        default_loop_seconds=default_loop_seconds,
        base_stale_seconds=base_stale_seconds,
    )
    stale = bool(age is not None and age > threshold)
    return {
        "age_sec": age,
        "threshold_sec": threshold,
        "stale": stale,
        "payload": payload,
    }


def overlay_runtime_status(path: Path, max_age_hours: float) -> dict:
    out = {
        "exists": False,
        "age_hours": None,
        "age_source": "unknown",
        "generated_at_utc": None,
        "max_age_hours": float(max_age_hours),
        "stale": False,
        "runtime_context_present": False,
        "runtime_context": {},
        "risk_flags": [],
    }
    if not isinstance(path, Path) or not path.exists():
        return out

    out["exists"] = True
    payload = _read_json(path, {})
    mtime_age = None
    try:
        ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        mtime_age = float((datetime.now(timezone.utc) - ts).total_seconds() / 3600.0)
    except Exception:
        mtime_age = None

    if isinstance(payload, dict):
        raw_ts = payload.get("generated_at_utc", payload.get("generated_at"))
        s = str(raw_ts or "").strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s) if s else None
            if dt is not None:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                dt = dt.astimezone(timezone.utc)
                out["generated_at_utc"] = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                out["age_hours"] = float((datetime.now(timezone.utc) - dt).total_seconds() / 3600.0)
                out["age_source"] = "payload"
        except Exception:
            pass

    if out["age_hours"] is None and isinstance(mtime_age, float):
        out["age_hours"] = float(max(0.0, mtime_age))
        out["age_source"] = "mtime"
    if isinstance(out["age_hours"], float):
        out["age_hours"] = float(max(0.0, out["age_hours"]))
    if isinstance(out["age_hours"], float) and float(max_age_hours) > 0:
        out["stale"] = bool(out["age_hours"] > float(max_age_hours))

    if not isinstance(payload, dict):
        return out

    ext_ctx = payload.get("runtime_context", {})
    if isinstance(ext_ctx, dict):
        out["runtime_context_present"] = len(ext_ctx) > 0
        out["runtime_context"] = ext_ctx
        flags = ext_ctx.get("risk_flags", [])
        if isinstance(flags, list):
            out["risk_flags"] = [str(x).strip().lower() for x in flags if str(x).strip()]
    return out


def aion_feedback_runtime_info(
    runtime_controls: dict | None,
    external_overlay_runtime: dict | None,
) -> dict:
    rc = runtime_controls if isinstance(runtime_controls, dict) else {}
    ext = external_overlay_runtime if isinstance(external_overlay_runtime, dict) else {}
    ext_ctx = ext.get("runtime_context", {}) if isinstance(ext.get("runtime_context"), dict) else {}
    ext_af = ext_ctx.get("aion_feedback", {}) if isinstance(ext_ctx.get("aion_feedback"), dict) else {}

    rc_has = any(
        k in rc
        for k in [
            "aion_feedback_active",
            "aion_feedback_status",
            "aion_feedback_risk_scale",
            "aion_feedback_closed_trades",
            "aion_feedback_age_hours",
            "aion_feedback_stale",
            "aion_feedback_source",
            "aion_feedback_source_selected",
        ]
    )
    if rc_has:
        source = "runtime_controls"
        feedback_source = (
            str(rc.get("aion_feedback_source", rc.get("aion_feedback_source_selected", ""))).strip().lower()
            or "unknown"
        )
        feedback_source_selected = (
            str(rc.get("aion_feedback_source_selected", feedback_source)).strip().lower() or feedback_source
        )
        feedback_source_preference = (
            str(rc.get("aion_feedback_source_preference", "auto")).strip().lower() or "auto"
        )
        data = {
            "active": bool(rc.get("aion_feedback_active", False)),
            "status": str(rc.get("aion_feedback_status", "unknown")).strip().lower() or "unknown",
            "feedback_source": feedback_source,
            "feedback_source_selected": feedback_source_selected,
            "feedback_source_preference": feedback_source_preference,
            "risk_scale": _safe_float(rc.get("aion_feedback_risk_scale"), None),
            "closed_trades": max(0, _safe_int(rc.get("aion_feedback_closed_trades", 0), 0)),
            "hit_rate": _safe_float(rc.get("aion_feedback_hit_rate"), None),
            "profit_factor": _safe_float(rc.get("aion_feedback_profit_factor"), None),
            "expectancy": _safe_float(rc.get("aion_feedback_expectancy"), None),
            "drawdown_norm": _safe_float(rc.get("aion_feedback_drawdown_norm"), None),
            "age_hours": _safe_float(rc.get("aion_feedback_age_hours"), None),
            "max_age_hours": _safe_float(rc.get("aion_feedback_max_age_hours"), None),
            "stale": bool(rc.get("aion_feedback_stale", False)),
            "last_closed_ts": str(rc.get("aion_feedback_last_closed_ts", "")).strip() or None,
            "path": str(rc.get("aion_feedback_path", "")).strip(),
            "reasons": _safe_list(rc.get("aion_feedback_reasons", [])),
            "block_new_entries": bool(rc.get("aion_feedback_block_new_entries", False)),
        }
    elif ext_af:
        source = "overlay_runtime_context"
        feedback_source = (
            str(ext_af.get("source", ext_af.get("source_selected", ""))).strip().lower() or "unknown"
        )
        feedback_source_selected = (
            str(ext_af.get("source_selected", feedback_source)).strip().lower() or feedback_source
        )
        feedback_source_preference = str(ext_af.get("source_preference", "auto")).strip().lower() or "auto"
        data = {
            "active": bool(ext_af.get("active", False)),
            "status": str(ext_af.get("status", "unknown")).strip().lower() or "unknown",
            "feedback_source": feedback_source,
            "feedback_source_selected": feedback_source_selected,
            "feedback_source_preference": feedback_source_preference,
            "risk_scale": _safe_float(ext_af.get("risk_scale"), None),
            "closed_trades": max(0, _safe_int(ext_af.get("closed_trades", 0), 0)),
            "hit_rate": _safe_float(ext_af.get("hit_rate"), None),
            "profit_factor": _safe_float(ext_af.get("profit_factor"), None),
            "expectancy": _safe_float(ext_af.get("expectancy"), None),
            "drawdown_norm": _safe_float(ext_af.get("drawdown_norm"), None),
            "age_hours": _safe_float(ext_af.get("age_hours"), None),
            "max_age_hours": _safe_float(ext_af.get("max_age_hours"), None),
            "stale": bool(ext_af.get("stale", False)),
            "last_closed_ts": str(ext_af.get("last_closed_ts", "")).strip() or None,
            "path": str(ext_af.get("path", "")).strip(),
            "reasons": _safe_list(ext_af.get("reasons", [])),
            "block_new_entries": bool(ext_af.get("block_new_entries", False)),
        }
    else:
        source = "none"
        data = {
            "active": False,
            "status": "unknown",
            "feedback_source": "unknown",
            "feedback_source_selected": "unknown",
            "feedback_source_preference": "auto",
            "risk_scale": None,
            "closed_trades": 0,
            "hit_rate": None,
            "profit_factor": None,
            "expectancy": None,
            "drawdown_norm": None,
            "age_hours": None,
            "max_age_hours": None,
            "stale": False,
            "last_closed_ts": None,
            "path": "",
            "reasons": [],
            "block_new_entries": False,
        }

    age_hours = data.get("age_hours")
    max_age_hours = data.get("max_age_hours")
    stale = bool(data.get("stale", False))
    if (not stale) and age_hours is not None and max_age_hours is not None and max_age_hours > 0:
        stale = bool(age_hours > max_age_hours)
    data["stale"] = stale

    status = str(data.get("status", "unknown")).strip().lower() or "unknown"
    active = bool(data.get("active", False))
    risk_scale = data.get("risk_scale")
    if not active:
        state = "inactive"
    elif stale:
        state = "stale"
    elif status in {"alert", "hard"}:
        state = "alert"
    elif status == "warn":
        state = "warn"
    elif status == "ok":
        state = "ok"
    elif risk_scale is not None and risk_scale <= 0.82:
        state = "alert"
    elif risk_scale is not None and risk_scale <= 0.94:
        state = "warn"
    else:
        state = "unknown"

    severity = {"inactive": 0, "unknown": 1, "ok": 1, "warn": 2, "stale": 2, "alert": 3}.get(state, 1)
    return {
        **data,
        "state": state,
        "severity": int(severity),
        "source": source,
        "present": source != "none",
    }


def memory_feedback_runtime_info(
    runtime_controls: dict | None,
    external_overlay_runtime: dict | None,
) -> dict:
    rc = runtime_controls if isinstance(runtime_controls, dict) else {}
    ext = external_overlay_runtime if isinstance(external_overlay_runtime, dict) else {}
    ext_ctx = ext.get("runtime_context", {}) if isinstance(ext.get("runtime_context"), dict) else {}
    ext_mf = ext_ctx.get("memory_feedback", {}) if isinstance(ext_ctx.get("memory_feedback"), dict) else {}

    rc_has = any(
        k in rc
        for k in [
            "memory_feedback_active",
            "memory_feedback_status",
            "memory_feedback_risk_scale",
            "memory_feedback_trades_scale",
            "memory_feedback_open_scale",
            "memory_feedback_block_new_entries",
            "memory_feedback_reasons",
        ]
    )
    if rc_has:
        source = "runtime_controls"
        data = {
            "active": bool(rc.get("memory_feedback_active", False)),
            "status": str(rc.get("memory_feedback_status", "unknown")).strip().lower() or "unknown",
            "risk_scale": _safe_float(rc.get("memory_feedback_risk_scale"), None),
            "max_trades_scale": _safe_float(rc.get("memory_feedback_trades_scale"), None),
            "max_open_scale": _safe_float(rc.get("memory_feedback_open_scale"), None),
            "turnover_pressure": _safe_float(rc.get("memory_feedback_turnover_pressure"), None),
            "turnover_dampener": _safe_float(rc.get("memory_feedback_turnover_dampener"), None),
            "block_new_entries": bool(rc.get("memory_feedback_block_new_entries", False)),
            "reasons": _safe_list(rc.get("memory_feedback_reasons", [])),
        }
    elif ext_mf:
        source = "overlay_runtime_context"
        data = {
            "active": bool(ext_mf.get("active", False)),
            "status": str(ext_mf.get("status", "unknown")).strip().lower() or "unknown",
            "risk_scale": _safe_float(ext_mf.get("risk_scale"), None),
            "max_trades_scale": _safe_float(ext_mf.get("max_trades_scale"), None),
            "max_open_scale": _safe_float(ext_mf.get("max_open_scale"), None),
            "turnover_pressure": _safe_float(ext_mf.get("turnover_pressure"), None),
            "turnover_dampener": _safe_float(ext_mf.get("turnover_dampener"), None),
            "block_new_entries": bool(ext_mf.get("block_new_entries", False)),
            "reasons": _safe_list(ext_mf.get("reasons", [])),
        }
    else:
        source = "none"
        data = {
            "active": False,
            "status": "unknown",
            "risk_scale": None,
            "max_trades_scale": None,
            "max_open_scale": None,
            "turnover_pressure": None,
            "turnover_dampener": None,
            "block_new_entries": False,
            "reasons": [],
        }

    status = str(data.get("status", "unknown")).strip().lower() or "unknown"
    active = bool(data.get("active", False))
    risk_scale = data.get("risk_scale")
    pressure = data.get("turnover_pressure")
    dampener = data.get("turnover_dampener")

    if not active:
        state = "inactive"
    elif status in {"alert", "hard"}:
        state = "alert"
    elif status == "warn":
        state = "warn"
    elif risk_scale is not None and risk_scale <= 0.84:
        state = "alert"
    elif risk_scale is not None and risk_scale <= 0.95:
        state = "warn"
    elif pressure is not None and pressure >= 0.72:
        state = "alert"
    elif pressure is not None and pressure >= 0.45:
        state = "warn"
    elif dampener is not None and dampener >= 0.10:
        state = "alert"
    elif dampener is not None and dampener >= 0.05:
        state = "warn"
    elif status == "ok":
        state = "ok"
    else:
        state = "unknown"

    severity = {"inactive": 0, "unknown": 1, "ok": 1, "warn": 2, "alert": 3}.get(state, 1)
    return {
        **data,
        "state": state,
        "severity": int(severity),
        "source": source,
        "present": source != "none",
    }


def memory_outbox_runtime_info(
    runtime_controls: dict | None,
    *,
    warn_files: int = 5,
    alert_files: int = 20,
) -> dict:
    rc = runtime_controls if isinstance(runtime_controls, dict) else {}
    enabled = bool(rc.get("memory_replay_enabled", False))
    source = "runtime_controls" if any(k.startswith("memory_replay_") for k in rc.keys()) else "none"

    data = {
        "enabled": enabled,
        "interval_sec": _safe_float(rc.get("memory_replay_interval_sec"), None),
        "last_ts_utc": str(rc.get("memory_replay_last_ts_utc", "")).strip() or None,
        "last_ok": rc.get("memory_replay_last_ok", None),
        "last_error": str(rc.get("memory_replay_last_error", "")).strip() or None,
        "last_replayed": max(0, _safe_int(rc.get("memory_replay_last_replayed", 0), 0)),
        "last_failed": max(0, _safe_int(rc.get("memory_replay_last_failed", 0), 0)),
        "last_processed_files": max(0, _safe_int(rc.get("memory_replay_last_processed_files", 0), 0)),
        "queued_files": _safe_int(rc.get("memory_replay_queued_files"), 0) if rc.get("memory_replay_queued_files") is not None else None,
        "remaining_files": _safe_int(rc.get("memory_replay_remaining_files"), 0)
        if rc.get("memory_replay_remaining_files") is not None
        else None,
    }

    if not enabled:
        state = "inactive"
    else:
        queued = data.get("queued_files")
        remaining = data.get("remaining_files")
        last_ok = data.get("last_ok")
        last_failed = int(data.get("last_failed") or 0)
        err = str(data.get("last_error") or "").strip().lower()
        backlog = max(
            int(queued) if isinstance(queued, int) else 0,
            int(remaining) if isinstance(remaining, int) else 0,
        )
        warn_n = max(1, int(warn_files))
        alert_n = max(warn_n + 1, int(alert_files))

        if last_failed > 0:
            state = "alert"
        elif backlog >= alert_n:
            state = "alert"
        elif bool(last_ok is False) and ("unreachable" in err or "health_status" in err):
            state = "alert"
        elif backlog >= warn_n:
            state = "warn"
        elif bool(last_ok is False) and ("cooldown_active" in err):
            state = "warn"
        elif bool(last_ok is False):
            state = "warn"
        elif last_ok is True:
            state = "ok"
        else:
            state = "unknown"

    severity = {"inactive": 0, "unknown": 1, "ok": 1, "warn": 2, "alert": 3}.get(state, 1)
    return {
        **data,
        "state": state,
        "severity": int(severity),
        "source": source,
        "present": source != "none",
    }
