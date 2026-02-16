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
        ]
    )
    if rc_has:
        source = "runtime_controls"
        data = {
            "active": bool(rc.get("aion_feedback_active", False)),
            "status": str(rc.get("aion_feedback_status", "unknown")).strip().lower() or "unknown",
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
        data = {
            "active": bool(ext_af.get("active", False)),
            "status": str(ext_af.get("status", "unknown")).strip().lower() or "unknown",
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
