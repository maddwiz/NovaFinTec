from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _uniq_flags(flags) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    if not isinstance(flags, list):
        return out
    for raw in flags:
        key = str(raw).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _canonicalize_flags(flags) -> list[str]:
    out = _uniq_flags(flags)
    stronger_to_weaker = [
        ("drift_alert", "drift_warn"),
        ("fracture_alert", "fracture_warn"),
        ("exec_risk_hard", "exec_risk_tight"),
        ("nested_leakage_alert", "nested_leakage_warn"),
        ("hive_stress_alert", "hive_stress_warn"),
        ("heartbeat_alert", "heartbeat_warn"),
        ("council_divergence_alert", "council_divergence_warn"),
        ("memory_feedback_alert", "memory_feedback_warn"),
    ]
    s = set(out)
    for strong, weak in stronger_to_weaker:
        if strong in s and weak in s:
            s.discard(weak)
    return [x for x in out if x in s]


def _parse_utc_timestamp(raw) -> datetime | None:
    s = str(raw or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalize_signal(payload: dict, min_confidence: float, max_bias: float):
    bias = _safe_float(payload.get("bias"), 0.0)
    conf = _safe_float(payload.get("confidence"), 0.0)
    conf = _clamp(conf, 0.0, 1.0)
    if conf < float(min_confidence):
        return None
    bias = _clamp(bias, -abs(float(max_bias)), abs(float(max_bias)))
    return {"bias": bias, "confidence": conf}


def _as_signal_map_from_payload(payload, min_confidence: float, max_bias: float) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if isinstance(payload, dict):
        global_sig = payload.get("global")
        if isinstance(global_sig, dict):
            g = _normalize_signal(global_sig, min_confidence=min_confidence, max_bias=max_bias)
            if g is not None:
                out["__GLOBAL__"] = g

        signals = payload.get("signals")
        if isinstance(signals, dict):
            for sym, sig in signals.items():
                if not isinstance(sig, dict):
                    continue
                n = _normalize_signal(sig, min_confidence=min_confidence, max_bias=max_bias)
                if n is None:
                    continue
                key = str(sym).strip().upper()
                if key:
                    out[key] = n

    # Fallback format: list of rows
    # [{"symbol":"SPY","bias":0.2,"confidence":0.7}, ...]
    if not out and isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            key = str(row.get("symbol", "")).strip().upper()
            if not key:
                continue
            n = _normalize_signal(row, min_confidence=min_confidence, max_bias=max_bias)
            if n is None:
                continue
            out[key] = n

    return out


def load_external_signal_bundle(
    path: Path,
    min_confidence: float = 0.55,
    max_bias: float = 0.90,
    max_age_hours: float | None = None,
) -> dict:
    """
    Load external signals plus optional runtime context metadata.

    Returns:
      {
        "signals": dict[str, {"bias","confidence"}],
        "runtime_multiplier": float,
        "risk_flags": list[str],
        "regime": str,
        "degraded_safe_mode": bool,
        "quality_gate_ok": bool,
        "overlay_age_hours": float|None,
        "overlay_age_source": str,
        "overlay_generated_at_utc": str|None,
        "overlay_stale": bool,
        "memory_feedback": dict,
      }
    """
    empty = {
        "signals": {},
        "runtime_multiplier": 1.0,
        "risk_flags": [],
        "regime": "unknown",
        "source_mode": "unknown",
        "degraded_safe_mode": False,
        "quality_gate_ok": True,
        "overlay_age_hours": None,
        "overlay_age_source": "unknown",
        "overlay_generated_at_utc": None,
        "overlay_stale": False,
        "memory_feedback": {
            "active": False,
            "status": "unknown",
            "risk_scale": 1.0,
            "max_trades_scale": 1.0,
            "max_open_scale": 1.0,
            "block_new_entries": False,
            "reasons": [],
        },
    }
    try:
        p = Path(path)
    except Exception:
        return dict(empty)
    if not p.exists():
        return dict(empty)

    try:
        payload = json.loads(p.read_text())
    except Exception:
        return dict(empty)

    out = dict(empty)
    out["signals"] = _as_signal_map_from_payload(payload, min_confidence=min_confidence, max_bias=max_bias)
    age_mtime_h = None
    try:
        age_mtime_h = max(0.0, (time.time() - float(p.stat().st_mtime)) / 3600.0)
    except Exception:
        age_mtime_h = None

    if isinstance(payload, dict):
        gen_raw = payload.get("generated_at_utc", payload.get("generated_at"))
        gen_dt = _parse_utc_timestamp(gen_raw)
        if gen_dt is not None:
            out["overlay_generated_at_utc"] = gen_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            now_utc = datetime.now(timezone.utc)
            age_payload_h = max(0.0, (now_utc - gen_dt).total_seconds() / 3600.0)
            out["overlay_age_hours"] = float(age_payload_h)
            out["overlay_age_source"] = "payload"
        elif isinstance(age_mtime_h, float):
            out["overlay_age_hours"] = float(age_mtime_h)
            out["overlay_age_source"] = "mtime"
        else:
            out["overlay_age_hours"] = None
            out["overlay_age_source"] = "unknown"

        ctx = payload.get("runtime_context", {})
        if isinstance(ctx, dict):
            out["runtime_multiplier"] = _clamp(_safe_float(ctx.get("runtime_multiplier", 1.0), 1.0), 0.20, 1.20)
            regime = str(ctx.get("regime", "unknown")).strip().lower()
            out["regime"] = regime or "unknown"
            flags = ctx.get("risk_flags", [])
            out["risk_flags"] = _canonicalize_flags(flags)
            mf = ctx.get("memory_feedback", {})
            if isinstance(mf, dict):
                out["memory_feedback"] = {
                    "active": bool(mf.get("active", False)),
                    "status": str(mf.get("status", "unknown")).strip().lower() or "unknown",
                    "risk_scale": _clamp(_safe_float(mf.get("risk_scale", 1.0), 1.0), 0.20, 1.20),
                    "max_trades_scale": _clamp(_safe_float(mf.get("max_trades_scale", 1.0), 1.0), 0.20, 1.20),
                    "max_open_scale": _clamp(_safe_float(mf.get("max_open_scale", 1.0), 1.0), 0.20, 1.20),
                    "block_new_entries": bool(mf.get("block_new_entries", False)),
                    "reasons": _uniq_flags(mf.get("reasons", [])),
                }

        out["degraded_safe_mode"] = bool(payload.get("degraded_safe_mode", False))
        out["source_mode"] = str(payload.get("source_mode", "unknown")).strip() or "unknown"
        qg = payload.get("quality_gate", {})
        if isinstance(qg, dict):
            out["quality_gate_ok"] = bool(qg.get("ok", True))
    elif isinstance(age_mtime_h, float):
        out["overlay_age_hours"] = float(age_mtime_h)
        out["overlay_age_source"] = "mtime"

    if max_age_hours is not None and float(max_age_hours) > 0 and isinstance(out.get("overlay_age_hours"), float):
        if float(out["overlay_age_hours"]) > float(max_age_hours):
            out["overlay_stale"] = True
            out["signals"] = {}
            out["risk_flags"] = _canonicalize_flags(list(out.get("risk_flags", [])) + ["overlay_stale"])

    return out


def blend_external_signals(primary: dict | None, secondary: dict | None, max_bias: float = 0.90) -> dict | None:
    if not primary and not secondary:
        return None
    if not primary:
        return _normalize_signal(secondary or {}, min_confidence=0.0, max_bias=max_bias)
    if not secondary:
        return _normalize_signal(primary or {}, min_confidence=0.0, max_bias=max_bias)

    p = _normalize_signal(primary, min_confidence=0.0, max_bias=max_bias) or {"bias": 0.0, "confidence": 0.0}
    s = _normalize_signal(secondary, min_confidence=0.0, max_bias=max_bias) or {"bias": 0.0, "confidence": 0.0}

    wp = max(0.0, _safe_float(p.get("confidence"), 0.0))
    ws = max(0.0, _safe_float(s.get("confidence"), 0.0))
    wsum = wp + ws
    if wsum <= 1e-12:
        wp = 1.0
        ws = 1.0
        wsum = 2.0

    bias = (wp * _safe_float(p.get("bias"), 0.0) + ws * _safe_float(s.get("bias"), 0.0)) / wsum
    conf = max(_safe_float(p.get("confidence"), 0.0), _safe_float(s.get("confidence"), 0.0))
    return {"bias": _clamp(bias, -abs(float(max_bias)), abs(float(max_bias))), "confidence": _clamp(conf, 0.0, 1.0)}


def load_external_signal_map(path: Path, min_confidence: float = 0.55, max_bias: float = 0.90) -> dict[str, dict]:
    return load_external_signal_bundle(path, min_confidence=min_confidence, max_bias=max_bias).get("signals", {})


def runtime_overlay_scale(
    bundle: dict | None,
    min_scale: float = 0.55,
    max_scale: float = 1.05,
    degraded_scale: float = 0.70,
    quality_fail_scale: float = 0.82,
    flag_scale: float = 0.90,
    drift_warn_scale: float = 0.94,
    drift_alert_scale: float = 0.82,
    quality_step_spike_scale: float = 0.86,
    fracture_warn_scale: float = 0.88,
    fracture_alert_scale: float = 0.75,
    exec_risk_tight_scale: float = 0.84,
    exec_risk_hard_scale: float = 0.68,
    nested_leak_warn_scale: float = 0.90,
    nested_leak_alert_scale: float = 0.76,
    hive_stress_warn_scale: float = 0.90,
    hive_stress_alert_scale: float = 0.74,
    heartbeat_warn_scale: float = 0.88,
    heartbeat_alert_scale: float = 0.72,
    council_divergence_warn_scale: float = 0.90,
    council_divergence_alert_scale: float = 0.74,
    overlay_stale_scale: float = 0.82,
):
    """
    Convert Q runtime context into an AION overlay-strength scalar.
    Returns (scale, diagnostics_dict).
    """
    if not isinstance(bundle, dict):
        return 1.0, {"active": False, "flags": [], "degraded": False, "quality_gate_ok": True, "source_mode": "unknown"}

    scale = _clamp(_safe_float(bundle.get("runtime_multiplier", 1.0), 1.0), 0.20, 1.20)
    degraded = bool(bundle.get("degraded_safe_mode", False))
    q_ok = bool(bundle.get("quality_gate_ok", True))
    overlay_stale = bool(bundle.get("overlay_stale", False))
    flags = bundle.get("risk_flags", [])
    flags = _canonicalize_flags(flags)
    if degraded:
        scale *= float(_clamp(degraded_scale, 0.20, 1.20))
    if not q_ok:
        scale *= float(_clamp(quality_fail_scale, 0.20, 1.20))
    if overlay_stale:
        scale *= float(_clamp(overlay_stale_scale, 0.20, 1.20))
    if flags:
        scale *= float(_clamp(flag_scale, 0.20, 1.20)) ** len(flags)
        # Drift-quality warnings from Q governance.
        if "drift_alert" in flags:
            scale *= float(_clamp(drift_alert_scale, 0.20, 1.20))
        elif "drift_warn" in flags:
            scale *= float(_clamp(drift_warn_scale, 0.20, 1.20))
        if "quality_governor_step_spike" in flags:
            scale *= float(_clamp(quality_step_spike_scale, 0.20, 1.20))
        # Severity-aware fracture penalties from Q risk_flags.
        if "fracture_alert" in flags:
            scale *= float(_clamp(fracture_alert_scale, 0.20, 1.20))
        elif "fracture_warn" in flags:
            scale *= float(_clamp(fracture_warn_scale, 0.20, 1.20))
        # Execution-risk penalties exported by Q execution constraints.
        if "exec_risk_hard" in flags:
            scale *= float(_clamp(exec_risk_hard_scale, 0.20, 1.20))
        elif "exec_risk_tight" in flags:
            scale *= float(_clamp(exec_risk_tight_scale, 0.20, 1.20))
        if "nested_leakage_alert" in flags:
            scale *= float(_clamp(nested_leak_alert_scale, 0.20, 1.20))
        elif "nested_leakage_warn" in flags:
            scale *= float(_clamp(nested_leak_warn_scale, 0.20, 1.20))
        if "hive_stress_alert" in flags:
            scale *= float(_clamp(hive_stress_alert_scale, 0.20, 1.20))
        elif "hive_stress_warn" in flags:
            scale *= float(_clamp(hive_stress_warn_scale, 0.20, 1.20))
        if "heartbeat_alert" in flags:
            scale *= float(_clamp(heartbeat_alert_scale, 0.20, 1.20))
        elif "heartbeat_warn" in flags:
            scale *= float(_clamp(heartbeat_warn_scale, 0.20, 1.20))
        if "council_divergence_alert" in flags:
            scale *= float(_clamp(council_divergence_alert_scale, 0.20, 1.20))
        elif "council_divergence_warn" in flags:
            scale *= float(_clamp(council_divergence_warn_scale, 0.20, 1.20))
    scale = _clamp(scale, float(min_scale), float(max_scale))
    diag = {
        "active": bool((degraded or (not q_ok) or bool(flags) or abs(scale - 1.0) > 1e-6)),
        "flags": flags,
        "degraded": degraded,
        "overlay_stale": overlay_stale,
        "overlay_age_hours": bundle.get("overlay_age_hours", None),
        "overlay_age_source": bundle.get("overlay_age_source", "unknown"),
        "overlay_generated_at_utc": bundle.get("overlay_generated_at_utc", None),
        "quality_gate_ok": q_ok,
        "runtime_multiplier": _safe_float(bundle.get("runtime_multiplier", 1.0), 1.0),
        "regime": str(bundle.get("regime", "unknown")),
        "source_mode": str(bundle.get("source_mode", "unknown")),
        "memory_feedback": bundle.get("memory_feedback", {}),
    }
    return scale, diag
