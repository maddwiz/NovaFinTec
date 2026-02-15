from __future__ import annotations

import json
import math
from pathlib import Path


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


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


def load_external_signal_bundle(path: Path, min_confidence: float = 0.55, max_bias: float = 0.90) -> dict:
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

    if isinstance(payload, dict):
        ctx = payload.get("runtime_context", {})
        if isinstance(ctx, dict):
            out["runtime_multiplier"] = _clamp(_safe_float(ctx.get("runtime_multiplier", 1.0), 1.0), 0.20, 1.20)
            regime = str(ctx.get("regime", "unknown")).strip().lower()
            out["regime"] = regime or "unknown"
            flags = ctx.get("risk_flags", [])
            if isinstance(flags, list):
                out["risk_flags"] = [str(x).strip().lower() for x in flags if str(x).strip()]

        out["degraded_safe_mode"] = bool(payload.get("degraded_safe_mode", False))
        out["source_mode"] = str(payload.get("source_mode", "unknown")).strip() or "unknown"
        qg = payload.get("quality_gate", {})
        if isinstance(qg, dict):
            out["quality_gate_ok"] = bool(qg.get("ok", True))

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
    flags = bundle.get("risk_flags", [])
    flags = [str(x).strip().lower() for x in flags] if isinstance(flags, list) else []
    if degraded:
        scale *= float(_clamp(degraded_scale, 0.20, 1.20))
    if not q_ok:
        scale *= float(_clamp(quality_fail_scale, 0.20, 1.20))
    if flags:
        scale *= float(_clamp(flag_scale, 0.20, 1.20)) ** len(flags)
    scale = _clamp(scale, float(min_scale), float(max_scale))
    diag = {
        "active": bool((degraded or (not q_ok) or bool(flags) or abs(scale - 1.0) > 1e-6)),
        "flags": flags,
        "degraded": degraded,
        "quality_gate_ok": q_ok,
        "runtime_multiplier": _safe_float(bundle.get("runtime_multiplier", 1.0), 1.0),
        "regime": str(bundle.get("regime", "unknown")),
        "source_mode": str(bundle.get("source_mode", "unknown")),
    }
    return scale, diag
