from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _uniq_str_flags(flags) -> list[str]:
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


def normalize_source_tag(raw: str | None, default: str = "unknown") -> str:
    tag = str(raw or "").strip().lower() or str(default)
    if tag == "shadow":
        return "shadow_trades"
    return tag


def normalize_source_preference(raw: str | None) -> str:
    tag = str(raw or "").strip().lower() or "auto"
    if tag in {"auto", "overlay", "shadow"}:
        return tag
    return "auto"


def feedback_lineage(
    payload: dict | None,
    *,
    selected_source: str | None = None,
    source_preference: str | None = None,
    default_source: str = "unknown",
) -> dict:
    af = payload if isinstance(payload, dict) else {}
    selected = normalize_source_tag(
        selected_source if selected_source is not None else af.get("source_selected", af.get("source", default_source)),
        default=default_source,
    )
    reported = normalize_source_tag(af.get("source", selected), default=selected)
    pref = normalize_source_preference(
        source_preference if source_preference is not None else af.get("source_preference", "auto")
    )
    return {
        "source": reported,
        "source_selected": selected,
        "source_preference": pref,
    }


def lineage_quality(lineage: dict | None) -> tuple[float | None, dict]:
    if not isinstance(lineage, dict):
        return None, {
            "available": False,
            "source": "unknown",
            "source_selected": "unknown",
            "source_preference": "auto",
            "issues": [],
        }

    source = normalize_source_tag(lineage.get("source"), default="unknown")
    selected = normalize_source_tag(lineage.get("source_selected", source), default=source)
    pref = normalize_source_preference(lineage.get("source_preference", "auto"))

    score = 1.0
    issues: list[str] = []
    if source in {"unknown", "none"}:
        score *= 0.82
        issues.append("reported_source_missing")
    if selected in {"unknown", "none"}:
        score *= 0.78
        issues.append("selected_source_missing")

    if source not in {"unknown", "none"} and selected not in {"unknown", "none"} and source != selected:
        score *= 0.86
        issues.append("reported_selected_mismatch")

    expected_selected = None
    if pref == "overlay":
        expected_selected = "overlay"
    elif pref == "shadow":
        expected_selected = "shadow_trades"
    if expected_selected and selected != expected_selected:
        score *= 0.92
        issues.append("preference_fallback")

    score = float(np.clip(score, 0.45, 1.0))
    return score, {
        "available": True,
        "source": source,
        "source_selected": selected,
        "source_preference": pref,
        "issues": issues,
    }


def resolve_shadow_trades_path(root: Path | None = None) -> Path:
    env_path = str(os.getenv("Q_AION_SHADOW_TRADES", "")).strip()
    if env_path:
        return Path(env_path)
    q_root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    aion_home = str(os.getenv("Q_AION_HOME", str(q_root.parent / "aion"))).strip()
    return Path(aion_home) / "logs" / "shadow_trades.csv"


def load_outcome_feedback(
    *,
    root: Path | None = None,
    lookback: int | None = None,
    min_trades: int | None = None,
    max_age_hours: float | None = None,
    mark_stale_reason: bool = True,
) -> dict:
    p = resolve_shadow_trades_path(root=root)
    out = {
        "active": False,
        "status": "missing",
        "risk_scale": 1.0,
        "reasons": [],
        "path": str(p),
        "last_closed_ts": None,
        "age_hours": None,
        "max_age_hours": None,
        "stale": False,
    }
    if (not p.exists()) or p.is_dir():
        return out

    try:
        df = pd.read_csv(p)
    except Exception:
        out["status"] = "read_error"
        out["reasons"] = ["read_error"]
        return out
    if df.empty or ("side" not in df.columns) or ("pnl" not in df.columns):
        out["status"] = "schema_missing"
        out["reasons"] = ["schema_missing"]
        return out

    lb = int(np.clip(_safe_float(lookback if lookback is not None else os.getenv("Q_AION_FEEDBACK_LOOKBACK", 30), 30), 5, 400))
    min_n = int(np.clip(_safe_float(min_trades if min_trades is not None else os.getenv("Q_AION_FEEDBACK_MIN_TRADES", 8), 8), 3, 100))
    closed = df.copy()
    side = closed["side"].astype(str).str.upper()
    closed = closed[side.str.contains("EXIT") | side.str.contains("PARTIAL")]
    if closed.empty:
        out["status"] = "no_closed_trades"
        out["reasons"] = ["no_closed_trades"]
        return out

    if "timestamp" in closed.columns:
        try:
            closed = closed.assign(_ts=pd.to_datetime(closed["timestamp"], errors="coerce")).sort_values("_ts")
        except Exception:
            pass
    pnl = pd.to_numeric(closed["pnl"], errors="coerce").fillna(0.0).values.astype(float)
    pnl = pnl[-lb:]
    n = int(len(pnl))
    if n <= 0:
        out["status"] = "no_closed_trades"
        out["reasons"] = ["no_closed_trades"]
        return out

    wins = int(np.sum(pnl > 0.0))
    losses = int(np.sum(pnl < 0.0))
    hit = float(wins / max(1, wins + losses))
    gross_win = float(np.sum(pnl[pnl > 0.0])) if wins else 0.0
    gross_loss = float(np.abs(np.sum(pnl[pnl < 0.0]))) if losses else 0.0
    pf = float(gross_win / max(1e-9, gross_loss)) if gross_loss > 1e-9 else (2.5 if gross_win > 0 else 1.0)
    expectancy = float(np.mean(pnl))
    abs_mean = float(np.mean(np.abs(pnl))) if n else 0.0
    exp_norm = float(expectancy / max(1e-9, abs_mean)) if abs_mean > 0 else 0.0
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(eq)
    max_dd = float(np.max(peak - eq)) if len(eq) else 0.0
    dd_norm = float(max_dd / max(1e-9, abs_mean * max(1.0, float(np.sqrt(n))))) if abs_mean > 0 else 0.0

    last_closed_ts = None
    age_hours = None
    if "_ts" in closed.columns:
        try:
            ts = pd.to_datetime(closed["_ts"], errors="coerce").dropna()
            if len(ts):
                last_dt = pd.Timestamp(ts.iloc[-1])
                if last_dt.tzinfo is None:
                    last_dt = last_dt.tz_localize("UTC")
                else:
                    last_dt = last_dt.tz_convert("UTC")
                now_dt = pd.Timestamp.now(tz="UTC")
                last_closed_ts = last_dt.isoformat()
                age_hours = float(max(0.0, (now_dt - last_dt).total_seconds() / 3600.0))
        except Exception:
            last_closed_ts = None
            age_hours = None

    risk_scale = 1.0
    reasons = []
    if n >= min_n:
        if hit < 0.36:
            risk_scale *= 0.78
            reasons.append("low_hit_rate_alert")
        elif hit < 0.42:
            risk_scale *= 0.90
            reasons.append("low_hit_rate_warn")
        if pf < 0.75:
            risk_scale *= 0.74
            reasons.append("low_profit_factor_alert")
        elif pf < 0.95:
            risk_scale *= 0.88
            reasons.append("low_profit_factor_warn")
        if exp_norm < -0.30:
            risk_scale *= 0.80
            reasons.append("negative_expectancy_alert")
        elif exp_norm < -0.15:
            risk_scale *= 0.92
            reasons.append("negative_expectancy_warn")
        if dd_norm > 3.0:
            risk_scale *= 0.82
            reasons.append("drawdown_pressure_alert")
        elif dd_norm > 2.0:
            risk_scale *= 0.92
            reasons.append("drawdown_pressure_warn")
        status = "ok"
    else:
        status = "insufficient"

    risk_scale = float(_clamp(risk_scale, 0.65, 1.05))
    if n >= min_n and risk_scale <= 0.82:
        status = "alert"
    elif n >= min_n and risk_scale <= 0.94:
        status = "warn"

    max_age = max_age_hours
    if max_age is None:
        max_age = _safe_float(os.getenv("Q_AION_FEEDBACK_MAX_AGE_HOURS", 72.0), 72.0)
    max_age = float(np.clip(float(max_age), 1.0, 24.0 * 90.0))
    stale = bool(age_hours is not None and age_hours > max_age)
    if stale and mark_stale_reason:
        reasons.append("stale_feedback")

    return {
        "active": True,
        "status": status,
        "path": str(p),
        "closed_trades": int(n),
        "hit_rate": float(hit),
        "profit_factor": float(pf),
        "expectancy": float(expectancy),
        "expectancy_norm": float(exp_norm),
        "max_drawdown": float(max_dd),
        "drawdown_norm": float(dd_norm),
        "risk_scale": float(risk_scale),
        "last_closed_ts": last_closed_ts,
        "age_hours": age_hours,
        "max_age_hours": max_age,
        "stale": stale,
        "reasons": _uniq_str_flags(reasons),
    }


def feedback_has_metrics(payload: dict | None) -> bool:
    if not isinstance(payload, dict):
        return False
    if bool(payload.get("active", False)):
        return True
    status = str(payload.get("status", "unknown")).strip().lower()
    if status not in {"", "unknown", "missing"}:
        return True
    closed = int(max(0, _safe_float(payload.get("closed_trades", 0), 0)))
    if closed > 0:
        return True
    if bool(payload.get("stale", False)):
        return True
    risk_scale = _safe_float(payload.get("risk_scale", np.nan), default=np.nan)
    if math.isfinite(risk_scale) and abs(float(risk_scale) - 1.0) > 1e-9:
        return True
    for key in ["hit_rate", "profit_factor", "expectancy", "drawdown_norm", "age_hours", "max_age_hours"]:
        v = _safe_float(payload.get(key, np.nan), default=np.nan)
        if math.isfinite(v):
            return True
    return False


def feedback_is_stale(payload: dict | None, *, default_max_age_hours: float | None = None) -> bool:
    if not isinstance(payload, dict):
        return False
    stale_flag = bool(payload.get("stale", False))
    age_hours = _safe_float(payload.get("age_hours", np.nan), default=np.nan)
    max_age_hours = _safe_float(payload.get("max_age_hours", np.nan), default=np.nan)
    if not math.isfinite(max_age_hours):
        if default_max_age_hours is None:
            default_max_age_hours = _safe_float(os.getenv("Q_AION_FEEDBACK_MAX_AGE_HOURS", 72.0), 72.0)
        max_age_hours = float(default_max_age_hours)
    max_age_hours = float(np.clip(float(max_age_hours), 1.0, 24.0 * 90.0))
    age_stale = bool(math.isfinite(age_hours) and age_hours > max_age_hours)
    return bool(stale_flag or age_stale)


def choose_feedback_source(
    overlay_feedback: dict | None,
    shadow_feedback: dict | None,
    *,
    source_pref: str = "auto",
    prefer_overlay_when_fresh: bool = False,
) -> tuple[dict, str]:
    pref = normalize_source_preference(source_pref)
    overlay_has = feedback_has_metrics(overlay_feedback)
    shadow_has = feedback_has_metrics(shadow_feedback)
    overlay_stale = feedback_is_stale(overlay_feedback)
    shadow_stale = feedback_is_stale(shadow_feedback)

    if pref == "overlay":
        if overlay_has:
            return dict(overlay_feedback), "overlay"
        if shadow_has:
            return dict(shadow_feedback), "shadow_trades"
    elif pref == "shadow":
        if shadow_has:
            return dict(shadow_feedback), "shadow_trades"
        if overlay_has:
            return dict(overlay_feedback), "overlay"
    else:
        if prefer_overlay_when_fresh:
            if overlay_has and (not overlay_stale):
                return dict(overlay_feedback), "overlay"
            if shadow_has and (not shadow_stale):
                return dict(shadow_feedback), "shadow_trades"
            if overlay_has:
                return dict(overlay_feedback), "overlay"
            if shadow_has:
                return dict(shadow_feedback), "shadow_trades"
        else:
            if shadow_has and (not shadow_stale):
                return dict(shadow_feedback), "shadow_trades"
            if overlay_has and (not overlay_stale):
                return dict(overlay_feedback), "overlay"
            if shadow_has:
                return dict(shadow_feedback), "shadow_trades"
            if overlay_has:
                return dict(overlay_feedback), "overlay"

    if isinstance(shadow_feedback, dict):
        return dict(shadow_feedback), "shadow_trades"
    if isinstance(overlay_feedback, dict):
        return dict(overlay_feedback), "overlay"
    return {}, "none"
