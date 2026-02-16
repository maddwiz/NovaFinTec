#!/usr/bin/env python3
"""
Export a compact signal pack for AION consumption.

Primary input: runs_plus/walk_forward_table_plus.csv
Primary output: runs_plus/q_signal_overlay.json
Optional mirror: a second JSON path (e.g. AION state folder).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmods.aion_feedback import (  # noqa: E402
    feedback_lineage,
    load_outcome_feedback,
    normalize_source_preference,
)

RUNS = ROOT / "runs_plus"


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


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


def _canonicalize_risk_flags(flags) -> list[str]:
    out = _uniq_str_flags(flags)
    stronger_to_weaker = [
        ("drift_alert", "drift_warn"),
        ("fracture_alert", "fracture_warn"),
        ("exec_risk_hard", "exec_risk_tight"),
        ("nested_leakage_alert", "nested_leakage_warn"),
        ("hive_stress_alert", "hive_stress_warn"),
        ("hive_crowding_alert", "hive_crowding_warn"),
        ("hive_entropy_alert", "hive_entropy_warn"),
        ("hive_turnover_alert", "hive_turnover_warn"),
        ("aion_outcome_alert", "aion_outcome_warn"),
        ("heartbeat_alert", "heartbeat_warn"),
        ("council_divergence_alert", "council_divergence_warn"),
        ("memory_feedback_alert", "memory_feedback_warn"),
        ("memory_turnover_alert", "memory_turnover_warn"),
    ]
    s = set(out)
    for strong, weak in stronger_to_weaker:
        if strong in s and weak in s:
            s.discard(weak)
    return [x for x in out if x in s]


def _canonical_symbol(sym: str) -> str:
    s = str(sym or "").strip().upper()
    if not s:
        return ""
    if "_" in s:
        s = s.split("_", 1)[0]
    return s


def _build_scores(df: pd.DataFrame) -> pd.DataFrame:
    w = df.copy()
    for col in ["sharpe", "hit", "meta_weight", "council_weight"]:
        if col not in w.columns:
            w[col] = 0.0
        w[col] = pd.to_numeric(w[col], errors="coerce").fillna(0.0)

    # Forecast direction preference from model-side voting.
    vote = w["meta_weight"] + w["council_weight"]
    sign_vote = np.sign(vote)
    sign_fallback = np.sign(w["sharpe"])
    direction = np.where(sign_vote != 0.0, sign_vote, sign_fallback)

    sharpe_q = np.tanh(w["sharpe"] / 2.0)
    hit_q = np.clip((w["hit"] - 0.5) / 0.2, -1.0, 1.0)
    quality = 0.65 * sharpe_q + 0.35 * hit_q

    # Bias is directional and bounded to avoid oversized external influence.
    bias = np.clip(direction * np.abs(quality), -1.0, 1.0)
    confidence = np.clip(0.50 + 0.50 * np.abs(quality), 0.0, 1.0)

    out = pd.DataFrame(
        {
            "symbol": w["asset"].astype(str).str.upper(),
            "bias": bias.astype(float),
            "confidence": confidence.astype(float),
            "sharpe": w["sharpe"].astype(float),
            "hit": w["hit"].astype(float),
        }
    )
    out["rank"] = out["confidence"] * out["bias"].abs()
    return out

def _build_scores_from_final_weights(weights_path: Path) -> pd.DataFrame:
    if not weights_path.exists():
        return pd.DataFrame(columns=["symbol", "bias", "confidence", "sharpe", "hit", "rank"])
    try:
        w = np.loadtxt(weights_path, delimiter=",")
    except Exception:
        try:
            w = np.loadtxt(weights_path, delimiter=",", skiprows=1)
        except Exception:
            return pd.DataFrame(columns=["symbol", "bias", "confidence", "sharpe", "hit", "rank"])
    w = np.asarray(w, float)
    if w.ndim == 1:
        row = w.ravel()
    elif w.ndim == 2 and w.shape[0] >= 1:
        row = w[-1].ravel()
    else:
        return pd.DataFrame(columns=["symbol", "bias", "confidence", "sharpe", "hit", "rank"])

    syms = []
    an = RUNS / "asset_names.csv"
    if an.exists():
        try:
            adf = pd.read_csv(an)
            if len(adf.columns) >= 1:
                c0 = adf.columns[0]
                syms = [str(x).upper().strip() for x in adf[c0].tolist() if str(x).strip()]
        except Exception:
            syms = []
    if not syms:
        syms = sorted([p.stem.replace("_prices", "").upper() for p in (ROOT / "data").glob("*.csv") if p.is_file()])
    if not syms:
        syms = [f"ASSET_{i+1}" for i in range(len(row))]
    n = min(len(syms), len(row))
    syms = syms[:n]
    vals = row[:n].astype(float)
    absv = np.abs(vals)
    scale = float(np.nanpercentile(absv, 75)) if np.isfinite(absv).any() else 1.0
    scale = max(scale, 1e-6)
    bias = np.clip(vals / (2.5 * scale), -1.0, 1.0)
    confidence = np.clip(absv / (2.0 * scale), 0.0, 1.0)
    out = pd.DataFrame({"symbol": syms, "bias": bias, "confidence": confidence})
    out["sharpe"] = 0.0
    out["hit"] = 0.5
    out["rank"] = out["confidence"] * out["bias"].abs()
    return out


def _collapse_to_canonical(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()

    rows = []
    scored = scored.copy()
    scored["symbol"] = scored["symbol"].map(_canonical_symbol)
    scored = scored[scored["symbol"] != ""]

    for symbol, g in scored.groupby("symbol"):
        conf = pd.to_numeric(g["confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        w = conf.values.astype(float)
        w_sum = float(w.sum())
        if w_sum <= 1e-12:
            w = np.ones(len(g), dtype=float)
            w_sum = float(w.sum())

        bias_vals = pd.to_numeric(g["bias"], errors="coerce").fillna(0.0).values.astype(float)
        sharpe_vals = pd.to_numeric(g["sharpe"], errors="coerce").fillna(0.0).values.astype(float)
        hit_vals = pd.to_numeric(g["hit"], errors="coerce").fillna(0.5).values.astype(float)

        bias = float(np.dot(w, bias_vals) / w_sum)
        sharpe = float(np.dot(w, sharpe_vals) / w_sum)
        hit = float(np.dot(w, hit_vals) / w_sum)
        confidence = float(conf.max())

        rows.append(
            {
                "symbol": symbol,
                "bias": _clamp(bias, -1.0, 1.0),
                "confidence": _clamp(confidence, 0.0, 1.0),
                "sharpe": sharpe,
                "hit": _clamp(hit, 0.0, 1.0),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rank"] = out["confidence"] * out["bias"].abs()
    return out.sort_values("rank", ascending=False).reset_index(drop=True)


def _load_watchlist(path_value: str) -> set[str]:
    raw = str(path_value or "").strip()
    if not raw:
        return set()
    p = Path(raw)
    if not p.exists() or p.is_dir():
        return set()
    out = set()
    for line in p.read_text().splitlines():
        sym = _canonical_symbol(line)
        if sym:
            out.add(sym)
    return out


def _load_json(path: Path):
    if not path.exists() or path.is_dir():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _hours_since_file(path: Path) -> float | None:
    if not path.exists():
        return None
    ts = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
    return float((dt.datetime.now(dt.timezone.utc) - ts).total_seconds() / 3600.0)


def _load_series(path: Path):
    if not path.exists() or path.is_dir():
        return None
    try:
        a = np.loadtxt(path, delimiter=",")
    except Exception:
        # Many Q artifacts are date-indexed CSVs (DATE,<value>), so fall back
        # to a permissive parser and extract the last numeric column.
        try:
            raw = pd.read_csv(path, header=None)
        except Exception:
            return None
        if raw.empty:
            return None
        num = raw.apply(pd.to_numeric, errors="coerce")
        cols = [c for c in num.columns if num[c].notna().any()]
        if not cols:
            return None
        for col in reversed(cols):
            arr = num[col].to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr):
                return arr.ravel()
        return None
    a = np.asarray(a, float)
    if a.ndim == 2 and a.shape[1] >= 1:
        a = a[:, -1]
    a = a.ravel()
    return a if len(a) else None


def _latest_series_value(path: Path, lo: float, hi: float, default: float = 1.0):
    s = _load_series(path)
    if s is None or len(s) == 0:
        return float(default), False
    v = _safe_float(s[-1], default=default)
    return _clamp(v, lo, hi), True


def _memory_feedback(runs_dir: Path):
    """
    Build closed-loop runtime feedback from NovaSpine recall artifacts.
    """
    nctx = _load_json(runs_dir / "novaspine_context.json") or {}
    nhive = _load_json(runs_dir / "novaspine_hive_feedback.json") or {}
    if not isinstance(nctx, dict):
        nctx = {}
    if not isinstance(nhive, dict):
        nhive = {}

    ctx_status = str(nctx.get("status", "unknown")).strip().lower()
    hive_status = str(nhive.get("status", "unknown")).strip().lower()
    enabled = bool(nctx.get("enabled", False) or nhive.get("enabled", False))
    ctx_res = _safe_float(nctx.get("context_resonance", np.nan), default=np.nan)
    ctx_boost = _safe_float(nctx.get("context_boost", np.nan), default=np.nan)
    hive_boost = _safe_float(nhive.get("global_boost", np.nan), default=np.nan)
    ctx_pressure = _safe_float(nctx.get("turnover_pressure", np.nan), default=np.nan)
    hive_pressure = _safe_float(nhive.get("turnover_pressure", np.nan), default=np.nan)
    ctx_damp = _safe_float(nctx.get("turnover_dampener", np.nan), default=np.nan)
    hive_damp = _safe_float(nhive.get("turnover_dampener", np.nan), default=np.nan)

    reasons = []
    base = 1.0
    have_memory = False
    if math.isfinite(ctx_boost):
        base *= _clamp(ctx_boost, 0.85, 1.15)
        have_memory = True
    if math.isfinite(hive_boost):
        base *= _clamp(hive_boost, 0.85, 1.15)
        have_memory = True
    if math.isfinite(ctx_res):
        base *= _clamp(0.92 + 0.20 * ctx_res, 0.88, 1.08)
        have_memory = True
        if ctx_res < 0.08:
            reasons.append("low_context_resonance")
        elif ctx_res > 0.72:
            reasons.append("high_context_resonance")

    pressure_vals = [v for v in [ctx_pressure, hive_pressure] if math.isfinite(v)]
    damp_vals = [v for v in [ctx_damp, hive_damp] if math.isfinite(v)]
    turnover_pressure = float(np.max(pressure_vals)) if pressure_vals else np.nan
    turnover_dampener = float(np.mean(damp_vals)) if damp_vals else np.nan
    if math.isfinite(turnover_pressure):
        pressure_mod = _clamp(1.02 - 0.22 * max(0.0, turnover_pressure), 0.80, 1.02)
        base *= pressure_mod
        have_memory = True
        if turnover_pressure >= 0.72:
            reasons.append("memory_turnover_pressure_high")
        elif turnover_pressure >= 0.45:
            reasons.append("memory_turnover_pressure_warn")
    if math.isfinite(turnover_dampener):
        damp_mod = _clamp(1.01 - 1.40 * max(0.0, turnover_dampener), 0.82, 1.01)
        base *= damp_mod
        have_memory = True

    status_tokens = {ctx_status, hive_status}
    if any(s in {"unreachable", "error", "failed"} or s.startswith("http_") for s in status_tokens):
        base *= 0.96
        reasons.append("memory_unreachable")
    elif any(s in {"skipped", "disabled", "unknown"} for s in status_tokens):
        reasons.append("memory_partial_or_disabled")

    risk_scale = _clamp(base, 0.78, 1.10)
    trades_scale = _clamp(1.0 + 0.85 * (risk_scale - 1.0), 0.80, 1.08)
    open_scale = _clamp(1.0 + 0.75 * (risk_scale - 1.0), 0.82, 1.06)

    if risk_scale <= 0.84:
        status = "alert"
    elif risk_scale <= 0.95:
        status = "warn"
    elif have_memory:
        status = "ok"
    else:
        status = "neutral"

    block_new = bool(
        have_memory
        and status == "alert"
        and math.isfinite(ctx_res)
        and ctx_res < 0.05
        and (math.isfinite(ctx_boost) and ctx_boost < 0.95)
        and (math.isfinite(hive_boost) and hive_boost < 0.95)
    )

    return {
        "active": bool(have_memory or enabled),
        "enabled": bool(enabled),
        "status": status,
        "context_status": ctx_status or "unknown",
        "hive_status": hive_status or "unknown",
        "context_resonance": (float(ctx_res) if math.isfinite(ctx_res) else None),
        "context_boost": (float(ctx_boost) if math.isfinite(ctx_boost) else None),
        "hive_global_boost": (float(hive_boost) if math.isfinite(hive_boost) else None),
        "turnover_pressure": (float(turnover_pressure) if math.isfinite(turnover_pressure) else None),
        "turnover_dampener": (float(turnover_dampener) if math.isfinite(turnover_dampener) else None),
        "risk_scale": float(risk_scale),
        "max_trades_scale": float(trades_scale),
        "max_open_scale": float(open_scale),
        "block_new_entries": bool(block_new),
        "reasons": _uniq_str_flags(reasons),
    }


def _aion_outcome_feedback():
    """
    Optional closed-loop outcome signal from recent AION realized trades.
    Reads AION shadow_trades.csv and maps realized quality into a risk scalar.
    """
    out = load_outcome_feedback(root=ROOT)
    if not isinstance(out, dict):
        return out
    source_pref = normalize_source_preference(os.getenv("Q_AION_FEEDBACK_SOURCE", "auto"))
    lineage = feedback_lineage(
        out,
        source_preference=source_pref,
        default_source="shadow_trades",
    )
    out.setdefault("source", str(lineage.get("source", "shadow_trades")))
    out.setdefault("source_selected", str(lineage.get("source_selected", "shadow_trades")))
    out.setdefault("source_preference", str(lineage.get("source_preference", source_pref)))
    return out


def _runtime_context(runs_dir: Path):
    comps = {}
    specs = {
        "global_governor": ("global_governor.csv", 0.45, 1.10, 1.0),
        "quality_governor": ("quality_governor.csv", 0.55, 1.15, 1.0),
        "quality_runtime_modifier": ("quality_runtime_modifier.csv", 0.55, 1.15, 1.0),
        "meta_mix_reliability_governor": ("meta_mix_reliability_governor.csv", 0.70, 1.20, 1.0),
        "hive_diversification_governor": ("hive_diversification_governor.csv", 0.80, 1.05, 1.0),
        "hive_persistence_governor": ("hive_persistence_governor.csv", 0.75, 1.06, 1.0),
        "dream_coherence_governor": ("dream_coherence_governor.csv", 0.70, 1.20, 1.0),
        "novaspine_context_boost": ("novaspine_context_boost.csv", 0.85, 1.15, 1.0),
        "novaspine_hive_boost": ("novaspine_hive_boost.csv", 0.85, 1.15, 1.0),
        "heartbeat_exposure_scaler": ("heartbeat_exposure_scaler.csv", 0.40, 1.20, 1.0),
        "dna_stress_governor": ("dna_stress_governor.csv", 0.70, 1.15, 1.0),
        "reflex_health_governor": ("reflex_health_governor.csv", 0.70, 1.15, 1.0),
        "symbolic_governor": ("symbolic_governor.csv", 0.70, 1.15, 1.0),
        "legacy_exposure": ("legacy_exposure.csv", 0.40, 1.30, 1.0),
        "regime_fracture_governor": ("regime_fracture_governor.csv", 0.70, 1.06, 1.0),
    }
    active_vals = []
    for k, (fname, lo, hi, dflt) in specs.items():
        v, found = _latest_series_value(runs_dir / fname, lo=lo, hi=hi, default=dflt)
        comps[k] = {"value": float(v), "found": bool(found)}
        if found:
            active_vals.append(float(v))

    # Derived council-mix diagnostics from adaptive blending artifacts.
    qv, qf = _latest_series_value(runs_dir / "meta_mix_quality.csv", lo=0.0, hi=1.0, default=0.5)
    dsv, dsf = _latest_series_value(runs_dir / "meta_mix_disagreement.csv", lo=0.0, hi=1.0, default=0.0)
    av, af = _latest_series_value(runs_dir / "meta_mix_alpha.csv", lo=0.0, hi=1.0, default=0.5)
    gv, gf = _latest_series_value(runs_dir / "meta_mix_gross.csv", lo=0.05, hi=0.60, default=0.24)
    if qf:
        q_mod = _clamp(0.80 + 0.40 * qv, 0.80, 1.20)
        comps["meta_mix_quality_modifier"] = {"value": float(q_mod), "found": True}
        active_vals.append(float(q_mod))
    if dsf:
        d_mod = _clamp(1.04 - 0.22 * dsv, 0.78, 1.04)
        comps["meta_mix_disagreement_modifier"] = {"value": float(d_mod), "found": True}
        active_vals.append(float(d_mod))
    if af:
        balance = 1.0 - abs(2.0 * av - 1.0)
        a_mod = _clamp(0.90 + 0.14 * balance, 0.90, 1.04)
        comps["meta_mix_alpha_balance_modifier"] = {"value": float(a_mod), "found": True}
        active_vals.append(float(a_mod))
    if gf:
        gn = _clamp((gv - 0.12) / max(1e-9, 0.45 - 0.12), 0.0, 1.0)
        g_mod = _clamp(0.88 + 0.24 * gn, 0.88, 1.12)
        comps["meta_mix_gross_modifier"] = {"value": float(g_mod), "found": True}
        active_vals.append(float(g_mod))

    # Quality governor stability modifier from per-step jump diagnostics.
    qsnap = _load_json(runs_dir / "quality_snapshot.json") or {}
    q_step = _safe_float(qsnap.get("quality_governor_max_abs_step", np.nan), default=np.nan)
    if math.isfinite(q_step):
        q_step_mod = _clamp(1.04 - 1.50 * max(0.0, q_step - 0.01), 0.80, 1.05)
        comps["quality_governor_step_modifier"] = {"value": float(q_step_mod), "found": True}
        active_vals.append(float(q_step_mod))

    # Portfolio drift watch modifier.
    drift_watch = _load_json(runs_dir / "portfolio_drift_watch.json") or {}
    drift = drift_watch.get("drift", {}) if isinstance(drift_watch, dict) else {}
    drift_status = str((drift or {}).get("status", "")).lower()
    latest_over_p95 = _safe_float((drift or {}).get("latest_over_p95", np.nan), default=np.nan)
    if drift_status in {"ok", "warn", "alert"}:
        if drift_status == "alert":
            d_mod = 0.75
        elif drift_status == "warn":
            d_mod = 0.90
        else:
            d_mod = 1.00
        if math.isfinite(latest_over_p95):
            d_mod = float(np.clip(d_mod * _clamp(1.02 - 0.10 * max(0.0, latest_over_p95 - 1.0), 0.80, 1.02), 0.70, 1.02))
        comps["portfolio_drift_modifier"] = {"value": float(d_mod), "found": True}
        active_vals.append(float(d_mod))

    # Execution constraints adaptive-risk scalar (from fracture + quality feedback).
    exec_info = _load_json(runs_dir / "execution_constraints_info.json") or {}
    exec_adapt = _safe_float(exec_info.get("adaptive_risk_scale", np.nan), default=np.nan)
    if math.isfinite(exec_adapt):
        emod = _clamp(exec_adapt, 0.40, 1.05)
        comps["execution_adaptive_risk_modifier"] = {"value": float(emod), "found": True}
        active_vals.append(float(emod))

    # Heartbeat stress diagnostics (volatility-metabolism stress state).
    hb_stress, hb_stress_found = _latest_series_value(runs_dir / "heartbeat_stress.csv", lo=0.0, hi=1.0, default=0.5)
    hb_mod = _clamp(1.02 - 0.42 * max(0.0, hb_stress - 0.55), 0.70, 1.02)
    comps["heartbeat_stress_modifier"] = {"value": float(hb_mod), "found": bool(hb_stress_found)}
    if hb_stress_found:
        active_vals.append(float(hb_mod))

    hb_bpm = _load_series(runs_dir / "heartbeat_bpm.csv")
    hb_rise = np.nan
    if hb_bpm is not None and len(hb_bpm) >= 6:
        tail = np.asarray(hb_bpm[-6:], float)
        dif = np.diff(tail)
        hb_rise = float(np.mean(np.clip(dif, -10.0, 10.0)))

    # Nested WF leakage/coverage diagnostics modifier.
    nested = _load_json(runs_dir / "nested_wf_summary.json") or {}
    n_assets = _safe_float(nested.get("assets", np.nan), default=np.nan)
    n_util = _safe_float(nested.get("avg_outer_fold_utilization", np.nan), default=np.nan)
    n_low = _safe_float(nested.get("low_utilization_assets", np.nan), default=np.nan)
    n_train = _safe_float(nested.get("avg_train_ratio_mean", np.nan), default=np.nan)
    n_params = nested.get("params", {}) if isinstance(nested.get("params"), dict) else {}
    n_gap = _safe_float(n_params.get("purge_embargo_ratio", np.nan), default=np.nan)
    if math.isfinite(n_assets) and n_assets >= 5 and math.isfinite(n_util):
        util_mod = _clamp(0.82 + 0.28 * n_util, 0.70, 1.08)
        low_frac = float(np.clip(n_low / max(1.0, n_assets), 0.0, 1.0)) if math.isfinite(n_low) else 0.0
        low_mod = _clamp(1.00 - 0.22 * low_frac, 0.74, 1.00)
        train_mod = _clamp(0.86 + 0.20 * (n_train if math.isfinite(n_train) else 0.75), 0.78, 1.05)
        if math.isfinite(n_gap):
            gap_mod = _clamp(1.00 - 0.18 * max(0.0, n_gap - 0.35), 0.76, 1.00)
        else:
            gap_mod = 1.00
        nested_mod = _clamp(util_mod * low_mod * train_mod * gap_mod, 0.60, 1.05)
        comps["nested_wf_leakage_modifier"] = {"value": float(nested_mod), "found": True}
        active_vals.append(float(nested_mod))
    else:
        low_frac = np.nan

    # Hive/cross-ecosystem stress diagnostics.
    cross_hive = _load_json(runs_dir / "cross_hive_summary.json") or {}
    ch_ad = cross_hive.get("adaptive_diagnostics", {}) if isinstance(cross_hive.get("adaptive_diagnostics"), dict) else {}
    ch_dis = _safe_float(ch_ad.get("mean_disagreement", np.nan), default=np.nan)
    ch_disp = _safe_float(ch_ad.get("mean_stability_dispersion", np.nan), default=np.nan)
    ch_frac = _safe_float(ch_ad.get("mean_regime_fracture", np.nan), default=np.nan)
    ch_ent = cross_hive.get("entropy_adaptive_diagnostics", {}) if isinstance(cross_hive.get("entropy_adaptive_diagnostics"), dict) else {}
    ent_target_mean = _safe_float(ch_ent.get("entropy_target_mean", np.nan), default=np.nan)
    ent_target_max = _safe_float(ch_ent.get("entropy_target_max", np.nan), default=np.nan)
    ent_strength_mean = _safe_float(ch_ent.get("entropy_strength_mean", np.nan), default=np.nan)
    ent_strength_max = _safe_float(ch_ent.get("entropy_strength_max", np.nan), default=np.nan)
    ch_turn_mean = _safe_float(cross_hive.get("mean_turnover", np.nan), default=np.nan)
    ch_turn_max = _safe_float(cross_hive.get("max_turnover", np.nan), default=np.nan)
    ch_turn_roll = _safe_float(cross_hive.get("rolling_turnover_max", np.nan), default=np.nan)
    ch_crowd = np.nan
    ch_crowd_obj = cross_hive.get("crowding_penalty_mean", {})
    if isinstance(ch_crowd_obj, dict):
        cvals = []
        for v in ch_crowd_obj.values():
            xv = _safe_float(v, np.nan)
            if math.isfinite(xv):
                cvals.append(float(xv))
        if cvals:
            ch_crowd = float(np.mean(cvals))
    else:
        ch_crowd = _safe_float(ch_crowd_obj, default=np.nan)
    if math.isfinite(ch_dis) or math.isfinite(ch_disp) or math.isfinite(ch_frac):
        dis_mod = _clamp(1.04 - 0.28 * (ch_dis if math.isfinite(ch_dis) else 0.60), 0.72, 1.04)
        disp_mod = _clamp(1.04 - 0.24 * (ch_disp if math.isfinite(ch_disp) else 0.60), 0.72, 1.04)
        frac_mod = _clamp(1.02 - 0.55 * (ch_frac if math.isfinite(ch_frac) else 0.10), 0.70, 1.02)
        hive_mod = _clamp(dis_mod * disp_mod * frac_mod, 0.65, 1.08)
        comps["hive_ecosystem_stability_modifier"] = {"value": float(hive_mod), "found": True}
        active_vals.append(float(hive_mod))
    if math.isfinite(ch_crowd):
        crowd_mod = _clamp(1.02 - 0.45 * max(0.0, ch_crowd), 0.68, 1.02)
        comps["hive_crowding_modifier"] = {"value": float(crowd_mod), "found": True}
        active_vals.append(float(crowd_mod))
    if math.isfinite(ent_strength_mean) or math.isfinite(ent_target_mean):
        esm = ent_strength_mean if math.isfinite(ent_strength_mean) else 0.25
        etm = ent_target_mean if math.isfinite(ent_target_mean) else 0.60
        ent_mod = _clamp(1.03 - 0.32 * max(0.0, esm - 0.35) - 0.20 * max(0.0, etm - 0.62), 0.72, 1.03)
        comps["hive_entropy_pressure_modifier"] = {"value": float(ent_mod), "found": True}
        active_vals.append(float(ent_mod))
    if math.isfinite(ch_turn_mean) or math.isfinite(ch_turn_max) or math.isfinite(ch_turn_roll):
        tm = ch_turn_mean if math.isfinite(ch_turn_mean) else 0.20
        tx = ch_turn_max if math.isfinite(ch_turn_max) else 0.70
        tr = ch_turn_roll if math.isfinite(ch_turn_roll) else 1.00
        turn_mod = _clamp(
            1.03
            - 0.38 * max(0.0, tm - 0.20)
            - 0.18 * max(0.0, tx - 0.70)
            - 0.12 * max(0.0, tr - 1.00),
            0.68,
            1.03,
        )
        comps["hive_turnover_modifier"] = {"value": float(turn_mod), "found": True}
        active_vals.append(float(turn_mod))

    hive_evo = _load_json(runs_dir / "hive_evolution.json") or {}
    he_pressure = _safe_float(hive_evo.get("action_pressure_mean", np.nan), default=np.nan)
    he_vital = hive_evo.get("latest_vitality", {}) if isinstance(hive_evo.get("latest_vitality"), dict) else {}
    vit_vals = []
    if isinstance(he_vital, dict):
        for x in he_vital.values():
            xv = _safe_float(x, np.nan)
            if math.isfinite(xv):
                vit_vals.append(float(xv))
    vit_min = float(min(vit_vals)) if vit_vals else np.nan
    vit_mean = float(np.mean(vit_vals)) if vit_vals else np.nan
    if math.isfinite(he_pressure) or math.isfinite(vit_mean):
        pressure = he_pressure if math.isfinite(he_pressure) else 0.0
        vitality = vit_mean if math.isfinite(vit_mean) else 0.55
        evo_mod = _clamp(0.84 + 0.30 * vitality - 0.40 * pressure, 0.65, 1.08)
        comps["hive_evolution_modifier"] = {"value": float(evo_mod), "found": True}
        active_vals.append(float(evo_mod))

    if active_vals:
        arr = np.clip(np.asarray(active_vals, float), 0.20, 2.00)
        mult = float(np.exp(np.mean(np.log(arr + 1e-12))))
    else:
        mult = 1.0
    mult = float(np.clip(mult, 0.50, 1.10))

    if mult < 0.72:
        regime = "defensive"
    elif mult > 0.98:
        regime = "risk_on"
    else:
        regime = "balanced"

    risk_flags = []
    if drift_status == "warn":
        risk_flags.append("drift_warn")
    elif drift_status == "alert":
        risk_flags.append("drift_alert")
    if math.isfinite(q_step) and q_step > 0.10:
        risk_flags.append("quality_governor_step_spike")
    if dsf:
        if dsv >= 0.82:
            risk_flags.append("council_divergence_alert")
        elif dsv >= 0.66:
            risk_flags.append("council_divergence_warn")

    # Regime fracture signal from disagreement/volatility/breadth stress.
    rf = _load_json(runs_dir / "regime_fracture_info.json") or {}
    rf_state = str(rf.get("state", "")).lower()
    rf_score = _safe_float(rf.get("latest_score", np.nan), default=np.nan)
    if rf_state:
        if rf_state == "fracture_warn":
            risk_flags.append("fracture_warn")
        elif rf_state == "fracture_alert":
            risk_flags.append("fracture_alert")
    elif math.isfinite(rf_score):
        if rf_score >= 0.85:
            risk_flags.append("fracture_alert")
        elif rf_score >= 0.72:
            risk_flags.append("fracture_warn")

    if math.isfinite(exec_adapt):
        if exec_adapt < 0.55:
            risk_flags.append("exec_risk_hard")
        elif exec_adapt < 0.75:
            risk_flags.append("exec_risk_tight")

    if hb_stress_found:
        if (hb_stress >= 0.84) or (math.isfinite(hb_rise) and hb_stress >= 0.70 and hb_rise >= 2.0):
            risk_flags.append("heartbeat_alert")
        elif (hb_stress >= 0.68) or (math.isfinite(hb_rise) and hb_stress >= 0.60 and hb_rise >= 1.0):
            risk_flags.append("heartbeat_warn")

    if math.isfinite(n_assets) and n_assets >= 5 and math.isfinite(n_util):
        if (n_util < 0.30) or (math.isfinite(low_frac) and low_frac > 0.65):
            risk_flags.append("nested_leakage_alert")
        elif (n_util < 0.45) or (math.isfinite(low_frac) and low_frac > 0.45):
            risk_flags.append("nested_leakage_warn")

    if math.isfinite(ch_dis) or math.isfinite(ch_disp) or math.isfinite(ch_frac):
        dis = ch_dis if math.isfinite(ch_dis) else 0.0
        disp = ch_disp if math.isfinite(ch_disp) else 0.0
        frac = ch_frac if math.isfinite(ch_frac) else 0.0
        if (dis > 0.78) or (disp > 0.80) or (frac > 0.32):
            risk_flags.append("hive_stress_alert")
        elif (dis > 0.62) or (disp > 0.66) or (frac > 0.22):
            risk_flags.append("hive_stress_warn")
    if math.isfinite(ch_crowd):
        if ch_crowd > 0.58:
            risk_flags.append("hive_crowding_alert")
        elif ch_crowd > 0.42:
            risk_flags.append("hive_crowding_warn")
    if math.isfinite(ent_strength_max) or math.isfinite(ent_target_max):
        esm = ent_strength_max if math.isfinite(ent_strength_max) else 0.0
        etm = ent_target_max if math.isfinite(ent_target_max) else 0.0
        if (esm > 0.92) or (etm > 0.86):
            risk_flags.append("hive_entropy_alert")
        elif (esm > 0.78) or (etm > 0.74):
            risk_flags.append("hive_entropy_warn")
    if math.isfinite(ch_turn_mean) or math.isfinite(ch_turn_max) or math.isfinite(ch_turn_roll):
        tm = ch_turn_mean if math.isfinite(ch_turn_mean) else 0.0
        tx = ch_turn_max if math.isfinite(ch_turn_max) else 0.0
        tr = ch_turn_roll if math.isfinite(ch_turn_roll) else 0.0
        if (tx > 1.05) or (tr > 1.35):
            risk_flags.append("hive_turnover_alert")
        elif (tm > 0.38) or (tx > 0.85) or (tr > 1.15):
            risk_flags.append("hive_turnover_warn")

    if math.isfinite(he_pressure) or math.isfinite(vit_min):
        pressure = he_pressure if math.isfinite(he_pressure) else 0.0
        vmin = vit_min if math.isfinite(vit_min) else 1.0
        if (pressure > 0.30) or (vmin < 0.30):
            risk_flags.append("hive_stress_alert")
        elif (pressure > 0.16) or (vmin < 0.42):
            risk_flags.append("hive_stress_warn")

    mem_fb = _memory_feedback(runs_dir)
    if bool(mem_fb.get("active", False)):
        mscale = _safe_float(mem_fb.get("risk_scale", np.nan), default=np.nan)
        if math.isfinite(mscale):
            comps["novaspine_memory_feedback_modifier"] = {"value": float(_clamp(mscale, 0.70, 1.10)), "found": True}
            active_vals.append(float(_clamp(mscale, 0.70, 1.10)))
            if mscale <= 0.84:
                risk_flags.append("memory_feedback_alert")
            elif mscale <= 0.95:
                risk_flags.append("memory_feedback_warn")
        mpress = _safe_float(mem_fb.get("turnover_pressure", np.nan), default=np.nan)
        if math.isfinite(mpress):
            if mpress >= 0.72:
                risk_flags.append("memory_turnover_alert")
            elif mpress >= 0.45:
                risk_flags.append("memory_turnover_warn")

    aion_fb = _aion_outcome_feedback()
    if bool(aion_fb.get("active", False)):
        asc = _safe_float(aion_fb.get("risk_scale", np.nan), default=np.nan)
        if math.isfinite(asc):
            stale = bool(aion_fb.get("stale", False))
            if stale:
                # Stale outcome feedback should not strongly perturb live runtime scaling.
                amod = _clamp(0.985 + 0.03 * (asc - 1.0), 0.95, 1.02)
            else:
                amod = _clamp(asc, 0.65, 1.08)
            comps["aion_outcome_modifier"] = {"value": float(amod), "found": True}
            active_vals.append(float(amod))
            st = str(aion_fb.get("status", "unknown")).strip().lower()
            if stale:
                risk_flags.append("aion_outcome_stale")
            else:
                if st == "alert":
                    risk_flags.append("aion_outcome_alert")
                elif st == "warn":
                    risk_flags.append("aion_outcome_warn")

    # Recompute final multiplier after late-stage feedback modifiers.
    if active_vals:
        arr = np.clip(np.asarray(active_vals, float), 0.20, 2.00)
        mult = float(np.exp(np.mean(np.log(arr + 1e-12))))
    else:
        mult = 1.0
    mult = float(np.clip(mult, 0.50, 1.10))
    if mult < 0.72:
        regime = "defensive"
    elif mult > 0.98:
        regime = "risk_on"
    else:
        regime = "balanced"

    return {
        "runtime_multiplier": mult,
        "regime": regime,
        "components": comps,
        "active_component_count": int(len(active_vals)),
        "risk_flags": _canonicalize_risk_flags(risk_flags),
        "memory_feedback": mem_fb,
        "aion_feedback": aion_fb,
    }


def _quality_gate(
    health_json: Path,
    alerts_json: Path,
    min_health_score: float,
    max_health_age_hours: float,
):
    issues = []
    health = _load_json(health_json) or {}
    alerts = _load_json(alerts_json) or {}

    score = _safe_float(health.get("health_score", 0.0), default=0.0)
    age_h = _hours_since_file(health_json)
    alerts_ok = bool(alerts.get("ok", True))

    if score < float(min_health_score):
        issues.append(f"health_score<{min_health_score} ({score:.1f})")
    if age_h is None:
        issues.append("system_health missing")
    elif age_h > float(max_health_age_hours):
        issues.append(f"system_health stale>{max_health_age_hours}h ({age_h:.2f}h)")
    if not alerts_ok:
        issues.append("health alerts not clear")

    return {
        "ok": len(issues) == 0,
        "score": score,
        "health_age_hours": age_h,
        "alerts_ok": alerts_ok,
        "issues": issues,
    }


def _build_global_overlay(scored: pd.DataFrame) -> dict:
    if scored.empty:
        return {"bias": 0.0, "confidence": 0.0}
    conf = pd.to_numeric(scored["confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0).values.astype(float)
    abs_bias = pd.to_numeric(scored["bias"], errors="coerce").fillna(0.0).abs().values.astype(float)
    w = conf * np.maximum(abs_bias, 0.05)
    w_sum = float(w.sum())
    if w_sum <= 1e-12:
        w = np.ones(len(scored), dtype=float)
        w_sum = float(w.sum())

    bias_vals = pd.to_numeric(scored["bias"], errors="coerce").fillna(0.0).values.astype(float)
    bias = float(np.dot(w, bias_vals) / w_sum)
    conf_global = float(np.dot(w, conf) / w_sum)
    return {
        "bias": round(_clamp(bias, -1.0, 1.0), 6),
        "confidence": round(_clamp(conf_global, 0.0, 1.0), 6),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wf", default=str(RUNS / "walk_forward_table_plus.csv"))
    ap.add_argument("--out-json", default=str(RUNS / "q_signal_overlay.json"))
    ap.add_argument("--out-csv", default=str(RUNS / "q_signal_overlay.csv"))
    ap.add_argument("--mirror-json", default="", help="Optional second JSON write target.")
    ap.add_argument("--min-confidence", type=float, default=0.56)
    ap.add_argument("--max-symbols", type=int, default=80)
    ap.add_argument(
        "--watchlist-txt",
        default="",
        help="Optional watchlist file. If provided, exported symbols are filtered to this set.",
    )
    ap.add_argument("--health-json", default=str(RUNS / "system_health.json"))
    ap.add_argument("--alerts-json", default=str(RUNS / "health_alerts.json"))
    ap.add_argument("--min-health-score", type=float, default=70.0)
    ap.add_argument("--max-health-age-hours", type=float, default=8.0)
    ap.add_argument(
        "--allow-degraded",
        action="store_true",
        help="If set, still export scored signals even when quality gate fails.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    wf_path = Path(args.wf)
    used_fallback = False
    if wf_path.exists():
        df = pd.read_csv(wf_path)
        if "asset" not in df.columns:
            raise SystemExit(f"{wf_path} must include an 'asset' column.")
        scored = _build_scores(df)
    else:
        scored = _build_scores_from_final_weights(RUNS / "portfolio_weights_final.csv")
        used_fallback = True

    scored_raw = scored.copy()
    scored = scored.replace([np.inf, -np.inf], np.nan).dropna(subset=["symbol", "bias", "confidence"])
    scored = scored[scored["confidence"] >= float(args.min_confidence)].copy()
    scored = scored[scored["bias"].abs() > 1e-6].copy()
    if used_fallback and scored.empty and not scored_raw.empty:
        alt = scored_raw.replace([np.inf, -np.inf], np.nan).dropna(subset=["symbol", "bias", "confidence"]).copy()
        alt = alt[alt["bias"].abs() > 1e-6].copy()
        alt = alt[alt["confidence"] >= 0.10].copy()
        scored = alt.sort_values("rank", ascending=False).head(max(1, min(20, int(args.max_symbols))))
    scored = _collapse_to_canonical(scored)

    watchlist = _load_watchlist(args.watchlist_txt)
    if watchlist:
        scored = scored[scored["symbol"].isin(watchlist)].copy()

    scored = scored.sort_values("rank", ascending=False).head(max(1, int(args.max_symbols)))
    qgate = _quality_gate(
        health_json=Path(args.health_json),
        alerts_json=Path(args.alerts_json),
        min_health_score=float(args.min_health_score),
        max_health_age_hours=float(args.max_health_age_hours),
    )
    ctx = _runtime_context(RUNS)
    degrade = (not qgate["ok"]) and (not bool(args.allow_degraded))
    if degrade:
        scored = scored.iloc[0:0].copy()
    else:
        health_scale = _clamp(_safe_float(qgate.get("score", 0.0), default=0.0) / 100.0, 0.70, 1.05)
        runtime_scale = _clamp(_safe_float(ctx.get("runtime_multiplier", 1.0), default=1.0), 0.50, 1.10)
        conf_scale = _clamp(health_scale * runtime_scale, 0.45, 1.10)
        bias_scale = _clamp(math.sqrt(max(0.0, runtime_scale)), 0.70, 1.05)
        if not scored.empty:
            scored = scored.copy()
            scored["confidence"] = np.clip(scored["confidence"].astype(float) * conf_scale, 0.0, 1.0)
            scored["bias"] = np.clip(scored["bias"].astype(float) * bias_scale, -1.0, 1.0)
            scored = scored[scored["confidence"] >= 0.05].copy()
            scored["rank"] = scored["confidence"] * scored["bias"].abs()
            scored = scored.sort_values("rank", ascending=False)

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    global_overlay = _build_global_overlay(scored)
    if degrade:
        global_overlay = {"bias": 0.0, "confidence": 0.0}
    payload = {
        "generated_at": ts,
        "generated_at_utc": ts,
        "source": "q.walk_forward_plus",
        "global": global_overlay,
        "signals": {
            row["symbol"]: {
                "bias": round(_clamp(row["bias"], -1.0, 1.0), 6),
                "confidence": round(_clamp(row["confidence"], 0.0, 1.0), 6),
            }
            for _, row in scored.iterrows()
        },
        "coverage": {
            "symbols": int(len(scored)),
            "watchlist_filtered": bool(watchlist),
        },
        "runtime_context": ctx,
        "quality_gate": qgate,
        "degraded_safe_mode": bool(degrade),
        "source_mode": "wf_table" if not used_fallback else "final_weights_fallback",
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    scored_out = scored[["symbol", "bias", "confidence", "sharpe", "hit"]].copy()
    scored_out.insert(0, "generated_at", ts)
    scored_out.to_csv(out_csv, index=False)

    mirror = str(args.mirror_json).strip()
    if mirror:
        mirror_path = Path(mirror)
        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        mirror_path.write_text(json.dumps(payload, indent=2))
        print(f"✅ Mirrored JSON: {mirror_path}")

    print(f"✅ Wrote {out_json}")
    print(f"✅ Wrote {out_csv}")
    print(f"Signals exported: {len(payload['signals'])}")
    print(f"Global overlay: bias={_safe_float(global_overlay.get('bias')):.4f}, confidence={_safe_float(global_overlay.get('confidence')):.4f}")
    if degrade:
        print(f"(!) Degraded safe mode enabled: {', '.join(qgate['issues'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
