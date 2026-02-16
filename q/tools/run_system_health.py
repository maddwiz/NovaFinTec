#!/usr/bin/env python3
# System health snapshot for Q pipeline.
#
# Writes:
#   runs_plus/system_health.json

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmods.aion_feedback import (  # noqa: E402
    choose_feedback_source,
    feedback_lineage,
    feedback_has_metrics,
    load_outcome_feedback,
    normalize_source_preference,
)

RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


def _now():
    return datetime.now(timezone.utc)


def _hours_since(path: Path):
    if not path.exists():
        return None
    ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return float((_now() - ts).total_seconds() / 3600.0)


def _load_series(path: Path):
    if not path.exists():
        return None
    try:
        a = np.loadtxt(path, delimiter=",")
    except Exception:
        try:
            a = np.loadtxt(path, delimiter=",", skiprows=1)
        except Exception:
            return None
    a = np.asarray(a, float)
    if a.ndim == 2 and a.shape[1] >= 1:
        a = a[:, -1]
    return a.ravel()


def _load_named_series(path: Path, column: str):
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if column not in df.columns:
        return None
    vals = pd.to_numeric(df[column], errors="coerce").fillna(0.0).values.astype(float)
    return vals.ravel() if len(vals) else None


def _load_row_mean_series(path: Path):
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    cols = [c for c in df.columns if str(c).lower() not in {"date", "timestamp", "time"}]
    if not cols:
        return None
    vals = [pd.to_numeric(df[c], errors="coerce").fillna(0.0).values.astype(float) for c in cols]
    if not vals:
        return None
    mat = np.column_stack(vals)
    return np.nan_to_num(np.mean(mat, axis=1), nan=0.0, posinf=0.0, neginf=0.0)


def _load_matrix(path: Path):
    if not path.exists():
        return None
    try:
        a = np.loadtxt(path, delimiter=",")
    except Exception:
        try:
            a = np.loadtxt(path, delimiter=",", skiprows=1)
        except Exception:
            return None
    a = np.asarray(a, float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def _to_float(x):
    try:
        v = float(x)
    except Exception:
        return None
    return v if np.isfinite(v) else None


def _analyze_execution_constraints(info: dict | None):
    metrics = {}
    issues = []
    if not isinstance(info, dict):
        return metrics, issues

    gross_before = _to_float(info.get("gross_before_mean"))
    gross_after = _to_float(info.get("gross_after_mean"))
    turn_before = _to_float(info.get("turnover_before_mean"))
    turn_after = _to_float(info.get("turnover_after_mean"))
    turn_after_max = _to_float(info.get("turnover_after_max"))
    step_cap = _to_float(info.get("max_step_turnover"))

    if gross_before is not None:
        metrics["exec_gross_before_mean"] = gross_before
    if gross_after is not None:
        metrics["exec_gross_after_mean"] = gross_after
    if turn_before is not None:
        metrics["exec_turnover_before_mean"] = turn_before
    if turn_after is not None:
        metrics["exec_turnover_after_mean"] = turn_after
    if turn_after_max is not None:
        metrics["exec_turnover_after_max"] = turn_after_max
    if step_cap is not None:
        metrics["exec_max_step_turnover"] = step_cap

    if turn_before is not None and turn_after is not None and turn_after > turn_before + 1e-9:
        issues.append("execution constraints increased mean turnover")
    if turn_before is not None and turn_after is not None:
        if turn_before >= 0.20 and turn_after <= 0.01:
            issues.append("execution constraints may be over-throttling turnover")
    if step_cap is not None and turn_after_max is not None and turn_after_max > step_cap + 1e-6:
        issues.append("execution turnover exceeds configured max_step_turnover")
    if gross_before is not None and gross_after is not None:
        if gross_before >= 0.25 and gross_after <= 0.02:
            issues.append("execution constraints collapsed gross exposure")

    return metrics, issues


def _analyze_cross_hive_turnover(
    cross_info: dict | None,
    *,
    max_mean_turnover: float = 0.45,
    max_step_turnover: float = 1.00,
    max_rolling_turnover: float = 1.25,
):
    metrics = {}
    issues = []
    if not isinstance(cross_info, dict):
        return metrics, issues

    mean_turnover = _to_float(cross_info.get("mean_turnover"))
    max_turnover = _to_float(cross_info.get("max_turnover"))
    rolling_turnover = _to_float(cross_info.get("rolling_turnover_max"))
    rolling_window = _to_float(cross_info.get("rolling_turnover_window"))
    rolling_limit = _to_float(cross_info.get("rolling_turnover_limit"))

    if mean_turnover is not None:
        metrics["cross_hive_mean_turnover"] = mean_turnover
    if max_turnover is not None:
        metrics["cross_hive_max_turnover"] = max_turnover
    if rolling_turnover is not None:
        metrics["cross_hive_rolling_turnover_max"] = rolling_turnover
    if rolling_window is not None:
        metrics["cross_hive_rolling_turnover_window"] = float(rolling_window)
    if rolling_limit is not None:
        metrics["cross_hive_rolling_turnover_limit"] = float(rolling_limit)

    if mean_turnover is not None and mean_turnover > float(max_mean_turnover):
        issues.append("cross-hive mean turnover exceeds threshold")
    if max_turnover is not None and max_turnover > float(max_step_turnover):
        issues.append("cross-hive max turnover exceeds threshold")
    if rolling_turnover is not None and rolling_turnover > float(max_rolling_turnover):
        issues.append("cross-hive rolling turnover exceeds threshold")

    return metrics, issues


def _analyze_novaspine_turnover(
    nctx: dict | None,
    nhive: dict | None,
    *,
    max_turnover_pressure: float = 0.72,
    max_turnover_dampener: float = 0.10,
):
    metrics = {}
    issues = []
    ctx = nctx if isinstance(nctx, dict) else {}
    hive = nhive if isinstance(nhive, dict) else {}

    status_ctx = str(ctx.get("status", "na")).strip().lower()
    status_hive = str(hive.get("status", "na")).strip().lower()
    p_ctx = _to_float(ctx.get("turnover_pressure"))
    p_hive = _to_float(hive.get("turnover_pressure"))
    d_ctx = _to_float(ctx.get("turnover_dampener"))
    d_hive = _to_float(hive.get("turnover_dampener"))

    if status_ctx:
        metrics["novaspine_context_status"] = status_ctx
    if status_hive:
        metrics["novaspine_hive_status"] = status_hive
    if p_ctx is not None:
        metrics["novaspine_context_turnover_pressure"] = p_ctx
    if p_hive is not None:
        metrics["novaspine_hive_turnover_pressure"] = p_hive
    if d_ctx is not None:
        metrics["novaspine_context_turnover_dampener"] = d_ctx
    if d_hive is not None:
        metrics["novaspine_hive_turnover_dampener"] = d_hive

    pressure_vals = [x for x in [p_ctx, p_hive] if x is not None]
    damp_vals = [x for x in [d_ctx, d_hive] if x is not None]
    if pressure_vals:
        metrics["novaspine_turnover_pressure_max"] = float(np.max(pressure_vals))
        metrics["novaspine_turnover_pressure_mean"] = float(np.mean(pressure_vals))
    if damp_vals:
        metrics["novaspine_turnover_dampener_max"] = float(np.max(damp_vals))
        metrics["novaspine_turnover_dampener_mean"] = float(np.mean(damp_vals))

    if pressure_vals and float(np.max(pressure_vals)) > float(max_turnover_pressure):
        issues.append("novaspine turnover pressure exceeds threshold")
    if damp_vals and float(np.max(damp_vals)) > float(max_turnover_dampener):
        issues.append("novaspine turnover dampener exceeds threshold")
    if any(s in {"unreachable", "error"} or str(s).startswith("http_") for s in [status_ctx, status_hive]):
        issues.append("novaspine recall endpoint unhealthy")

    return metrics, issues


def _analyze_novaspine_replay(
    replay_info: dict | None,
    *,
    warn_backlog_files: int = 5,
    alert_backlog_files: int = 20,
    max_failed_events: int = 0,
):
    metrics = {}
    issues = []
    info = replay_info if isinstance(replay_info, dict) else {}
    if not info:
        return metrics, issues

    enabled = bool(info.get("enabled", False))
    backend = str(info.get("backend", "na")).strip().lower() or "na"
    queued_files = _to_float(info.get("queued_files"))
    replayed_events = _to_float(info.get("replayed_events"))
    failed_events = _to_float(info.get("failed_events"))

    metrics["novaspine_replay_enabled"] = bool(enabled)
    metrics["novaspine_replay_backend"] = backend
    if queued_files is not None:
        metrics["novaspine_replay_queued_files"] = float(max(0.0, queued_files))
    if replayed_events is not None:
        metrics["novaspine_replay_replayed_events"] = float(max(0.0, replayed_events))
    if failed_events is not None:
        metrics["novaspine_replay_failed_events"] = float(max(0.0, failed_events))

    if not enabled:
        return metrics, issues

    warn_n = max(1, int(warn_backlog_files))
    alert_n = max(warn_n + 1, int(alert_backlog_files))
    fail_cap = max(0, int(max_failed_events))
    qn = int(queued_files) if queued_files is not None else 0
    fn = int(failed_events) if failed_events is not None else 0

    if fn > fail_cap:
        issues.append("novaspine replay failed events exceed threshold")
    if qn >= alert_n:
        issues.append("novaspine replay backlog exceeds alert threshold")
    elif qn >= warn_n:
        issues.append("novaspine replay backlog exceeds warn threshold")

    return metrics, issues


def _analyze_runtime_total_scalar(
    runtime_total_scalar: np.ndarray | None,
    *,
    min_mean: float = 0.22,
    min_p10: float = 0.10,
    min_min: float = 0.04,
):
    metrics = {}
    issues = []
    if runtime_total_scalar is None:
        return metrics, issues
    arr = np.asarray(runtime_total_scalar, float).ravel()
    if arr.size == 0:
        return metrics, issues
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return metrics, issues

    mean_v = float(np.mean(arr))
    min_v = float(np.min(arr))
    p10_v = float(np.percentile(arr, 10))
    p25_v = float(np.percentile(arr, 25))

    metrics["runtime_total_scalar_mean"] = mean_v
    metrics["runtime_total_scalar_min"] = min_v
    metrics["runtime_total_scalar_p10"] = p10_v
    metrics["runtime_total_scalar_p25"] = p25_v

    if mean_v < float(min_mean):
        issues.append("runtime_total_scalar mean below threshold")
    if p10_v < float(min_p10):
        issues.append("runtime_total_scalar p10 below threshold")
    if min_v < float(min_min):
        issues.append("runtime_total_scalar min below threshold")
    return metrics, issues


def _overlay_aion_feedback_metrics(overlay: dict | None):
    return _overlay_aion_feedback_metrics_with_fallback(overlay, fallback_feedback=None)


def _aion_feedback_has_metrics(af: dict | None):
    return feedback_has_metrics(af)


def _overlay_aion_feedback_metrics_with_fallback(
    overlay: dict | None,
    fallback_feedback: dict | None = None,
    source_pref: str = "auto",
):
    metrics = {}
    issues = []
    overlay_af = None
    if isinstance(overlay, dict):
        rt = overlay.get("runtime_context", {})
        if isinstance(rt, dict):
            overlay_af = rt.get("aion_feedback", {})
    af, source = choose_feedback_source(
        overlay_af,
        fallback_feedback,
        source_pref=source_pref,
        prefer_overlay_when_fresh=True,
    )
    if not _aion_feedback_has_metrics(af):
        return metrics, issues

    active = bool(af.get("active", False))
    status = str(af.get("status", "unknown")).strip().lower() or "unknown"
    closed = int(max(0, _to_float(af.get("closed_trades")) or 0))
    risk_scale = _to_float(af.get("risk_scale"))
    hit_rate = _to_float(af.get("hit_rate"))
    profit_factor = _to_float(af.get("profit_factor"))
    expectancy = _to_float(af.get("expectancy"))
    dd_norm = _to_float(af.get("drawdown_norm"))
    age_hours = _to_float(af.get("age_hours"))
    stale = bool(af.get("stale", False))
    max_age_hours = _to_float(af.get("max_age_hours"))
    if max_age_hours is None:
        try:
            max_age_hours = float(os.getenv("Q_MAX_AION_FEEDBACK_AGE_HOURS", os.getenv("Q_AION_FEEDBACK_MAX_AGE_HOURS", "72")))
        except Exception:
            max_age_hours = 72.0
    if (not stale) and age_hours is not None and np.isfinite(age_hours) and max_age_hours is not None:
        stale = bool(age_hours > max_age_hours)

    lineage = feedback_lineage(
        af,
        selected_source=source,
        source_preference=source_pref,
        default_source="unknown",
    )

    metrics["aion_feedback_active"] = active
    metrics["aion_feedback_status"] = status
    metrics["aion_feedback_closed_trades"] = closed
    metrics["aion_feedback_risk_scale"] = risk_scale
    metrics["aion_feedback_hit_rate"] = hit_rate
    metrics["aion_feedback_profit_factor"] = profit_factor
    metrics["aion_feedback_expectancy"] = expectancy
    metrics["aion_feedback_drawdown_norm"] = dd_norm
    metrics["aion_feedback_age_hours"] = age_hours
    metrics["aion_feedback_stale"] = stale
    metrics["aion_feedback_max_age_hours"] = max_age_hours
    metrics["aion_feedback_source"] = str(lineage.get("source", "unknown"))
    metrics["aion_feedback_source_selected"] = str(lineage.get("source_selected", "unknown"))
    metrics["aion_feedback_source_preference"] = str(lineage.get("source_preference", "auto"))

    enough_closed = closed >= 8
    if active and stale:
        issues.append("aion_feedback_stale")
    elif active and status in {"alert", "hard"}:
        issues.append("aion_feedback_status=alert")
    if active and enough_closed and risk_scale is not None and risk_scale < 0.75:
        issues.append("aion_feedback_risk_scale_low")
    return metrics, issues


def _staleness_issues(
    checks: list[dict],
    max_required_hours: float = 24.0,
    max_optional_hours: float | None = None,
):
    issues = []
    stale_required = 0
    stale_optional = 0
    req_h = float(max(0.0, max_required_hours))
    opt_h = None if max_optional_hours is None else float(max(0.0, max_optional_hours))
    for c in checks:
        if not isinstance(c, dict):
            continue
        exists = bool(c.get("exists", False))
        if not exists:
            continue
        h = c.get("hours_since_update", None)
        try:
            h = float(h) if h is not None else None
        except Exception:
            h = None
        if h is None or (not np.isfinite(h)):
            continue
        name = str(c.get("file", "unknown"))
        is_required = bool(c.get("required", False))
        if is_required and h > req_h:
            stale_required += 1
            issues.append(f"stale_required_file>{req_h:.1f}h ({name}:{h:.2f}h)")
        if (not is_required) and opt_h is not None and h > opt_h:
            stale_optional += 1
            issues.append(f"stale_optional_file>{opt_h:.1f}h ({name}:{h:.2f}h)")
    return {"stale_required_count": stale_required, "stale_optional_count": stale_optional}, issues


def _append_card(title, html):
    for name in ["report_all.html", "report_best_plus.html", "report_plus.html", "report.html"]:
        p = ROOT / name
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        txt = txt.replace("</body>", card + "</body>") if "</body>" in txt else txt + card
        p.write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    required = [
        RUNS / "portfolio_weights_final.csv",
        RUNS / "meta_mix.csv",
        RUNS / "global_governor.csv",
        RUNS / "quality_governor.csv",
        RUNS / "heartbeat_exposure_scaler.csv",
        RUNS / "heartbeat_stress.csv",
        RUNS / "legacy_exposure.csv",
        RUNS / "cross_hive_weights.csv",
        RUNS / "weights_cross_hive_governed.csv",
        RUNS / "weights_turnover_budget_governed.csv",
    ]
    optional = [
        RUNS / "hive_wf_metrics.csv",
        RUNS / "shock_mask.csv",
        RUNS / "shock_mask_info.json",
        RUNS / "hive_diversification_governor.csv",
        RUNS / "hive_persistence_governor.csv",
        RUNS / "hive_dynamic_quality.csv",
        RUNS / "hive_crowding_penalty.csv",
        RUNS / "hive_entropy_schedule.csv",
        RUNS / "reflex_signal_gated.csv",
        RUNS / "synapses_summary.json",
        RUNS / "meta_stack_summary.json",
        RUNS / "hive_transparency.json",
        RUNS / "meta_mix_confidence_raw.csv",
        RUNS / "meta_mix_confidence_calibrated.csv",
        RUNS / "meta_mix_reliability_governor.csv",
        RUNS / "meta_mix_alpha.csv",
        RUNS / "meta_mix_gross.csv",
        RUNS / "meta_mix_quality.csv",
        RUNS / "meta_mix_disagreement.csv",
        RUNS / "tune_best_config.json",
        RUNS / "quality_runtime_modifier.csv",
        RUNS / "regime_fracture_signal.csv",
        RUNS / "regime_fracture_governor.csv",
        RUNS / "regime_fracture_info.json",
        RUNS / "turnover_budget_rolling_after.csv",
        RUNS / "concentration_governor_info.json",
        RUNS / "final_governor_trace.csv",
        RUNS / "dream_coherence_governor.csv",
        RUNS / "dream_coherence_info.json",
        RUNS / "dna_stress_governor.csv",
        RUNS / "dna_stress_info.json",
        RUNS / "dna_stress_components.csv",
        RUNS / "reflex_health_governor.csv",
        RUNS / "reflex_health_info.json",
        RUNS / "reflex_health_components.csv",
        RUNS / "symbolic_governor.csv",
        RUNS / "symbolic_governor_info.json",
        RUNS / "portfolio_weights_exec.csv",
        RUNS / "execution_constraints_info.json",
        RUNS / "portfolio_drift_watch.json",
        RUNS / "q_signal_overlay.json",
        RUNS / "q_signal_overlay.csv",
        RUNS / "novaspine_sync_status.json",
        RUNS / "novaspine_last_batch.json",
        RUNS / "novaspine_context.json",
        RUNS / "novaspine_context_boost.csv",
        RUNS / "novaspine_hive_feedback.json",
        RUNS / "novaspine_hive_boost.csv",
        RUNS / "novaspine_replay_status.json",
        RUNS / "immune_drill.json",
    ]

    checks = []
    for p in required:
        checks.append(
            {
                "file": p.name,
                "required": True,
                "exists": p.exists(),
                "hours_since_update": _hours_since(p),
            }
        )
    for p in optional:
        checks.append(
            {
                "file": p.name,
                "required": False,
                "exists": p.exists(),
                "hours_since_update": _hours_since(p),
            }
        )

    w = _load_matrix(RUNS / "portfolio_weights_final.csv")
    daily = _load_series(RUNS / "daily_returns.csv")
    gov = _load_series(RUNS / "global_governor.csv")
    hive_gov = _load_series(RUNS / "hive_diversification_governor.csv")
    hive_pg = _load_series(RUNS / "hive_persistence_governor.csv")
    quality_gov = _load_series(RUNS / "quality_governor.csv")
    meta_rel = _load_series(RUNS / "meta_mix_reliability_governor.csv")
    meta_alpha = _load_series(RUNS / "meta_mix_alpha.csv")
    meta_gross = _load_series(RUNS / "meta_mix_gross.csv")
    meta_q = _load_series(RUNS / "meta_mix_quality.csv")
    meta_d = _load_series(RUNS / "meta_mix_disagreement.csv")
    dream_gov = _load_series(RUNS / "dream_coherence_governor.csv")
    dna_gov = _load_series(RUNS / "dna_stress_governor.csv")
    reflex_gov = _load_series(RUNS / "reflex_health_governor.csv")
    sym_gov = _load_series(RUNS / "symbolic_governor.csv")
    rf_gov = _load_series(RUNS / "regime_fracture_governor.csv")
    hive_crowding = _load_row_mean_series(RUNS / "hive_crowding_penalty.csv")
    hive_entropy_target = _load_named_series(RUNS / "hive_entropy_schedule.csv", "entropy_target")
    hive_entropy_strength = _load_named_series(RUNS / "hive_entropy_schedule.csv", "entropy_strength")
    nsp_ctx_boost = _load_series(RUNS / "novaspine_context_boost.csv")
    nsp_hive_boost = _load_series(RUNS / "novaspine_hive_boost.csv")
    gov_trace_total = _load_series(RUNS / "final_governor_trace.csv")
    hb_stress = _load_series(RUNS / "heartbeat_stress.csv")
    exec_info = _load_json(RUNS / "execution_constraints_info.json")
    cross_info = _load_json(RUNS / "cross_hive_summary.json")
    replay_info = _load_json(RUNS / "novaspine_replay_status.json")
    nsp_ctx = _load_json(RUNS / "novaspine_context.json")
    nsp_hive = _load_json(RUNS / "novaspine_hive_feedback.json")
    overlay = _load_json(RUNS / "q_signal_overlay.json")
    aion_source_pref = normalize_source_preference(os.getenv("Q_AION_FEEDBACK_SOURCE", "auto"))

    shape = {}
    if w is not None:
        shape["weights_rows"] = int(w.shape[0])
        shape["weights_cols"] = int(w.shape[1])
        shape["weights_abs_mean"] = float(np.mean(np.abs(w)))
    if daily is not None:
        shape["daily_returns_rows"] = int(len(daily))
    if gov is not None:
        shape["global_governor_rows"] = int(len(gov))
        shape["global_governor_mean"] = float(np.mean(gov))
    if hive_gov is not None:
        shape["hive_governor_rows"] = int(len(hive_gov))
        shape["hive_governor_mean"] = float(np.mean(hive_gov))
    if hive_pg is not None:
        shape["hive_persistence_rows"] = int(len(hive_pg))
        shape["hive_persistence_mean"] = float(np.mean(hive_pg))
        shape["hive_persistence_min"] = float(np.min(hive_pg))
        shape["hive_persistence_max"] = float(np.max(hive_pg))
    if quality_gov is not None:
        shape["quality_governor_rows"] = int(len(quality_gov))
        shape["quality_governor_mean"] = float(np.mean(quality_gov))
    if meta_rel is not None:
        shape["meta_mix_reliability_rows"] = int(len(meta_rel))
        shape["meta_mix_reliability_mean"] = float(np.mean(meta_rel))
        shape["meta_mix_reliability_min"] = float(np.min(meta_rel))
        shape["meta_mix_reliability_max"] = float(np.max(meta_rel))
    if meta_alpha is not None:
        shape["meta_mix_alpha_rows"] = int(len(meta_alpha))
        shape["meta_mix_alpha_mean"] = float(np.mean(meta_alpha))
        shape["meta_mix_alpha_min"] = float(np.min(meta_alpha))
        shape["meta_mix_alpha_max"] = float(np.max(meta_alpha))
    if meta_gross is not None:
        shape["meta_mix_gross_rows"] = int(len(meta_gross))
        shape["meta_mix_gross_mean"] = float(np.mean(meta_gross))
        shape["meta_mix_gross_min"] = float(np.min(meta_gross))
        shape["meta_mix_gross_max"] = float(np.max(meta_gross))
    if meta_q is not None:
        shape["meta_mix_quality_rows"] = int(len(meta_q))
        shape["meta_mix_quality_mean"] = float(np.mean(meta_q))
        shape["meta_mix_quality_min"] = float(np.min(meta_q))
        shape["meta_mix_quality_max"] = float(np.max(meta_q))
    if meta_d is not None:
        shape["meta_mix_disagreement_rows"] = int(len(meta_d))
        shape["meta_mix_disagreement_mean"] = float(np.mean(meta_d))
        shape["meta_mix_disagreement_min"] = float(np.min(meta_d))
        shape["meta_mix_disagreement_max"] = float(np.max(meta_d))
    if hb_stress is not None:
        shape["heartbeat_stress_rows"] = int(len(hb_stress))
        shape["heartbeat_stress_mean"] = float(np.mean(hb_stress))
        shape["heartbeat_stress_max"] = float(np.max(hb_stress))
    if dream_gov is not None:
        shape["dream_coherence_rows"] = int(len(dream_gov))
        shape["dream_coherence_mean"] = float(np.mean(dream_gov))
        shape["dream_coherence_min"] = float(np.min(dream_gov))
        shape["dream_coherence_max"] = float(np.max(dream_gov))
    if dna_gov is not None:
        shape["dna_stress_governor_rows"] = int(len(dna_gov))
        shape["dna_stress_governor_mean"] = float(np.mean(dna_gov))
        shape["dna_stress_governor_min"] = float(np.min(dna_gov))
        shape["dna_stress_governor_max"] = float(np.max(dna_gov))
    if reflex_gov is not None:
        shape["reflex_health_governor_rows"] = int(len(reflex_gov))
        shape["reflex_health_governor_mean"] = float(np.mean(reflex_gov))
        shape["reflex_health_governor_min"] = float(np.min(reflex_gov))
        shape["reflex_health_governor_max"] = float(np.max(reflex_gov))
    if sym_gov is not None:
        shape["symbolic_governor_rows"] = int(len(sym_gov))
        shape["symbolic_governor_mean"] = float(np.mean(sym_gov))
    if rf_gov is not None:
        shape["regime_fracture_governor_rows"] = int(len(rf_gov))
        shape["regime_fracture_governor_mean"] = float(np.mean(rf_gov))
    if hive_crowding is not None:
        shape["hive_crowding_rows"] = int(len(hive_crowding))
        shape["hive_crowding_mean"] = float(np.mean(hive_crowding))
        shape["hive_crowding_max"] = float(np.max(hive_crowding))
    if hive_entropy_target is not None:
        shape["hive_entropy_target_rows"] = int(len(hive_entropy_target))
        shape["hive_entropy_target_mean"] = float(np.mean(hive_entropy_target))
        shape["hive_entropy_target_max"] = float(np.max(hive_entropy_target))
    if hive_entropy_strength is not None:
        shape["hive_entropy_strength_rows"] = int(len(hive_entropy_strength))
        shape["hive_entropy_strength_mean"] = float(np.mean(hive_entropy_strength))
        shape["hive_entropy_strength_max"] = float(np.max(hive_entropy_strength))
    if sym_gov is not None:
        shape["symbolic_governor_min"] = float(np.min(sym_gov))
        shape["symbolic_governor_max"] = float(np.max(sym_gov))
    if nsp_ctx_boost is not None:
        shape["novaspine_context_boost_rows"] = int(len(nsp_ctx_boost))
        shape["novaspine_context_boost_mean"] = float(np.mean(nsp_ctx_boost))
    if nsp_hive_boost is not None:
        shape["novaspine_hive_boost_rows"] = int(len(nsp_hive_boost))
        shape["novaspine_hive_boost_mean"] = float(np.mean(nsp_hive_boost))
    if gov_trace_total is not None:
        shape["final_governor_trace_rows"] = int(len(gov_trace_total))
        shape["runtime_total_scalar_mean"] = float(np.mean(gov_trace_total))
        shape["runtime_total_scalar_min"] = float(np.min(gov_trace_total))
        shape["runtime_total_scalar_max"] = float(np.max(gov_trace_total))
        shape["runtime_total_scalar_p10"] = float(np.percentile(gov_trace_total, 10))
        shape["runtime_total_scalar_p25"] = float(np.percentile(gov_trace_total, 25))
    exec_shape, exec_issues = _analyze_execution_constraints(exec_info)
    if exec_shape:
        shape.update(exec_shape)
    cross_shape, cross_issues = _analyze_cross_hive_turnover(
        cross_info,
        max_mean_turnover=float(np.clip(float(os.getenv("Q_SYSTEM_HEALTH_MAX_CROSS_HIVE_MEAN_TURNOVER", "0.45")), 0.05, 2.0)),
        max_step_turnover=float(np.clip(float(os.getenv("Q_SYSTEM_HEALTH_MAX_CROSS_HIVE_MAX_TURNOVER", "1.00")), 0.10, 3.0)),
        max_rolling_turnover=float(np.clip(float(os.getenv("Q_SYSTEM_HEALTH_MAX_CROSS_HIVE_ROLLING_TURNOVER", "1.25")), 0.10, 5.0)),
    )
    if cross_shape:
        shape.update(cross_shape)
    nsp_shape, nsp_issues = _analyze_novaspine_turnover(
        nsp_ctx,
        nsp_hive,
        max_turnover_pressure=float(np.clip(float(os.getenv("Q_SYSTEM_HEALTH_MAX_NOVASPINE_TURNOVER_PRESSURE", "0.72")), 0.05, 1.0)),
        max_turnover_dampener=float(np.clip(float(os.getenv("Q_SYSTEM_HEALTH_MAX_NOVASPINE_TURNOVER_DAMPENER", "0.10")), 0.005, 0.50)),
    )
    if nsp_shape:
        shape.update(nsp_shape)
    replay_shape, replay_issues = _analyze_novaspine_replay(
        replay_info,
        warn_backlog_files=int(max(1, int(os.getenv("Q_SYSTEM_HEALTH_MAX_NOVASPINE_REPLAY_WARN_FILES", "5")))),
        alert_backlog_files=int(max(2, int(os.getenv("Q_SYSTEM_HEALTH_MAX_NOVASPINE_REPLAY_ALERT_FILES", "20")))),
        max_failed_events=int(max(0, int(os.getenv("Q_SYSTEM_HEALTH_MAX_NOVASPINE_REPLAY_FAILED_EVENTS", "0")))),
    )
    if replay_shape:
        shape.update(replay_shape)
    runtime_scale_shape, runtime_scale_issues = _analyze_runtime_total_scalar(
        gov_trace_total,
        min_mean=float(np.clip(float(os.getenv("Q_SYSTEM_HEALTH_MIN_RUNTIME_TOTAL_SCALAR_MEAN", "0.22")), 0.01, 1.0)),
        min_p10=float(np.clip(float(os.getenv("Q_SYSTEM_HEALTH_MIN_RUNTIME_TOTAL_SCALAR_P10", "0.10")), 0.01, 1.0)),
        min_min=float(np.clip(float(os.getenv("Q_SYSTEM_HEALTH_MIN_RUNTIME_TOTAL_SCALAR_MIN", "0.04")), 0.0, 1.0)),
    )
    if runtime_scale_shape:
        shape.update(runtime_scale_shape)
    fallback_aion_feedback = load_outcome_feedback(root=ROOT, mark_stale_reason=False)
    aion_shape, aion_issues = _overlay_aion_feedback_metrics_with_fallback(
        overlay,
        fallback_feedback=fallback_aion_feedback,
        source_pref=aion_source_pref,
    )
    if aion_shape:
        shape.update(aion_shape)
    stale_stats, stale_issues = _staleness_issues(
        checks,
        max_required_hours=float(np.clip(float(os.getenv("Q_SYSTEM_HEALTH_MAX_REQUIRED_HOURS", "24")), 1.0, 240.0)),
        max_optional_hours=float(np.clip(float(os.getenv("Q_SYSTEM_HEALTH_MAX_OPTIONAL_HOURS", "72")), 1.0, 720.0)),
    )
    if stale_stats:
        shape.update(stale_stats)

    # Alignment diagnostics
    issues = []
    if w is None:
        issues.append("missing portfolio_weights_final.csv")
    if daily is not None and w is not None and len(daily) < int(0.5 * w.shape[0]):
        issues.append("daily_returns much shorter than final weights")
    if gov is not None and w is not None and len(gov) < int(0.5 * w.shape[0]):
        issues.append("global_governor much shorter than final weights")
    if w is not None:
        bad = np.isnan(w).any() or np.isinf(w).any()
        if bad:
            issues.append("portfolio_weights_final contains NaN/Inf")
    issues.extend(exec_issues)
    issues.extend(cross_issues)
    issues.extend(nsp_issues)
    issues.extend(replay_issues)
    issues.extend(runtime_scale_issues)
    issues.extend(aion_issues)
    issues.extend(stale_issues)

    required_ok = sum(1 for c in checks if c["required"] and c["exists"])
    required_total = sum(1 for c in checks if c["required"])
    optional_ok = sum(1 for c in checks if (not c["required"]) and c["exists"])
    optional_total = sum(1 for c in checks if not c["required"])
    health_score = 100.0 * (0.75 * (required_ok / max(1, required_total)) + 0.25 * (optional_ok / max(1, optional_total)))
    if issues:
        health_score = max(0.0, health_score - 10.0 * len(issues))

    aion_summary = {
        "source_preference": aion_source_pref,
        "source": shape.get("aion_feedback_source", "unknown"),
        "source_selected": shape.get("aion_feedback_source_selected", shape.get("aion_feedback_source", "unknown")),
        "active": bool(shape.get("aion_feedback_active", False)),
        "status": str(shape.get("aion_feedback_status", "unknown")),
        "stale": bool(shape.get("aion_feedback_stale", False)),
        "age_hours": shape.get("aion_feedback_age_hours", None),
        "max_age_hours": shape.get("aion_feedback_max_age_hours", None),
        "closed_trades": shape.get("aion_feedback_closed_trades", None),
        "risk_scale": shape.get("aion_feedback_risk_scale", None),
    }

    out = {
        "timestamp_utc": _now().isoformat(),
        "health_score": float(health_score),
        "required_ok": int(required_ok),
        "required_total": int(required_total),
        "optional_ok": int(optional_ok),
        "optional_total": int(optional_total),
        "aion_feedback": aion_summary,
        "checks": checks,
        "shape": shape,
        "issues": issues,
    }
    (RUNS / "system_health.json").write_text(json.dumps(out, indent=2))

    html = (
        f"<p>Health score: <b>{health_score:.1f}</b> "
        f"(required {required_ok}/{required_total}, optional {optional_ok}/{optional_total})</p>"
        f"<p>AION feedback: source={aion_summary['source']} (selected={aion_summary['source_selected']}) "
        f"(pref={aion_summary['source_preference']}), status={aion_summary['status']}, "
        f"stale={aion_summary['stale']}, age_h={aion_summary['age_hours']}</p>"
        f"<p>Issues: {', '.join(issues) if issues else 'none'}</p>"
    )
    _append_card("System Health ✔", html)
    print(f"✅ Wrote {RUNS/'system_health.json'}")
