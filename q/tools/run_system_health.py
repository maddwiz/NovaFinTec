#!/usr/bin/env python3
# System health snapshot for Q pipeline.
#
# Writes:
#   runs_plus/system_health.json

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
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
    nsp_ctx_boost = _load_series(RUNS / "novaspine_context_boost.csv")
    nsp_hive_boost = _load_series(RUNS / "novaspine_hive_boost.csv")
    gov_trace_total = _load_series(RUNS / "final_governor_trace.csv")
    hb_stress = _load_series(RUNS / "heartbeat_stress.csv")
    exec_info = _load_json(RUNS / "execution_constraints_info.json")

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
    exec_shape, exec_issues = _analyze_execution_constraints(exec_info)
    if exec_shape:
        shape.update(exec_shape)
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
    issues.extend(stale_issues)

    required_ok = sum(1 for c in checks if c["required"] and c["exists"])
    required_total = sum(1 for c in checks if c["required"])
    optional_ok = sum(1 for c in checks if (not c["required"]) and c["exists"])
    optional_total = sum(1 for c in checks if not c["required"])
    health_score = 100.0 * (0.75 * (required_ok / max(1, required_total)) + 0.25 * (optional_ok / max(1, optional_total)))
    if issues:
        health_score = max(0.0, health_score - 10.0 * len(issues))

    out = {
        "timestamp_utc": _now().isoformat(),
        "health_score": float(health_score),
        "required_ok": int(required_ok),
        "required_total": int(required_total),
        "optional_ok": int(optional_ok),
        "optional_total": int(optional_total),
        "checks": checks,
        "shape": shape,
        "issues": issues,
    }
    (RUNS / "system_health.json").write_text(json.dumps(out, indent=2))

    html = (
        f"<p>Health score: <b>{health_score:.1f}</b> "
        f"(required {required_ok}/{required_total}, optional {optional_ok}/{optional_total})</p>"
        f"<p>Issues: {', '.join(issues) if issues else 'none'}</p>"
    )
    _append_card("System Health ✔", html)
    print(f"✅ Wrote {RUNS/'system_health.json'}")
