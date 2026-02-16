#!/usr/bin/env python3
# Alert gate for unattended runs.
#
# Reads:
#   runs_plus/system_health.json
#   runs_plus/guardrails_summary.json (optional)
#
# Exits:
#   0 if healthy enough
#   2 if alert conditions are met

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmods.aion_feedback import choose_feedback_source, feedback_has_metrics, load_outcome_feedback  # noqa: E402

RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _normalize_source_tag(raw: str | None, default: str = "unknown") -> str:
    tag = str(raw or "").strip().lower() or str(default)
    if tag == "shadow":
        return "shadow_trades"
    return tag


def _normalize_source_preference(raw: str | None) -> str:
    tag = str(raw or "").strip().lower() or "auto"
    if tag in {"auto", "overlay", "shadow"}:
        return tag
    return "auto"


def build_alert_payload(
    health: dict,
    guards: dict,
    nested: dict,
    quality: dict,
    immune: dict,
    pipeline: dict,
    shock: dict,
    concentration: dict,
    drift_watch: dict,
    thresholds: dict,
    fracture: dict | None = None,
    overlay: dict | None = None,
    aion_feedback_fallback: dict | None = None,
    aion_feedback_source_pref: str = "auto",
):
    min_health = float(thresholds.get("min_health_score", 70.0))
    min_global = float(thresholds.get("min_global_governor_mean", 0.45))
    min_quality_gov = float(thresholds.get("min_quality_gov_mean", 0.60))
    min_quality_score = float(thresholds.get("min_quality_score", 0.45))
    require_immune_pass = bool(thresholds.get("require_immune_pass", False))
    max_issues = int(thresholds.get("max_health_issues", 2))
    min_nested_sharpe = float(thresholds.get("min_nested_sharpe", 0.20))
    min_nested_assets = int(thresholds.get("min_nested_assets", 3))
    max_shock_rate = float(thresholds.get("max_shock_rate", 0.25))
    max_conc_hhi_after = float(thresholds.get("max_concentration_hhi_after", 0.18))
    max_conc_top1_after = float(thresholds.get("max_concentration_top1_after", 0.30))
    max_portfolio_l1_drift = float(thresholds.get("max_portfolio_l1_drift", 1.20))
    min_dream_coherence = float(thresholds.get("min_dream_coherence", 0.45))
    max_heartbeat_stress = float(thresholds.get("max_heartbeat_stress", 0.85))
    min_exec_gross_retention = float(thresholds.get("min_exec_gross_retention", 0.10))
    min_exec_turnover_retention = float(thresholds.get("min_exec_turnover_retention", 0.05))
    max_exec_turnover_retention = float(thresholds.get("max_exec_turnover_retention", 1.10))
    max_stale_required = int(thresholds.get("max_stale_required_count", 0))
    max_quality_gov_step = float(thresholds.get("max_quality_governor_abs_step", 0.12))
    max_fracture_score = float(thresholds.get("max_fracture_score", 0.78))
    max_hive_crowding_mean = float(thresholds.get("max_hive_crowding_mean", 0.65))
    max_hive_entropy_strength_mean = float(thresholds.get("max_hive_entropy_strength_mean", 0.90))
    max_hive_entropy_target_mean = float(thresholds.get("max_hive_entropy_target_mean", 0.84))
    min_aion_feedback_risk_scale = float(thresholds.get("min_aion_feedback_risk_scale", 0.80))
    min_aion_feedback_closed_trades = int(thresholds.get("min_aion_feedback_closed_trades", 8))
    min_aion_feedback_hit_rate = float(thresholds.get("min_aion_feedback_hit_rate", 0.38))
    min_aion_feedback_profit_factor = float(thresholds.get("min_aion_feedback_profit_factor", 0.78))
    max_aion_feedback_age_hours = float(thresholds.get("max_aion_feedback_age_hours", 72.0))

    issues = []
    score = float(health.get("health_score", 0.0))
    n_issues = len(health.get("issues", []) or [])
    if score < min_health:
        issues.append(f"health_score<{min_health} ({score:.1f})")
    if n_issues > max_issues:
        issues.append(f"health_issues>{max_issues} ({n_issues})")
    health_issue_text = [str(x).lower() for x in (health.get("issues", []) or [])]
    if any("over-throttling turnover" in x for x in health_issue_text):
        issues.append("execution_constraints_over_throttling")
    if any("collapsed gross exposure" in x for x in health_issue_text):
        issues.append("execution_constraints_collapsed_gross")

    hb_stress = None
    exec_gross_before = None
    exec_gross_after = None
    exec_turn_before = None
    exec_turn_after = None
    exec_gross_ret = None
    exec_turn_ret = None
    stale_required_count = None
    hive_crowding_mean = None
    hive_entropy_strength_mean = None
    hive_entropy_target_mean = None
    aion_feedback_active = False
    aion_feedback_status = "unknown"
    aion_feedback_risk_scale = None
    aion_feedback_closed_trades = None
    aion_feedback_hit_rate = None
    aion_feedback_profit_factor = None
    aion_feedback_age_hours = None
    aion_feedback_stale = False
    aion_feedback_max_age_hours = max_aion_feedback_age_hours
    aion_feedback_source = "none"
    aion_feedback_source_selected = "none"
    aion_feedback_source_preference = _normalize_source_preference(aion_feedback_source_pref)
    shape = {}
    if isinstance(health, dict):
        shape = health.get("shape", {})
        if isinstance(shape, dict):
            try:
                hb_stress = float(shape.get("heartbeat_stress_mean", np.nan))
            except Exception:
                hb_stress = None
            try:
                exec_gross_before = float(shape.get("exec_gross_before_mean", np.nan))
            except Exception:
                exec_gross_before = None
            try:
                exec_gross_after = float(shape.get("exec_gross_after_mean", np.nan))
            except Exception:
                exec_gross_after = None
            try:
                exec_turn_before = float(shape.get("exec_turnover_before_mean", np.nan))
            except Exception:
                exec_turn_before = None
            try:
                exec_turn_after = float(shape.get("exec_turnover_after_mean", np.nan))
            except Exception:
                exec_turn_after = None
            try:
                stale_required_count = int(shape.get("stale_required_count"))
            except Exception:
                stale_required_count = None
            try:
                hive_crowding_mean = float(shape.get("hive_crowding_mean", np.nan))
            except Exception:
                hive_crowding_mean = None
            try:
                hive_entropy_strength_mean = float(shape.get("hive_entropy_strength_mean", np.nan))
            except Exception:
                hive_entropy_strength_mean = None
            try:
                hive_entropy_target_mean = float(shape.get("hive_entropy_target_mean", np.nan))
            except Exception:
                hive_entropy_target_mean = None
    if hb_stress is not None and np.isfinite(hb_stress) and hb_stress > max_heartbeat_stress:
        issues.append(f"heartbeat_stress_mean>{max_heartbeat_stress} ({hb_stress:.3f})")
    if (
        exec_gross_before is not None
        and exec_gross_after is not None
        and np.isfinite(exec_gross_before)
        and np.isfinite(exec_gross_after)
        and exec_gross_before > 1e-9
    ):
        exec_gross_ret = float(np.clip(exec_gross_after / exec_gross_before, 0.0, 5.0))
        if exec_gross_before >= 0.10 and exec_gross_ret < min_exec_gross_retention:
            issues.append(f"exec_gross_retention<{min_exec_gross_retention} ({exec_gross_ret:.3f})")
    if (
        exec_turn_before is not None
        and exec_turn_after is not None
        and np.isfinite(exec_turn_before)
        and np.isfinite(exec_turn_after)
        and exec_turn_before > 1e-9
    ):
        exec_turn_ret = float(np.clip(exec_turn_after / exec_turn_before, 0.0, 5.0))
        if exec_turn_before >= 0.02 and exec_turn_ret < min_exec_turnover_retention:
            issues.append(f"exec_turnover_retention<{min_exec_turnover_retention} ({exec_turn_ret:.3f})")
        if exec_turn_before >= 0.02 and exec_turn_ret > max_exec_turnover_retention:
            issues.append(f"exec_turnover_retention>{max_exec_turnover_retention} ({exec_turn_ret:.3f})")
    if stale_required_count is not None and stale_required_count > max_stale_required:
        issues.append(f"stale_required_count>{max_stale_required} ({stale_required_count})")
    if hive_crowding_mean is not None and np.isfinite(hive_crowding_mean) and hive_crowding_mean > max_hive_crowding_mean:
        issues.append(f"hive_crowding_mean>{max_hive_crowding_mean} ({hive_crowding_mean:.3f})")
    if (
        hive_entropy_strength_mean is not None
        and np.isfinite(hive_entropy_strength_mean)
        and hive_entropy_strength_mean > max_hive_entropy_strength_mean
    ):
        issues.append(
            f"hive_entropy_strength_mean>{max_hive_entropy_strength_mean} ({hive_entropy_strength_mean:.3f})"
        )
    if (
        hive_entropy_target_mean is not None
        and np.isfinite(hive_entropy_target_mean)
        and hive_entropy_target_mean > max_hive_entropy_target_mean
    ):
        issues.append(f"hive_entropy_target_mean>{max_hive_entropy_target_mean} ({hive_entropy_target_mean:.3f})")

    overlay_af = None
    if isinstance(overlay, dict):
        rt = overlay.get("runtime_context", {})
        if isinstance(rt, dict):
            overlay_af = rt.get("aion_feedback", {})
    af, af_source = choose_feedback_source(
        overlay_af,
        aion_feedback_fallback,
        source_pref=aion_feedback_source_pref,
        prefer_overlay_when_fresh=True,
    )
    if feedback_has_metrics(af):
        aion_feedback_source_selected = _normalize_source_tag(af_source, default="unknown")
        source_from_payload = _normalize_source_tag(
            af.get("source", af.get("source_selected", "unknown")),
            default="unknown",
        )
        aion_feedback_source = (
            source_from_payload
            if source_from_payload not in {"", "unknown"}
            else aion_feedback_source_selected
        )
        aion_feedback_active = bool(af.get("active", False))
        aion_feedback_status = str(af.get("status", "unknown")).strip().lower() or "unknown"
        try:
            aion_feedback_risk_scale = float(af.get("risk_scale", np.nan))
        except Exception:
            aion_feedback_risk_scale = None
        try:
            aion_feedback_closed_trades = int(af.get("closed_trades", 0))
        except Exception:
            aion_feedback_closed_trades = None
        try:
            aion_feedback_hit_rate = float(af.get("hit_rate", np.nan))
        except Exception:
            aion_feedback_hit_rate = None
        try:
            aion_feedback_profit_factor = float(af.get("profit_factor", np.nan))
        except Exception:
            aion_feedback_profit_factor = None
        try:
            aion_feedback_age_hours = float(af.get("age_hours", np.nan))
        except Exception:
            aion_feedback_age_hours = None
        try:
            aion_feedback_stale = bool(af.get("stale", False))
        except Exception:
            aion_feedback_stale = False
        try:
            aion_feedback_max_age_hours = float(af.get("max_age_hours", max_aion_feedback_age_hours))
        except Exception:
            aion_feedback_max_age_hours = max_aion_feedback_age_hours

    if (not aion_feedback_active) and isinstance(shape, dict):
        try:
            aion_feedback_active = bool(shape.get("aion_feedback_active", False))
        except Exception:
            aion_feedback_active = False
        if aion_feedback_active:
            aion_feedback_source = "system_health_shape"
            aion_feedback_source_selected = "system_health_shape"
            aion_feedback_status = str(shape.get("aion_feedback_status", "unknown")).strip().lower() or "unknown"
            try:
                aion_feedback_risk_scale = float(shape.get("aion_feedback_risk_scale", np.nan))
            except Exception:
                aion_feedback_risk_scale = None
            try:
                aion_feedback_closed_trades = int(shape.get("aion_feedback_closed_trades", 0))
            except Exception:
                aion_feedback_closed_trades = None
            try:
                aion_feedback_hit_rate = float(shape.get("aion_feedback_hit_rate", np.nan))
            except Exception:
                aion_feedback_hit_rate = None
            try:
                aion_feedback_profit_factor = float(shape.get("aion_feedback_profit_factor", np.nan))
            except Exception:
                aion_feedback_profit_factor = None
            try:
                aion_feedback_age_hours = float(shape.get("aion_feedback_age_hours", np.nan))
            except Exception:
                aion_feedback_age_hours = None
            try:
                aion_feedback_stale = bool(shape.get("aion_feedback_stale", False))
            except Exception:
                aion_feedback_stale = False
            try:
                aion_feedback_max_age_hours = float(shape.get("aion_feedback_max_age_hours", max_aion_feedback_age_hours))
            except Exception:
                aion_feedback_max_age_hours = max_aion_feedback_age_hours

    if aion_feedback_active:
        if (
            aion_feedback_age_hours is not None
            and np.isfinite(aion_feedback_age_hours)
            and aion_feedback_age_hours > max_aion_feedback_age_hours
        ):
            aion_feedback_stale = True
        if aion_feedback_stale:
            age_txt = (
                f"{float(aion_feedback_age_hours):.2f}h"
                if (aion_feedback_age_hours is not None and np.isfinite(aion_feedback_age_hours))
                else "na"
            )
            issues.append(
                f"aion_feedback_stale>{max_aion_feedback_age_hours}h ({age_txt})"
            )
        elif aion_feedback_status in {"alert", "hard"}:
            issues.append("aion_feedback_status=alert")
        enough_closed = (
            aion_feedback_closed_trades is not None and aion_feedback_closed_trades >= min_aion_feedback_closed_trades
        )
        if enough_closed and (not aion_feedback_stale):
            if (
                aion_feedback_risk_scale is not None
                and np.isfinite(aion_feedback_risk_scale)
                and aion_feedback_risk_scale < min_aion_feedback_risk_scale
            ):
                issues.append(
                    f"aion_feedback_risk_scale<{min_aion_feedback_risk_scale} ({aion_feedback_risk_scale:.3f})"
                )
            if (
                aion_feedback_hit_rate is not None
                and np.isfinite(aion_feedback_hit_rate)
                and aion_feedback_hit_rate < min_aion_feedback_hit_rate
            ):
                issues.append(f"aion_feedback_hit_rate<{min_aion_feedback_hit_rate} ({aion_feedback_hit_rate:.3f})")
            if (
                aion_feedback_profit_factor is not None
                and np.isfinite(aion_feedback_profit_factor)
                and aion_feedback_profit_factor < min_aion_feedback_profit_factor
            ):
                issues.append(
                    f"aion_feedback_profit_factor<{min_aion_feedback_profit_factor} ({aion_feedback_profit_factor:.3f})"
                )

    gg = guards.get("global_governor", {}) if isinstance(guards, dict) else {}
    gmean = gg.get("mean", None)
    if gmean is not None:
        try:
            gmean = float(gmean)
            if gmean < min_global:
                issues.append(f"global_governor_mean<{min_global} ({gmean:.3f})")
        except Exception:
            pass

    n_assets = nested.get("assets", None)
    n_sh = nested.get("avg_oos_sharpe", None)
    try:
        n_assets = int(n_assets) if n_assets is not None else None
    except Exception:
        n_assets = None
    try:
        n_sh = float(n_sh) if n_sh is not None else None
    except Exception:
        n_sh = None
    if n_assets is not None and n_sh is not None and n_assets >= min_nested_assets and n_sh < min_nested_sharpe:
        issues.append(f"nested_wf_sharpe<{min_nested_sharpe} ({n_sh:.3f}) over {n_assets} assets")

    q_mean = quality.get("quality_governor_mean", None)
    q_score = quality.get("quality_score", None)
    q_step = quality.get("quality_governor_max_abs_step", None)
    try:
        q_mean = float(q_mean) if q_mean is not None else None
    except Exception:
        q_mean = None
    try:
        q_score = float(q_score) if q_score is not None else None
    except Exception:
        q_score = None
    try:
        q_step = float(q_step) if q_step is not None else None
    except Exception:
        q_step = None
    if q_mean is not None and q_mean < min_quality_gov:
        issues.append(f"quality_governor_mean<{min_quality_gov} ({q_mean:.3f})")
    if q_score is not None and q_score < min_quality_score:
        issues.append(f"quality_score<{min_quality_score} ({q_score:.3f})")
    if q_step is not None and np.isfinite(q_step) and q_step > max_quality_gov_step:
        issues.append(f"quality_governor_abs_step>{max_quality_gov_step} ({q_step:.3f})")

    dream_score = None
    if isinstance(quality, dict):
        comps = quality.get("components", {})
        if isinstance(comps, dict):
            d = comps.get("dream_coherence", {})
            if isinstance(d, dict):
                try:
                    dream_score = float(d.get("score", np.nan))
                except Exception:
                    dream_score = None
    if dream_score is not None and np.isfinite(dream_score) and dream_score < min_dream_coherence:
        issues.append(f"dream_coherence<{min_dream_coherence} ({dream_score:.3f})")

    shock_rate = None
    if isinstance(shock, dict):
        try:
            shock_rate = float(shock.get("shock_rate", np.nan))
        except Exception:
            shock_rate = None
    if shock_rate is not None and np.isfinite(shock_rate) and shock_rate > max_shock_rate:
        issues.append(f"shock_rate>{max_shock_rate} ({shock_rate:.3f})")

    conc_hhi_after = None
    conc_top1_after = None
    if isinstance(concentration, dict):
        st = concentration.get("stats", {})
        if isinstance(st, dict):
            try:
                conc_hhi_after = float(st.get("hhi_after", np.nan))
            except Exception:
                conc_hhi_after = None
            try:
                conc_top1_after = float(st.get("top1_after", np.nan))
            except Exception:
                conc_top1_after = None
    if conc_hhi_after is not None and np.isfinite(conc_hhi_after) and conc_hhi_after > max_conc_hhi_after:
        issues.append(f"concentration_hhi_after>{max_conc_hhi_after} ({conc_hhi_after:.3f})")
    if conc_top1_after is not None and np.isfinite(conc_top1_after) and conc_top1_after > max_conc_top1_after:
        issues.append(f"concentration_top1_after>{max_conc_top1_after} ({conc_top1_after:.3f})")

    drift_latest_l1 = None
    drift_status = None
    if isinstance(drift_watch, dict):
        d = drift_watch.get("drift", {})
        if isinstance(d, dict):
            drift_status = str(d.get("status", "na"))
            try:
                drift_latest_l1 = float(d.get("latest_l1", np.nan))
            except Exception:
                drift_latest_l1 = None
    if drift_latest_l1 is not None and np.isfinite(drift_latest_l1) and drift_status != "bootstrap":
        if drift_latest_l1 > max_portfolio_l1_drift:
            issues.append(f"portfolio_latest_l1_drift>{max_portfolio_l1_drift} ({drift_latest_l1:.3f})")
    if drift_status == "alert":
        issues.append("portfolio_drift_status=alert")

    fracture_score = None
    fracture_state = None
    if isinstance(fracture, dict):
        try:
            fracture_score = float(fracture.get("latest_score", np.nan))
        except Exception:
            fracture_score = None
        fracture_state = str(fracture.get("state", "na")).lower()
    if fracture_state == "fracture_alert":
        issues.append("regime_fracture_state=alert")
    if fracture_score is not None and np.isfinite(fracture_score) and fracture_score > max_fracture_score:
        issues.append(f"regime_fracture_score>{max_fracture_score} ({fracture_score:.3f})")

    immune_ok = bool(immune.get("ok", False)) if isinstance(immune, dict) else False
    immune_pass = bool(immune.get("pass", False)) if isinstance(immune, dict) else False
    if require_immune_pass:
        if not immune_ok:
            issues.append("immune_drill_missing_or_invalid")
        elif not immune_pass:
            issues.append("immune_drill_failed")

    failed_steps = int(pipeline.get("failed_count", 0)) if isinstance(pipeline, dict) else 0
    if failed_steps > 0:
        issues.append(f"pipeline_failed_steps={failed_steps}")

    payload = {
        "ok": len(issues) == 0,
        "thresholds": {
            "min_health_score": min_health,
            "max_health_issues": max_issues,
            "min_global_governor_mean": min_global,
            "min_quality_governor_mean": min_quality_gov,
            "min_quality_score": min_quality_score,
            "require_immune_pass": require_immune_pass,
            "min_nested_sharpe": min_nested_sharpe,
            "min_nested_assets": min_nested_assets,
            "max_shock_rate": max_shock_rate,
            "max_concentration_hhi_after": max_conc_hhi_after,
            "max_concentration_top1_after": max_conc_top1_after,
            "max_portfolio_l1_drift": max_portfolio_l1_drift,
            "min_dream_coherence": min_dream_coherence,
            "max_heartbeat_stress": max_heartbeat_stress,
            "min_exec_gross_retention": min_exec_gross_retention,
            "min_exec_turnover_retention": min_exec_turnover_retention,
            "max_exec_turnover_retention": max_exec_turnover_retention,
            "max_stale_required_count": max_stale_required,
            "max_quality_governor_abs_step": max_quality_gov_step,
            "max_fracture_score": max_fracture_score,
            "max_hive_crowding_mean": max_hive_crowding_mean,
            "max_hive_entropy_strength_mean": max_hive_entropy_strength_mean,
            "max_hive_entropy_target_mean": max_hive_entropy_target_mean,
            "min_aion_feedback_risk_scale": min_aion_feedback_risk_scale,
            "min_aion_feedback_closed_trades": min_aion_feedback_closed_trades,
            "min_aion_feedback_hit_rate": min_aion_feedback_hit_rate,
            "min_aion_feedback_profit_factor": min_aion_feedback_profit_factor,
            "max_aion_feedback_age_hours": max_aion_feedback_age_hours,
        },
        "observed": {
            "health_score": score,
            "health_issues": n_issues,
            "heartbeat_stress_mean": hb_stress,
            "exec_gross_before_mean": exec_gross_before,
            "exec_gross_after_mean": exec_gross_after,
            "exec_turnover_before_mean": exec_turn_before,
            "exec_turnover_after_mean": exec_turn_after,
            "exec_gross_retention": exec_gross_ret,
            "exec_turnover_retention": exec_turn_ret,
            "stale_required_count": stale_required_count,
            "hive_crowding_mean": hive_crowding_mean,
            "hive_entropy_strength_mean": hive_entropy_strength_mean,
            "hive_entropy_target_mean": hive_entropy_target_mean,
            "aion_feedback_active": aion_feedback_active,
            "aion_feedback_source": aion_feedback_source,
            "aion_feedback_source_selected": aion_feedback_source_selected,
            "aion_feedback_source_preference": aion_feedback_source_preference,
            "aion_feedback_status": aion_feedback_status,
            "aion_feedback_risk_scale": aion_feedback_risk_scale,
            "aion_feedback_closed_trades": aion_feedback_closed_trades,
            "aion_feedback_hit_rate": aion_feedback_hit_rate,
            "aion_feedback_profit_factor": aion_feedback_profit_factor,
            "aion_feedback_age_hours": aion_feedback_age_hours,
            "aion_feedback_stale": aion_feedback_stale,
            "aion_feedback_max_age_hours": aion_feedback_max_age_hours,
            "global_governor_mean": gmean,
            "quality_governor_mean": q_mean,
            "quality_score": q_score,
            "quality_governor_max_abs_step": q_step,
            "dream_coherence": dream_score,
            "immune_ok": immune_ok,
            "immune_pass": immune_pass,
            "nested_assets": n_assets,
            "nested_avg_oos_sharpe": n_sh,
            "pipeline_failed_steps": failed_steps,
            "shock_rate": shock_rate,
            "concentration_hhi_after": conc_hhi_after,
            "concentration_top1_after": conc_top1_after,
            "portfolio_latest_l1_drift": drift_latest_l1,
            "portfolio_drift_status": drift_status,
            "regime_fracture_score": fracture_score,
            "regime_fracture_state": fracture_state,
        },
        "alerts": issues,
    }
    return payload


if __name__ == "__main__":
    min_health = float(os.getenv("Q_MIN_HEALTH_SCORE", "70"))
    min_global = float(os.getenv("Q_MIN_GLOBAL_GOV_MEAN", "0.45"))
    min_quality_gov = float(os.getenv("Q_MIN_QUALITY_GOV_MEAN", "0.58"))
    min_quality_score = float(os.getenv("Q_MIN_QUALITY_SCORE", "0.45"))
    require_immune_pass = str(os.getenv("Q_REQUIRE_IMMUNE_PASS", "0")).strip().lower() in {"1", "true", "yes", "on"}
    max_issues = int(os.getenv("Q_MAX_HEALTH_ISSUES", "2"))
    min_nested_sharpe = float(os.getenv("Q_MIN_NESTED_SHARPE", "0.20"))
    min_nested_assets = int(os.getenv("Q_MIN_NESTED_ASSETS", "3"))
    max_shock_rate = float(os.getenv("Q_MAX_SHOCK_RATE", "0.25"))
    max_conc_hhi_after = float(os.getenv("Q_MAX_CONCENTRATION_HHI_AFTER", "0.18"))
    max_conc_top1_after = float(os.getenv("Q_MAX_CONCENTRATION_TOP1_AFTER", "0.30"))
    min_dream_coherence = float(os.getenv("Q_MIN_DREAM_COHERENCE", "0.45"))
    max_heartbeat_stress = float(os.getenv("Q_MAX_HEARTBEAT_STRESS", "0.85"))
    min_exec_gross_retention = float(os.getenv("Q_MIN_EXEC_GROSS_RETENTION", "0.10"))
    min_exec_turnover_retention = float(os.getenv("Q_MIN_EXEC_TURNOVER_RETENTION", "0.05"))
    max_exec_turnover_retention = float(os.getenv("Q_MAX_EXEC_TURNOVER_RETENTION", "1.10"))
    max_stale_required = int(os.getenv("Q_MAX_STALE_REQUIRED_COUNT", "0"))
    max_quality_gov_step = float(os.getenv("Q_MAX_QUALITY_GOV_ABS_STEP", "0.12"))
    max_hive_crowding_mean = float(os.getenv("Q_MAX_HIVE_CROWDING_MEAN", "0.65"))
    max_hive_entropy_strength_mean = float(os.getenv("Q_MAX_HIVE_ENTROPY_STRENGTH_MEAN", "0.90"))
    max_hive_entropy_target_mean = float(os.getenv("Q_MAX_HIVE_ENTROPY_TARGET_MEAN", "0.84"))
    aion_feedback_source_pref = str(os.getenv("Q_AION_FEEDBACK_SOURCE", "auto")).strip().lower() or "auto"
    max_aion_feedback_age_hours = float(
        os.getenv("Q_MAX_AION_FEEDBACK_AGE_HOURS", os.getenv("Q_AION_FEEDBACK_MAX_AGE_HOURS", "72"))
    )

    health = _load_json(RUNS / "system_health.json") or {}
    guards = _load_json(RUNS / "guardrails_summary.json") or {}
    nested = _load_json(RUNS / "nested_wf_summary.json") or {}
    quality = _load_json(RUNS / "quality_snapshot.json") or {}
    immune = _load_json(RUNS / "immune_drill.json") or {}
    pipeline = _load_json(RUNS / "pipeline_status.json") or {}
    shock = _load_json(RUNS / "shock_mask_info.json") or {}
    concentration = _load_json(RUNS / "concentration_governor_info.json") or {}
    drift_watch = _load_json(RUNS / "portfolio_drift_watch.json") or {}
    fracture = _load_json(RUNS / "regime_fracture_info.json") or {}
    overlay = _load_json(RUNS / "q_signal_overlay.json") or {}
    fallback_aion_feedback = load_outcome_feedback(root=ROOT, mark_stale_reason=False)
    payload = build_alert_payload(
        health=health,
        guards=guards,
        nested=nested,
        quality=quality,
        immune=immune,
        pipeline=pipeline,
        shock=shock,
        concentration=concentration,
        drift_watch=drift_watch,
        fracture=fracture,
        overlay=overlay,
        aion_feedback_fallback=fallback_aion_feedback,
        aion_feedback_source_pref=aion_feedback_source_pref,
        thresholds={
            "min_health_score": min_health,
            "min_global_governor_mean": min_global,
            "min_quality_gov_mean": min_quality_gov,
            "min_quality_score": min_quality_score,
            "require_immune_pass": require_immune_pass,
            "max_health_issues": max_issues,
            "min_nested_sharpe": min_nested_sharpe,
            "min_nested_assets": min_nested_assets,
            "max_shock_rate": max_shock_rate,
            "max_concentration_hhi_after": max_conc_hhi_after,
            "max_concentration_top1_after": max_conc_top1_after,
            "max_portfolio_l1_drift": float(os.getenv("Q_MAX_PORTFOLIO_L1_DRIFT", "1.20")),
            "min_dream_coherence": min_dream_coherence,
            "max_heartbeat_stress": max_heartbeat_stress,
            "min_exec_gross_retention": min_exec_gross_retention,
            "min_exec_turnover_retention": min_exec_turnover_retention,
            "max_exec_turnover_retention": max_exec_turnover_retention,
            "max_stale_required_count": max_stale_required,
            "max_quality_governor_abs_step": max_quality_gov_step,
            "max_fracture_score": float(os.getenv("Q_MAX_FRACTURE_SCORE", "0.78")),
            "max_hive_crowding_mean": max_hive_crowding_mean,
            "max_hive_entropy_strength_mean": max_hive_entropy_strength_mean,
            "max_hive_entropy_target_mean": max_hive_entropy_target_mean,
            "min_aion_feedback_risk_scale": float(os.getenv("Q_MIN_AION_FEEDBACK_RISK_SCALE", "0.80")),
            "min_aion_feedback_closed_trades": int(os.getenv("Q_MIN_AION_FEEDBACK_CLOSED_TRADES", "8")),
            "min_aion_feedback_hit_rate": float(os.getenv("Q_MIN_AION_FEEDBACK_HIT_RATE", "0.38")),
            "min_aion_feedback_profit_factor": float(os.getenv("Q_MIN_AION_FEEDBACK_PROFIT_FACTOR", "0.78")),
            "max_aion_feedback_age_hours": max_aion_feedback_age_hours,
        },
    )
    (RUNS / "health_alerts.json").write_text(json.dumps(payload, indent=2))
    print(f"✅ Wrote {RUNS/'health_alerts.json'}")
    if payload.get("alerts"):
        print("ALERT:", "; ".join(payload["alerts"]))
        raise SystemExit(2)
    print("✅ Health alerts clear")
