import pandas as pd

import tools.run_health_alerts as rha
import tools.run_quality_governor as rqg
import tools.run_system_health as rsh


def _base_thresholds():
    return {
        "min_health_score": 70,
        "min_global_governor_mean": 0.45,
        "min_quality_gov_mean": 0.60,
        "min_quality_score": 0.45,
        "require_immune_pass": False,
        "max_health_issues": 2,
        "min_nested_sharpe": 0.2,
        "min_nested_assets": 3,
        "max_shock_rate": 0.25,
        "max_concentration_hhi_after": 0.18,
        "max_concentration_top1_after": 0.30,
        "max_portfolio_l1_drift": 1.2,
        "min_aion_feedback_risk_scale": 0.80,
        "min_aion_feedback_closed_trades": 8,
        "min_aion_feedback_hit_rate": 0.38,
        "min_aion_feedback_profit_factor": 0.78,
        "max_aion_feedback_age_hours": 24.0,
    }


def test_ops_surfaces_prefer_overlay_when_both_sources_fresh():
    overlay = {
        "runtime_context": {
            "aion_feedback": {
                "active": True,
                "status": "ok",
                "risk_scale": 0.97,
                "closed_trades": 20,
                "age_hours": 2.0,
                "max_age_hours": 24.0,
                "stale": False,
            }
        }
    }
    shadow = {
        "active": True,
        "status": "alert",
        "risk_scale": 0.70,
        "closed_trades": 20,
        "age_hours": 3.0,
        "max_age_hours": 24.0,
        "stale": False,
    }

    metrics, _issues = rsh._overlay_aion_feedback_metrics_with_fallback(
        overlay, fallback_feedback=shadow, source_pref="auto"
    )
    assert metrics.get("aion_feedback_source") == "overlay"
    assert metrics.get("aion_feedback_status") == "ok"

    payload = rha.build_alert_payload(
        health={"health_score": 95, "issues": [], "shape": {}},
        guards={"global_governor": {"mean": 0.85}},
        nested={"assets": 4, "avg_oos_sharpe": 0.8},
        quality={"quality_governor_mean": 0.88, "quality_score": 0.72},
        immune={"ok": True, "pass": True},
        pipeline={"failed_count": 0},
        shock={"shock_rate": 0.05},
        concentration={"stats": {"hhi_after": 0.12, "top1_after": 0.18}},
        drift_watch={"drift": {"status": "ok", "latest_l1": 0.5}},
        fracture={"state": "stable", "latest_score": 0.22},
        overlay=overlay,
        aion_feedback_fallback=shadow,
        thresholds=_base_thresholds(),
    )
    assert payload["observed"]["aion_feedback_source"] == "overlay"
    assert payload["observed"]["aion_feedback_status"] == "ok"


def test_ops_surfaces_fall_back_to_shadow_when_overlay_stale():
    overlay = {
        "runtime_context": {
            "aion_feedback": {
                "active": True,
                "status": "alert",
                "risk_scale": 0.70,
                "closed_trades": 20,
                "age_hours": 96.0,
                "max_age_hours": 24.0,
                "stale": True,
            }
        }
    }
    shadow = {
        "active": True,
        "status": "ok",
        "risk_scale": 0.97,
        "closed_trades": 20,
        "age_hours": 2.0,
        "max_age_hours": 24.0,
        "stale": False,
    }

    metrics, _issues = rsh._overlay_aion_feedback_metrics_with_fallback(
        overlay, fallback_feedback=shadow, source_pref="auto"
    )
    assert metrics.get("aion_feedback_source") == "shadow_trades"
    assert metrics.get("aion_feedback_status") == "ok"

    payload = rha.build_alert_payload(
        health={"health_score": 95, "issues": [], "shape": {}},
        guards={"global_governor": {"mean": 0.85}},
        nested={"assets": 4, "avg_oos_sharpe": 0.8},
        quality={"quality_governor_mean": 0.88, "quality_score": 0.72},
        immune={"ok": True, "pass": True},
        pipeline={"failed_count": 0},
        shock={"shock_rate": 0.05},
        concentration={"stats": {"hhi_after": 0.12, "top1_after": 0.18}},
        drift_watch={"drift": {"status": "ok", "latest_l1": 0.5}},
        fracture={"state": "stable", "latest_score": 0.22},
        overlay=overlay,
        aion_feedback_fallback=shadow,
        thresholds=_base_thresholds(),
    )
    assert payload["observed"]["aion_feedback_source"] == "shadow_trades"
    assert payload["observed"]["aion_feedback_status"] == "ok"


def test_quality_governor_auto_prefers_shadow_ground_truth(monkeypatch, tmp_path):
    monkeypatch.setattr(rqg, "RUNS", tmp_path)
    shadow = tmp_path / "shadow_trades.csv"
    pd.DataFrame(
        {
            "timestamp": ["2026-02-16 10:00:00", "2026-02-16 10:05:00"],
            "side": ["EXIT_BUY", "EXIT_SELL"],
            "pnl": [4.0, -1.0],
        }
    ).to_csv(shadow, index=False)
    monkeypatch.setenv("Q_AION_SHADOW_TRADES", str(shadow))

    overlay = tmp_path / "q_signal_overlay.json"
    overlay.write_text(
        '{"runtime_context":{"aion_feedback":{"active":true,"status":"ok","risk_scale":0.99,"closed_trades":20}}}'
    )

    fb, src = rqg._load_aion_feedback()
    assert src["source"] == "shadow_trades"
    assert int(fb["closed_trades"]) == 2
