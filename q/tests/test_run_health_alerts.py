import tools.run_health_alerts as rha


def test_build_alert_payload_triggers_shock_and_concentration_alerts():
    payload = rha.build_alert_payload(
        health={"health_score": 90, "issues": []},
        guards={"global_governor": {"mean": 0.8}},
        nested={"assets": 4, "avg_oos_sharpe": 0.5},
        quality={"quality_governor_mean": 0.9, "quality_score": 0.8},
        immune={"ok": True, "pass": True},
        pipeline={"failed_count": 0},
        shock={"shock_rate": 0.40},
        concentration={"stats": {"hhi_after": 0.22, "top1_after": 0.35}},
        drift_watch={"drift": {"status": "ok", "latest_l1": 1.5}},
        thresholds={
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
        },
    )
    alerts = payload["alerts"]
    assert any("shock_rate>" in a for a in alerts)
    assert any("concentration_hhi_after>" in a for a in alerts)
    assert any("concentration_top1_after>" in a for a in alerts)
    assert any("portfolio_latest_l1_drift>" in a for a in alerts)
    assert payload["ok"] is False


def test_build_alert_payload_ok_when_metrics_inside_limits():
    payload = rha.build_alert_payload(
        health={"health_score": 92, "issues": []},
        guards={"global_governor": {"mean": 0.85}},
        nested={"assets": 4, "avg_oos_sharpe": 0.8},
        quality={"quality_governor_mean": 0.88, "quality_score": 0.72},
        immune={"ok": True, "pass": True},
        pipeline={"failed_count": 0},
        shock={"shock_rate": 0.05},
        concentration={"stats": {"hhi_after": 0.12, "top1_after": 0.18}},
        drift_watch={"drift": {"status": "ok", "latest_l1": 0.5}},
        thresholds={
            "min_health_score": 70,
            "min_global_governor_mean": 0.45,
            "min_quality_gov_mean": 0.60,
            "min_quality_score": 0.45,
            "require_immune_pass": True,
            "max_health_issues": 2,
            "min_nested_sharpe": 0.2,
            "min_nested_assets": 3,
            "max_shock_rate": 0.25,
            "max_concentration_hhi_after": 0.18,
            "max_concentration_top1_after": 0.30,
            "max_portfolio_l1_drift": 1.2,
        },
    )
    assert payload["alerts"] == []
    assert payload["ok"] is True


def test_build_alert_payload_triggers_dream_coherence_alert():
    payload = rha.build_alert_payload(
        health={"health_score": 90, "issues": []},
        guards={"global_governor": {"mean": 0.8}},
        nested={"assets": 4, "avg_oos_sharpe": 0.6},
        quality={
            "quality_governor_mean": 0.9,
            "quality_score": 0.7,
            "components": {"dream_coherence": {"score": 0.30}},
        },
        immune={"ok": True, "pass": True},
        pipeline={"failed_count": 0},
        shock={"shock_rate": 0.05},
        concentration={"stats": {"hhi_after": 0.12, "top1_after": 0.18}},
        drift_watch={"drift": {"status": "ok", "latest_l1": 0.4}},
        thresholds={
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
            "min_dream_coherence": 0.45,
        },
    )
    assert any("dream_coherence<" in a for a in payload["alerts"])
    assert payload["ok"] is False
