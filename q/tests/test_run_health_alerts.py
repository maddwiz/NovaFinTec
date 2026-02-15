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


def test_build_alert_payload_triggers_heartbeat_stress_alert():
    payload = rha.build_alert_payload(
        health={"health_score": 92, "issues": [], "shape": {"heartbeat_stress_mean": 0.92}},
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
            "require_immune_pass": False,
            "max_health_issues": 2,
            "min_nested_sharpe": 0.2,
            "min_nested_assets": 3,
            "max_shock_rate": 0.25,
            "max_concentration_hhi_after": 0.18,
            "max_concentration_top1_after": 0.30,
            "max_portfolio_l1_drift": 1.2,
            "max_heartbeat_stress": 0.85,
        },
    )
    assert any("heartbeat_stress_mean>" in a for a in payload["alerts"])


def test_build_alert_payload_triggers_execution_retention_alerts():
    payload = rha.build_alert_payload(
        health={
            "health_score": 92,
            "issues": [],
            "shape": {
                "exec_gross_before_mean": 0.40,
                "exec_gross_after_mean": 0.01,
                "exec_turnover_before_mean": 0.20,
                "exec_turnover_after_mean": 0.002,
            },
        },
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
            "require_immune_pass": False,
            "max_health_issues": 2,
            "min_nested_sharpe": 0.2,
            "min_nested_assets": 3,
            "max_shock_rate": 0.25,
            "max_concentration_hhi_after": 0.18,
            "max_concentration_top1_after": 0.30,
            "max_portfolio_l1_drift": 1.2,
            "min_exec_gross_retention": 0.10,
            "min_exec_turnover_retention": 0.05,
            "max_exec_turnover_retention": 1.10,
        },
    )
    assert any("exec_gross_retention<" in a for a in payload["alerts"])
    assert any("exec_turnover_retention<" in a for a in payload["alerts"])


def test_build_alert_payload_triggers_execution_issue_text():
    payload = rha.build_alert_payload(
        health={
            "health_score": 95,
            "issues": ["execution constraints may be over-throttling turnover"],
        },
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
    assert any("execution_constraints_over_throttling" in a for a in payload["alerts"])


def test_build_alert_payload_triggers_stale_required_alert():
    payload = rha.build_alert_payload(
        health={
            "health_score": 95,
            "issues": [],
            "shape": {"stale_required_count": 2},
        },
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
            "require_immune_pass": False,
            "max_health_issues": 2,
            "min_nested_sharpe": 0.2,
            "min_nested_assets": 3,
            "max_shock_rate": 0.25,
            "max_concentration_hhi_after": 0.18,
            "max_concentration_top1_after": 0.30,
            "max_portfolio_l1_drift": 1.2,
            "max_stale_required_count": 0,
        },
    )
    assert any("stale_required_count>" in a for a in payload["alerts"])


def test_build_alert_payload_triggers_quality_governor_step_alert():
    payload = rha.build_alert_payload(
        health={"health_score": 95, "issues": []},
        guards={"global_governor": {"mean": 0.85}},
        nested={"assets": 4, "avg_oos_sharpe": 0.8},
        quality={
            "quality_governor_mean": 0.88,
            "quality_score": 0.72,
            "quality_governor_max_abs_step": 0.18,
        },
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
            "require_immune_pass": False,
            "max_health_issues": 2,
            "min_nested_sharpe": 0.2,
            "min_nested_assets": 3,
            "max_shock_rate": 0.25,
            "max_concentration_hhi_after": 0.18,
            "max_concentration_top1_after": 0.30,
            "max_portfolio_l1_drift": 1.2,
            "max_quality_governor_abs_step": 0.12,
        },
    )
    assert any("quality_governor_abs_step>" in a for a in payload["alerts"])
