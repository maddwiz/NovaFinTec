from aion.exec.runtime_decision import runtime_decision_summary


def test_runtime_decision_summary_blocks_and_alerts():
    out = runtime_decision_summary(
        runtime_controls={
            "overlay_block_new_entries": True,
            "overlay_block_reasons": ["critical_flag:fracture_alert"],
            "external_position_risk_scale": 0.66,
            "external_runtime_scale": 0.74,
            "exec_governor_state": "alert",
        },
        external_overlay_runtime={"stale": False},
        external_overlay_risk_flags=["fracture_alert"],
    )
    assert out["entry_blocked"] is True
    assert any("external_overlay" in x for x in out["entry_block_reasons"])
    assert out["throttle_state"] == "alert"
    assert any(a.get("id") == "overlay_refresh" for a in out["recommended_actions"])


def test_runtime_decision_summary_normal_when_clean():
    out = runtime_decision_summary(
        runtime_controls={
            "overlay_block_new_entries": False,
            "policy_block_new_entries": False,
            "external_position_risk_scale": 1.0,
            "external_runtime_scale": 1.0,
            "exec_governor_state": "ok",
            "memory_feedback_status": "ok",
        },
        external_overlay_runtime={"stale": False},
        external_overlay_risk_flags=[],
    )
    assert out["entry_blocked"] is False
    assert out["throttle_state"] == "normal"
    assert out["recommended_actions"][0]["id"] == "no_action"


def test_runtime_decision_summary_hive_crowding_adds_targeted_action():
    out = runtime_decision_summary(
        runtime_controls={
            "overlay_block_new_entries": False,
            "policy_block_new_entries": False,
            "external_position_risk_scale": 0.88,
            "external_runtime_scale": 0.90,
            "exec_governor_state": "ok",
            "memory_feedback_status": "ok",
        },
        external_overlay_runtime={"stale": False},
        external_overlay_risk_flags=["hive_crowding_alert"],
    )
    assert out["throttle_state"] in {"warn", "alert"}
    assert "hive_crowding_alert" in out["throttle_reasons"]
    assert any(a.get("id") == "hive_crowding_rebalance" for a in out["recommended_actions"])


def test_runtime_decision_summary_hive_entropy_adds_targeted_action():
    out = runtime_decision_summary(
        runtime_controls={
            "overlay_block_new_entries": False,
            "policy_block_new_entries": False,
            "external_position_risk_scale": 0.90,
            "external_runtime_scale": 0.90,
            "exec_governor_state": "ok",
            "memory_feedback_status": "ok",
        },
        external_overlay_runtime={"stale": False},
        external_overlay_risk_flags=["hive_entropy_alert"],
    )
    assert out["throttle_state"] in {"warn", "alert"}
    assert "hive_entropy_alert" in out["throttle_reasons"]
    assert any(a.get("id") == "hive_entropy_regime_reset" for a in out["recommended_actions"])


def test_runtime_decision_summary_aion_outcome_adds_targeted_action():
    out = runtime_decision_summary(
        runtime_controls={
            "overlay_block_new_entries": False,
            "policy_block_new_entries": False,
            "external_position_risk_scale": 0.92,
            "external_runtime_scale": 0.92,
            "exec_governor_state": "ok",
            "memory_feedback_status": "ok",
        },
        external_overlay_runtime={"stale": False},
        external_overlay_risk_flags=["aion_outcome_alert"],
    )
    assert out["throttle_state"] in {"warn", "alert"}
    assert "aion_outcome_alert" in out["throttle_reasons"]
    assert any(a.get("id") == "aion_outcome_recalibration" for a in out["recommended_actions"])
