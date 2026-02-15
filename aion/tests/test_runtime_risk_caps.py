from aion.exec.paper_loop import _runtime_position_risk_scale, _runtime_risk_caps


def test_runtime_risk_caps_apply_scale_neutral_case():
    trades, opens = _runtime_risk_caps(
        max_trades_cap=12,
        max_open_positions_cap=6,
        ext_runtime_scale=0.90,
        ext_runtime_diag={"flags": [], "degraded": False, "quality_gate_ok": True, "regime": "balanced"},
    )
    assert trades == 11
    assert opens == 5


def test_runtime_risk_caps_fracture_warn_tightens_open_positions():
    trades, opens = _runtime_risk_caps(
        max_trades_cap=15,
        max_open_positions_cap=7,
        ext_runtime_scale=1.00,
        ext_runtime_diag={"flags": ["fracture_warn"], "degraded": False, "quality_gate_ok": True, "regime": "risk_on"},
    )
    assert trades == 15
    assert opens <= 3


def test_runtime_risk_caps_drift_alert_tightens_both_caps():
    trades, opens = _runtime_risk_caps(
        max_trades_cap=20,
        max_open_positions_cap=8,
        ext_runtime_scale=1.00,
        ext_runtime_diag={"flags": ["drift_alert"], "degraded": False, "quality_gate_ok": True, "regime": "balanced"},
    )
    assert trades <= 12
    assert opens <= 2


def test_runtime_risk_caps_quality_step_spike_tightens_caps():
    trades, opens = _runtime_risk_caps(
        max_trades_cap=16,
        max_open_positions_cap=7,
        ext_runtime_scale=1.00,
        ext_runtime_diag={
            "flags": ["quality_governor_step_spike"],
            "degraded": False,
            "quality_gate_ok": True,
            "regime": "risk_on",
        },
    )
    assert trades <= 12
    assert opens <= 3


def test_runtime_risk_caps_fracture_alert_tightens_both_caps():
    trades, opens = _runtime_risk_caps(
        max_trades_cap=20,
        max_open_positions_cap=8,
        ext_runtime_scale=1.00,
        ext_runtime_diag={"flags": ["fracture_alert"], "degraded": False, "quality_gate_ok": True, "regime": "balanced"},
    )
    assert trades <= 12
    assert opens <= 2


def test_runtime_risk_caps_degraded_defensive_also_tightens():
    trades, opens = _runtime_risk_caps(
        max_trades_cap=10,
        max_open_positions_cap=5,
        ext_runtime_scale=0.95,
        ext_runtime_diag={"flags": [], "degraded": True, "quality_gate_ok": False, "regime": "defensive"},
    )
    assert trades <= 10
    assert opens <= 4


def test_runtime_risk_caps_exec_risk_hard_tightens_caps():
    trades, opens = _runtime_risk_caps(
        max_trades_cap=18,
        max_open_positions_cap=7,
        ext_runtime_scale=1.0,
        ext_runtime_diag={"flags": ["exec_risk_hard"], "degraded": False, "quality_gate_ok": True, "regime": "risk_on"},
    )
    assert trades <= 10
    assert opens <= 2


def test_runtime_position_risk_scale_tightens_for_exec_hard_flag():
    s_tight = _runtime_position_risk_scale(
        ext_runtime_scale=1.0,
        ext_runtime_diag={"flags": ["exec_risk_tight"], "degraded": False, "quality_gate_ok": True, "regime": "risk_on"},
    )
    s_hard = _runtime_position_risk_scale(
        ext_runtime_scale=1.0,
        ext_runtime_diag={"flags": ["exec_risk_hard"], "degraded": False, "quality_gate_ok": True, "regime": "risk_on"},
    )
    assert 0.2 <= s_hard < s_tight < 1.0


def test_runtime_position_risk_scale_drift_alert_tighter_than_warn():
    s_warn = _runtime_position_risk_scale(
        ext_runtime_scale=1.0,
        ext_runtime_diag={"flags": ["drift_warn"], "degraded": False, "quality_gate_ok": True, "regime": "risk_on"},
    )
    s_alert = _runtime_position_risk_scale(
        ext_runtime_scale=1.0,
        ext_runtime_diag={"flags": ["drift_alert"], "degraded": False, "quality_gate_ok": True, "regime": "risk_on"},
    )
    assert 0.2 <= s_alert < s_warn < 1.0


def test_runtime_position_risk_scale_quality_step_spike_tightens():
    s = _runtime_position_risk_scale(
        ext_runtime_scale=1.0,
        ext_runtime_diag={
            "flags": ["quality_governor_step_spike"],
            "degraded": False,
            "quality_gate_ok": True,
            "regime": "risk_on",
        },
    )
    assert 0.2 <= s < 1.0


def test_runtime_position_risk_scale_compounds_degraded_and_fracture():
    s = _runtime_position_risk_scale(
        ext_runtime_scale=0.9,
        ext_runtime_diag={"flags": ["fracture_alert"], "degraded": True, "quality_gate_ok": False, "regime": "defensive"},
    )
    assert 0.2 <= s < 0.7


def test_runtime_risk_caps_nested_leakage_alert_tightens_caps():
    trades, opens = _runtime_risk_caps(
        max_trades_cap=20,
        max_open_positions_cap=8,
        ext_runtime_scale=1.0,
        ext_runtime_diag={"flags": ["nested_leakage_alert"], "degraded": False, "quality_gate_ok": True, "regime": "balanced"},
    )
    assert trades <= 13
    assert opens <= 2


def test_runtime_risk_caps_hive_stress_alert_tightens_caps():
    trades, opens = _runtime_risk_caps(
        max_trades_cap=20,
        max_open_positions_cap=8,
        ext_runtime_scale=1.0,
        ext_runtime_diag={"flags": ["hive_stress_alert"], "degraded": False, "quality_gate_ok": True, "regime": "balanced"},
    )
    assert trades <= 12
    assert opens <= 2


def test_runtime_position_risk_scale_hive_alert_tighter_than_hive_warn():
    s_warn = _runtime_position_risk_scale(
        ext_runtime_scale=1.0,
        ext_runtime_diag={"flags": ["hive_stress_warn"], "degraded": False, "quality_gate_ok": True, "regime": "balanced"},
    )
    s_alert = _runtime_position_risk_scale(
        ext_runtime_scale=1.0,
        ext_runtime_diag={"flags": ["hive_stress_alert"], "degraded": False, "quality_gate_ok": True, "regime": "balanced"},
    )
    assert 0.2 <= s_alert < s_warn < 1.0
