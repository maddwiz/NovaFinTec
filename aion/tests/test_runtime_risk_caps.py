from aion.exec.paper_loop import _runtime_risk_caps


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
