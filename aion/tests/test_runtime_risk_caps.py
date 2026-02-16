import aion.exec.paper_loop as pl
from aion.exec.paper_loop import _runtime_position_risk_scale, _runtime_risk_caps
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone


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


def test_runtime_risk_caps_overlay_stale_tightens_caps():
    trades, opens = _runtime_risk_caps(
        max_trades_cap=20,
        max_open_positions_cap=8,
        ext_runtime_scale=1.00,
        ext_runtime_diag={"flags": ["overlay_stale"], "degraded": False, "quality_gate_ok": True, "regime": "balanced"},
    )
    assert trades <= 14
    assert opens <= 2


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


def test_runtime_position_risk_scale_overlay_stale_tightens():
    s = _runtime_position_risk_scale(
        ext_runtime_scale=1.0,
        ext_runtime_diag={"flags": ["overlay_stale"], "degraded": False, "quality_gate_ok": True, "overlay_stale": True, "regime": "risk_on"},
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


def test_runtime_risk_caps_hive_crowding_alert_tightens_caps():
    trades, opens = _runtime_risk_caps(
        max_trades_cap=20,
        max_open_positions_cap=8,
        ext_runtime_scale=1.0,
        ext_runtime_diag={"flags": ["hive_crowding_alert"], "degraded": False, "quality_gate_ok": True, "regime": "balanced"},
    )
    assert trades <= 13
    assert opens <= 2


def test_runtime_position_risk_scale_hive_crowding_alert_tighter_than_warn():
    s_warn = _runtime_position_risk_scale(
        ext_runtime_scale=1.0,
        ext_runtime_diag={"flags": ["hive_crowding_warn"], "degraded": False, "quality_gate_ok": True, "regime": "balanced"},
    )
    s_alert = _runtime_position_risk_scale(
        ext_runtime_scale=1.0,
        ext_runtime_diag={"flags": ["hive_crowding_alert"], "degraded": False, "quality_gate_ok": True, "regime": "balanced"},
    )
    assert 0.2 <= s_alert < s_warn < 1.0


def test_runtime_risk_caps_hive_entropy_alert_tightens_caps():
    trades, opens = _runtime_risk_caps(
        max_trades_cap=20,
        max_open_positions_cap=8,
        ext_runtime_scale=1.0,
        ext_runtime_diag={"flags": ["hive_entropy_alert"], "degraded": False, "quality_gate_ok": True, "regime": "balanced"},
    )
    assert trades <= 14
    assert opens <= 3


def test_runtime_risk_caps_hive_turnover_alert_tightens_caps():
    trades, opens = _runtime_risk_caps(
        max_trades_cap=20,
        max_open_positions_cap=8,
        ext_runtime_scale=1.0,
        ext_runtime_diag={"flags": ["hive_turnover_alert"], "degraded": False, "quality_gate_ok": True, "regime": "balanced"},
    )
    assert trades <= 12
    assert opens <= 2


def test_runtime_position_risk_scale_hive_entropy_alert_tighter_than_warn():
    s_warn = _runtime_position_risk_scale(
        ext_runtime_scale=1.0,
        ext_runtime_diag={"flags": ["hive_entropy_warn"], "degraded": False, "quality_gate_ok": True, "regime": "balanced"},
    )
    s_alert = _runtime_position_risk_scale(
        ext_runtime_scale=1.0,
        ext_runtime_diag={"flags": ["hive_entropy_alert"], "degraded": False, "quality_gate_ok": True, "regime": "balanced"},
    )
    assert 0.2 <= s_alert < s_warn < 1.0


def test_runtime_position_risk_scale_hive_turnover_alert_tighter_than_warn():
    s_warn = _runtime_position_risk_scale(
        ext_runtime_scale=1.0,
        ext_runtime_diag={"flags": ["hive_turnover_warn"], "degraded": False, "quality_gate_ok": True, "regime": "balanced"},
    )
    s_alert = _runtime_position_risk_scale(
        ext_runtime_scale=1.0,
        ext_runtime_diag={"flags": ["hive_turnover_alert"], "degraded": False, "quality_gate_ok": True, "regime": "balanced"},
    )
    assert 0.2 <= s_alert < s_warn < 1.0


def test_runtime_risk_caps_heartbeat_alert_tightens_caps():
    trades, opens = _runtime_risk_caps(
        max_trades_cap=20,
        max_open_positions_cap=8,
        ext_runtime_scale=1.0,
        ext_runtime_diag={"flags": ["heartbeat_alert"], "degraded": False, "quality_gate_ok": True, "regime": "balanced"},
    )
    assert trades <= 12
    assert opens <= 2


def test_runtime_position_risk_scale_heartbeat_alert_tighter_than_warn():
    s_warn = _runtime_position_risk_scale(
        ext_runtime_scale=1.0,
        ext_runtime_diag={"flags": ["heartbeat_warn"], "degraded": False, "quality_gate_ok": True, "regime": "balanced"},
    )
    s_alert = _runtime_position_risk_scale(
        ext_runtime_scale=1.0,
        ext_runtime_diag={"flags": ["heartbeat_alert"], "degraded": False, "quality_gate_ok": True, "regime": "balanced"},
    )
    assert 0.2 <= s_alert < s_warn < 1.0


def test_runtime_risk_caps_council_divergence_alert_tightens_caps():
    trades, opens = _runtime_risk_caps(
        max_trades_cap=20,
        max_open_positions_cap=8,
        ext_runtime_scale=1.0,
        ext_runtime_diag={"flags": ["council_divergence_alert"], "degraded": False, "quality_gate_ok": True, "regime": "balanced"},
    )
    assert trades <= 12
    assert opens <= 2


def test_runtime_position_risk_scale_council_alert_tighter_than_warn():
    s_warn = _runtime_position_risk_scale(
        ext_runtime_scale=1.0,
        ext_runtime_diag={"flags": ["council_divergence_warn"], "degraded": False, "quality_gate_ok": True, "regime": "balanced"},
    )
    s_alert = _runtime_position_risk_scale(
        ext_runtime_scale=1.0,
        ext_runtime_diag={"flags": ["council_divergence_alert"], "degraded": False, "quality_gate_ok": True, "regime": "balanced"},
    )
    assert 0.2 <= s_alert < s_warn < 1.0


def test_runtime_risk_caps_aion_outcome_alert_tightens_caps():
    trades, opens = _runtime_risk_caps(
        max_trades_cap=20,
        max_open_positions_cap=8,
        ext_runtime_scale=1.0,
        ext_runtime_diag={"flags": ["aion_outcome_alert"], "degraded": False, "quality_gate_ok": True, "regime": "balanced"},
    )
    assert trades <= 12
    assert opens <= 2


def test_runtime_position_risk_scale_aion_outcome_alert_tighter_than_warn():
    s_warn = _runtime_position_risk_scale(
        ext_runtime_scale=1.0,
        ext_runtime_diag={"flags": ["aion_outcome_warn"], "degraded": False, "quality_gate_ok": True, "regime": "balanced"},
    )
    s_alert = _runtime_position_risk_scale(
        ext_runtime_scale=1.0,
        ext_runtime_diag={"flags": ["aion_outcome_alert"], "degraded": False, "quality_gate_ok": True, "regime": "balanced"},
    )
    assert 0.2 <= s_alert < s_warn < 1.0


def test_runtime_risk_caps_aion_outcome_stale_tightens_caps():
    trades, opens = _runtime_risk_caps(
        max_trades_cap=20,
        max_open_positions_cap=8,
        ext_runtime_scale=1.0,
        ext_runtime_diag={"flags": ["aion_outcome_stale"], "degraded": False, "quality_gate_ok": True, "regime": "balanced"},
    )
    assert trades <= 15
    assert opens <= 3


def test_runtime_position_risk_scale_aion_outcome_stale_tightens():
    s = _runtime_position_risk_scale(
        ext_runtime_scale=1.0,
        ext_runtime_diag={"flags": ["aion_outcome_stale"], "degraded": False, "quality_gate_ok": True, "regime": "balanced"},
    )
    assert 0.2 <= s < 1.0


def test_overlay_entry_gate_blocks_critical_flags(monkeypatch):
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_ENABLED", True)
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_BLOCK_CRITICAL", True)
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_BLOCK_CRITICAL_FLAGS", ["fracture_alert", "exec_risk_hard"])
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_BLOCK_ON_QUALITY_FAIL", False)
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_BLOCK_STALE_HOURS", 24.0)

    blocked, reasons = pl._overlay_entry_gate(
        ext_runtime_diag={"flags": ["fracture_alert"], "quality_gate_ok": True, "overlay_stale": False},
        overlay_age_hours=2.0,
    )
    assert blocked is True
    assert "critical_flag:fracture_alert" in reasons


def test_overlay_entry_gate_blocks_stale_when_age_over_threshold(monkeypatch):
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_ENABLED", True)
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_BLOCK_CRITICAL", False)
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_BLOCK_ON_QUALITY_FAIL", False)
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_BLOCK_STALE_HOURS", 24.0)

    blocked, reasons = pl._overlay_entry_gate(
        ext_runtime_diag={"flags": ["overlay_stale"], "quality_gate_ok": True, "overlay_stale": True},
        overlay_age_hours=30.0,
    )
    assert blocked is True
    assert "overlay_stale" in reasons


def test_overlay_entry_gate_does_not_block_stale_under_threshold(monkeypatch):
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_ENABLED", True)
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_BLOCK_CRITICAL", False)
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_BLOCK_ON_QUALITY_FAIL", False)
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_BLOCK_STALE_HOURS", 24.0)

    blocked, reasons = pl._overlay_entry_gate(
        ext_runtime_diag={"flags": ["overlay_stale"], "quality_gate_ok": True, "overlay_stale": True},
        overlay_age_hours=8.0,
    )
    assert blocked is False
    assert reasons == []


def test_overlay_entry_gate_blocks_quality_fail_when_enabled(monkeypatch):
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_ENABLED", True)
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_BLOCK_CRITICAL", False)
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_BLOCK_ON_QUALITY_FAIL", True)
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_BLOCK_STALE_HOURS", 0.0)

    blocked, reasons = pl._overlay_entry_gate(
        ext_runtime_diag={"flags": [], "quality_gate_ok": False, "overlay_stale": False},
        overlay_age_hours=1.0,
    )
    assert blocked is True
    assert "quality_gate_fail" in reasons


def test_execution_quality_governor_warn_scales_caps(monkeypatch):
    now = datetime(2026, 2, 15, 12, 0, 0, tzinfo=timezone.utc)
    events = [
        {"ts": (now - timedelta(minutes=5)).isoformat(), "slippage_bps": 22.0},
        {"ts": (now - timedelta(minutes=7)).isoformat(), "slippage_bps": 21.0},
        {"ts": (now - timedelta(minutes=9)).isoformat(), "slippage_bps": 23.0},
        {"ts": (now - timedelta(minutes=11)).isoformat(), "slippage_bps": 20.0},
        {"ts": (now - timedelta(minutes=13)).isoformat(), "slippage_bps": 24.0},
        {"ts": (now - timedelta(minutes=15)).isoformat(), "slippage_bps": 22.5},
    ]
    mon = SimpleNamespace(state={"execution_events": events, "slippage_points": [e["slippage_bps"] for e in events]})

    monkeypatch.setattr(pl.cfg, "EXEC_GOVERNOR_ENABLED", True)
    monkeypatch.setattr(pl.cfg, "EXEC_GOVERNOR_LOOKBACK_MIN", 25)
    monkeypatch.setattr(pl.cfg, "EXEC_GOVERNOR_MIN_EXECUTIONS", 6)
    monkeypatch.setattr(pl.cfg, "EXEC_GOVERNOR_SLIP_WARN_BPS", 20.0)
    monkeypatch.setattr(pl.cfg, "EXEC_GOVERNOR_SLIP_ALERT_BPS", 30.0)
    monkeypatch.setattr(pl.cfg, "EXEC_GOVERNOR_RATE_WARN_PER_MIN", 0.9)
    monkeypatch.setattr(pl.cfg, "EXEC_GOVERNOR_RATE_ALERT_PER_MIN", 1.8)
    monkeypatch.setattr(pl.cfg, "EXEC_GOVERNOR_BLOCK_ON_ALERT", False)
    monkeypatch.setattr(pl.cfg, "EXEC_GOVERNOR_WARN_TRADES_SCALE", 0.8)
    monkeypatch.setattr(pl.cfg, "EXEC_GOVERNOR_WARN_OPEN_SCALE", 0.8)
    monkeypatch.setattr(pl.cfg, "EXEC_GOVERNOR_WARN_RISK_SCALE", 0.86)
    monkeypatch.setattr(pl.cfg, "EXEC_GOVERNOR_ALERT_TRADES_SCALE", 0.6)
    monkeypatch.setattr(pl.cfg, "EXEC_GOVERNOR_ALERT_OPEN_SCALE", 0.6)
    monkeypatch.setattr(pl.cfg, "EXEC_GOVERNOR_ALERT_RISK_SCALE", 0.72)

    out = pl._execution_quality_governor(
        max_trades_per_day=15,
        max_open_positions=6,
        risk_per_trade=0.02,
        max_position_notional_pct=0.20,
        max_gross_leverage=1.6,
        monitor=mon,
        now_utc=now,
    )
    assert out["state"] == "warn"
    assert out["block_new_entries"] is False
    assert out["max_trades_per_day"] < 15
    assert out["max_open_positions"] < 6
    assert out["risk_per_trade"] < 0.02


def test_execution_quality_governor_alert_can_block_entries(monkeypatch):
    now = datetime(2026, 2, 15, 12, 0, 0, tzinfo=timezone.utc)
    events = [
        {"ts": (now - timedelta(minutes=2)).isoformat(), "slippage_bps": 38.0},
        {"ts": (now - timedelta(minutes=3)).isoformat(), "slippage_bps": 42.0},
        {"ts": (now - timedelta(minutes=4)).isoformat(), "slippage_bps": 36.0},
        {"ts": (now - timedelta(minutes=5)).isoformat(), "slippage_bps": 40.0},
        {"ts": (now - timedelta(minutes=6)).isoformat(), "slippage_bps": 37.0},
        {"ts": (now - timedelta(minutes=7)).isoformat(), "slippage_bps": 39.0},
    ]
    mon = SimpleNamespace(state={"execution_events": events, "slippage_points": [e["slippage_bps"] for e in events]})

    monkeypatch.setattr(pl.cfg, "EXEC_GOVERNOR_ENABLED", True)
    monkeypatch.setattr(pl.cfg, "EXEC_GOVERNOR_LOOKBACK_MIN", 25)
    monkeypatch.setattr(pl.cfg, "EXEC_GOVERNOR_MIN_EXECUTIONS", 6)
    monkeypatch.setattr(pl.cfg, "EXEC_GOVERNOR_SLIP_WARN_BPS", 20.0)
    monkeypatch.setattr(pl.cfg, "EXEC_GOVERNOR_SLIP_ALERT_BPS", 30.0)
    monkeypatch.setattr(pl.cfg, "EXEC_GOVERNOR_RATE_WARN_PER_MIN", 0.9)
    monkeypatch.setattr(pl.cfg, "EXEC_GOVERNOR_RATE_ALERT_PER_MIN", 1.8)
    monkeypatch.setattr(pl.cfg, "EXEC_GOVERNOR_BLOCK_ON_ALERT", True)
    monkeypatch.setattr(pl.cfg, "EXEC_GOVERNOR_WARN_TRADES_SCALE", 0.8)
    monkeypatch.setattr(pl.cfg, "EXEC_GOVERNOR_WARN_OPEN_SCALE", 0.8)
    monkeypatch.setattr(pl.cfg, "EXEC_GOVERNOR_WARN_RISK_SCALE", 0.86)
    monkeypatch.setattr(pl.cfg, "EXEC_GOVERNOR_ALERT_TRADES_SCALE", 0.6)
    monkeypatch.setattr(pl.cfg, "EXEC_GOVERNOR_ALERT_OPEN_SCALE", 0.6)
    monkeypatch.setattr(pl.cfg, "EXEC_GOVERNOR_ALERT_RISK_SCALE", 0.72)

    out = pl._execution_quality_governor(
        max_trades_per_day=15,
        max_open_positions=6,
        risk_per_trade=0.02,
        max_position_notional_pct=0.20,
        max_gross_leverage=1.6,
        monitor=mon,
        now_utc=now,
    )
    assert out["state"] == "alert"
    assert out["block_new_entries"] is True
    assert out["max_trades_per_day"] <= 9
    assert out["max_open_positions"] <= 4


def test_memory_feedback_controls_scale_runtime_and_block(monkeypatch):
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_MEMORY_FEEDBACK_ENABLED", True)
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_MEMORY_FEEDBACK_MIN_SCALE", 0.70)
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_MEMORY_FEEDBACK_MAX_SCALE", 1.12)
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_MEMORY_FEEDBACK_ALERT_THRESHOLD", 0.84)
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_MEMORY_FEEDBACK_BLOCK_ON_ALERT", True)

    out = pl._memory_feedback_controls(
        max_trades_cap_runtime=15,
        max_open_positions_runtime=6,
        risk_per_trade_runtime=0.02,
        max_position_notional_pct_runtime=0.20,
        max_gross_leverage_runtime=1.6,
        memory_feedback={
            "active": True,
            "status": "alert",
            "risk_scale": 0.80,
            "max_trades_scale": 0.76,
            "max_open_scale": 0.82,
            "block_new_entries": False,
            "reasons": ["low_context_resonance"],
        },
    )
    assert out["active"] is True
    assert out["block_new_entries"] is True
    assert out["max_trades_cap_runtime"] < 15
    assert out["max_open_positions_runtime"] < 6
    assert out["risk_per_trade_runtime"] < 0.02


def test_memory_feedback_controls_disabled_noop(monkeypatch):
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_MEMORY_FEEDBACK_ENABLED", False)
    out = pl._memory_feedback_controls(
        max_trades_cap_runtime=12,
        max_open_positions_runtime=5,
        risk_per_trade_runtime=0.02,
        max_position_notional_pct_runtime=0.20,
        max_gross_leverage_runtime=1.6,
        memory_feedback={"active": True, "risk_scale": 0.70, "max_trades_scale": 0.70, "max_open_scale": 0.70},
    )
    assert out["active"] is False
    assert out["max_trades_cap_runtime"] == 12
    assert out["max_open_positions_runtime"] == 5
    assert out["risk_per_trade_runtime"] == 0.02


def test_aion_feedback_controls_block_on_alert_with_enough_closed_trades(monkeypatch):
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_AION_FEEDBACK_ENABLED", True)
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_AION_FEEDBACK_ALERT_THRESHOLD", 0.82)
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_AION_FEEDBACK_BLOCK_ON_ALERT", True)
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_AION_FEEDBACK_MIN_CLOSED_TRADES", 8)

    out = pl._aion_feedback_controls(
        {
            "active": True,
            "status": "alert",
            "source": "shadow_trades",
            "source_preference": "auto",
            "risk_scale": 0.78,
            "closed_trades": 12,
            "hit_rate": 0.39,
            "profit_factor": 0.81,
            "expectancy": -2.2,
            "drawdown_norm": 2.9,
            "last_closed_ts": "2026-02-16T15:35:00Z",
            "reasons": ["negative_expectancy_alert"],
            "path": "/tmp/shadow_trades.csv",
        }
    )
    assert out["active"] is True
    assert out["block_new_entries"] is True
    assert out["status"] == "alert"
    assert out["source"] == "shadow_trades"
    assert out["source_selected"] == "shadow_trades"
    assert out["source_preference"] == "auto"
    assert out["closed_trades"] == 12
    assert out["last_closed_ts"] == "2026-02-16T15:35:00Z"


def test_aion_feedback_controls_no_block_when_disabled(monkeypatch):
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_AION_FEEDBACK_ENABLED", False)
    out = pl._aion_feedback_controls(
        {
            "active": True,
            "status": "alert",
            "risk_scale": 0.70,
            "closed_trades": 20,
            "reasons": ["low_profit_factor_alert"],
        }
    )
    assert out["active"] is False
    assert out["block_new_entries"] is False


def test_aion_feedback_controls_stale_feedback_neutralizes_when_ignore_enabled(monkeypatch):
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_AION_FEEDBACK_ENABLED", True)
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_AION_FEEDBACK_ALERT_THRESHOLD", 0.82)
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_AION_FEEDBACK_BLOCK_ON_ALERT", True)
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_AION_FEEDBACK_MIN_CLOSED_TRADES", 8)
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_AION_FEEDBACK_MAX_AGE_HOURS", 24.0)
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_AION_FEEDBACK_IGNORE_STALE", True)

    out = pl._aion_feedback_controls(
        {
            "active": True,
            "status": "alert",
            "source": "overlay",
            "source_selected": "shadow_trades",
            "source_preference": "shadow",
            "risk_scale": 0.70,
            "closed_trades": 20,
            "age_hours": 72.0,
            "reasons": ["low_profit_factor_alert"],
        }
    )
    assert out["active"] is True
    assert out["stale"] is True
    assert out["status"] == "stale"
    assert out["block_new_entries"] is False
    assert float(out["risk_scale"]) == 1.0
    assert out["source"] == "overlay"
    assert out["source_selected"] == "shadow_trades"
    assert out["source_preference"] == "shadow"
