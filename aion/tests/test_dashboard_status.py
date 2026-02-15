import json
from pathlib import Path

import aion.exec.dashboard as dash


def _write(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_status_payload_includes_external_overlay_fields(tmp_path: Path, monkeypatch):
    log_dir = tmp_path / "logs"
    state_dir = tmp_path / "state"
    ops_status = tmp_path / "ops_guard_status.json"
    monkeypatch.setattr(dash.cfg, "LOG_DIR", log_dir)
    monkeypatch.setattr(dash.cfg, "STATE_DIR", state_dir)
    monkeypatch.setattr(dash.cfg, "OPS_GUARD_STATUS_FILE", ops_status)
    monkeypatch.setattr(dash.cfg, "OPS_GUARD_TARGETS", ["trade", "dashboard"])

    _write(
        log_dir / "doctor_report.json",
        {
            "ok": True,
            "checks": [
                {
                    "name": "external_overlay",
                    "ok": False,
                    "msg": "External overlay issues: degraded_safe_mode=true",
                    "details": {"degraded_safe_mode": True, "signals": 0, "risk_flags": ["fracture_alert"]},
                }
            ],
        },
    )
    _write(log_dir / "runtime_monitor.json", {"alerts": [], "system_events": []})
    _write(log_dir / "performance_report.json", {"trade_metrics": {"closed_trades": 5}, "equity_metrics": {}})
    _write(state_dir / "strategy_profile.json", {"trading_enabled": True, "adaptive_stats": {}})
    _write(
        ops_status,
        {
            "running": {
                "trade": {"running": True, "pids": [123]},
                "dashboard": {"running": True, "pids": [456]},
            }
        },
    )
    _write(
        state_dir / "runtime_controls.json",
        {
            "max_trades_cap_runtime": 9,
            "max_open_positions_runtime": 3,
            "external_position_risk_scale": 0.82,
            "policy_block_new_entries": False,
        },
    )
    (state_dir / "watchlist.txt").write_text("AAPL\nMSFT\n", encoding="utf-8")

    s = dash._status_payload()
    assert s["external_overlay_ok"] is False
    assert "degraded_safe_mode=true" in str(s["external_overlay_msg"])
    assert s["external_overlay"]["signals"] == 0
    assert "fracture_alert" in s["external_overlay_risk_flags"]
    assert s["external_fracture_state"] == "alert"
    assert s["ops_guard_ok"] is True
    assert s["runtime_controls"]["max_trades_cap_runtime"] == 9
    assert s["runtime_controls"]["external_position_risk_scale"] == 0.82
    assert s["watchlist_count"] == 2
