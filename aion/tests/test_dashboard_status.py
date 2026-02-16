import json
import os
from datetime import datetime, timezone
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
    overlay_file = tmp_path / "q_signal_overlay.json"
    monkeypatch.setattr(dash.cfg, "EXT_SIGNAL_FILE", overlay_file)
    monkeypatch.setattr(dash.cfg, "EXT_SIGNAL_MAX_AGE_HOURS", 12.0)

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
            "overlay_block_new_entries": True,
            "overlay_block_reasons": ["critical_flag:fracture_alert"],
            "policy_block_new_entries": False,
        },
    )
    _write(
        overlay_file,
        {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "runtime_context": {"runtime_multiplier": 0.91, "risk_flags": ["drift_warn"]},
            "signals": {"AAPL": {"bias": 0.2, "confidence": 0.8}},
        },
    )
    old_ts = 946684800
    os.utime(overlay_file, (old_ts, old_ts))
    (state_dir / "watchlist.txt").write_text("AAPL\nMSFT\n", encoding="utf-8")

    s = dash._status_payload()
    assert s["external_overlay_ok"] is False
    assert "degraded_safe_mode=true" in str(s["external_overlay_msg"])
    assert s["external_overlay"]["signals"] == 0
    assert s["external_overlay_runtime"]["exists"] is True
    assert s["external_overlay_runtime"]["stale"] is False
    assert s["external_overlay_runtime"]["age_source"] == "payload"
    assert s["external_overlay_runtime"]["generated_at_utc"] is not None
    assert s["external_overlay_runtime"]["runtime_context_present"] is True
    assert "drift_warn" in s["external_overlay_runtime"]["risk_flags"]
    assert "fracture_alert" in s["external_overlay_risk_flags"]
    assert "drift_warn" in s["external_overlay_risk_flags"]
    assert s["external_fracture_state"] == "alert"
    assert s["ops_guard_ok"] is True
    assert s["runtime_controls"]["max_trades_cap_runtime"] == 9
    assert s["runtime_controls"]["external_position_risk_scale"] == 0.82
    assert s["runtime_controls"]["overlay_block_new_entries"] is True
    assert "critical_flag:fracture_alert" in s["runtime_controls"]["overlay_block_reasons"]
    assert s["runtime_decision"]["entry_blocked"] is True
    assert any("external_overlay" in x for x in s["runtime_decision"]["entry_block_reasons"])
    assert s["runtime_decision"]["throttle_state"] in {"warn", "alert"}
    assert isinstance(s["runtime_remediation"], list)
    assert len(s["runtime_remediation"]) >= 1
    assert s["runtime_controls_age_sec"] is not None
    assert s["runtime_controls_stale_threshold_sec"] >= 60
    assert s["runtime_controls_stale"] is False
    assert s["watchlist_count"] == 2
