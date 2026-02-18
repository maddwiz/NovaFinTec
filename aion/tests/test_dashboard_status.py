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
    monkeypatch.setattr(dash.cfg, "TELEMETRY_SUMMARY_FILE", state_dir / "telemetry_summary.json")
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
        state_dir / "telemetry_summary.json",
        {
            "closed_trade_events": 5,
            "rolling_hit_rate": 0.6,
            "top_win_signal_category": "session_structure",
            "top_loss_signal_category": "multi_timeframe",
        },
    )
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
            "aion_feedback_active": True,
            "aion_feedback_status": "warn",
            "aion_feedback_source": "shadow_trades",
            "aion_feedback_source_selected": "shadow_trades",
            "aion_feedback_source_preference": "auto",
            "aion_feedback_risk_scale": 0.88,
            "aion_feedback_closed_trades": 12,
            "aion_feedback_age_hours": 96.0,
            "aion_feedback_max_age_hours": 72.0,
            "aion_feedback_stale": False,
            "memory_feedback_active": True,
            "memory_feedback_status": "ok",
            "memory_feedback_risk_scale": 0.93,
            "memory_feedback_trades_scale": 0.79,
            "memory_feedback_open_scale": 0.76,
            "memory_feedback_turnover_pressure": 0.53,
            "memory_feedback_turnover_dampener": 0.83,
            "memory_feedback_reasons": ["turnover_guard:warn"],
            "memory_feedback_block_new_entries": False,
            "memory_replay_enabled": True,
            "memory_replay_last_ok": True,
            "memory_replay_remaining_files": 6,
            "memory_replay_queued_files": 6,
            "memory_outbox_warn_files": 5,
            "memory_outbox_alert_files": 20,
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
    (log_dir / "signals.csv").write_text(
        "\n".join(
            [
                "timestamp,symbol,regime,long_conf,short_conf,decision,meta_prob,mtf_score,intraday_score,intraday_gate,mtf_gate,meta_gate,pattern_hits,indicator_hits,reasons",
                "2026-01-01 09:31:00,AAPL,trending,0.82,0.11,BUY,0.73,0.64,0.88,pass,pass,pass,3,5,Intraday align 0.88",
                "2026-01-01 09:32:00,MSFT,trending,0.76,0.22,HOLD,0.69,0.41,0.32,block,skip,skip,2,4,Intraday blocked (0.32)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (log_dir / "shadow_equity.csv").write_text(
        "\n".join(
            [
                "timestamp,equity,cash,open_pnl,closed_pnl",
                "2026-01-01 09:30:00,5000.00,5000.00,0.00,0.00",
                "2026-01-01 10:00:00,5005.50,5005.50,0.00,5.50",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

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
    assert s["aion_feedback_runtime"]["present"] is True
    assert s["aion_feedback_runtime"]["source"] == "runtime_controls"
    assert s["aion_feedback_runtime"]["state"] == "stale"
    assert s["aion_feedback_runtime"]["stale"] is True
    assert s["aion_feedback_runtime"]["feedback_source"] == "shadow_trades"
    assert s["aion_feedback_runtime"]["feedback_source_selected"] == "shadow_trades"
    assert s["aion_feedback_runtime"]["feedback_source_preference"] == "auto"
    assert s["memory_feedback_runtime"]["present"] is True
    assert s["memory_feedback_runtime"]["source"] == "runtime_controls"
    assert s["memory_feedback_runtime"]["state"] == "warn"
    assert s["memory_feedback_runtime"]["severity"] == 2
    assert s["memory_feedback_runtime"]["turnover_pressure"] == 0.53
    assert s["memory_feedback_runtime"]["turnover_dampener"] == 0.83
    assert s["memory_feedback_runtime"]["max_trades_scale"] == 0.79
    assert s["memory_feedback_runtime"]["max_open_scale"] == 0.76
    assert "turnover_guard:warn" in s["memory_feedback_runtime"]["reasons"]
    assert s["memory_outbox_runtime"]["present"] is True
    assert s["memory_outbox_runtime"]["source"] == "runtime_controls"
    assert s["memory_outbox_runtime"]["state"] == "warn"
    assert s["memory_outbox_runtime"]["severity"] == 2
    assert s["memory_outbox_runtime"]["remaining_files"] == 6
    assert s["runtime_decision"]["entry_blocked"] is True
    assert any("external_overlay" in x for x in s["runtime_decision"]["entry_block_reasons"])
    assert s["runtime_decision"]["throttle_state"] in {"warn", "alert"}
    assert isinstance(s["runtime_remediation"], list)
    assert len(s["runtime_remediation"]) >= 1
    assert s["runtime_controls_age_sec"] is not None
    assert s["runtime_controls_stale_threshold_sec"] >= 60
    assert s["runtime_controls_stale"] is False
    assert s["watchlist_count"] == 2
    assert s["signal_gate_summary"]["rows"] == 2
    assert s["signal_gate_summary"]["considered"] == 2
    assert s["signal_gate_summary"]["blocked_intraday"] == 1
    assert s["signal_gate_summary"]["blocked_total"] == 1
    assert s["signal_gate_summary"]["passed"] == 1
    assert s["signal_gate_summary"]["avg_intraday_score"] is not None
    assert s["pnl_summary"]["present"] is True
    assert s["pnl_summary"]["overall_pnl"] == 5.5
    assert s["pnl_summary"]["overall_return_pct"] is not None
    assert s["telemetry_summary"]["closed_trade_events"] == 5
    assert s["telemetry_summary"]["top_win_signal_category"] == "session_structure"
    assert s["telemetry_summary"]["top_loss_signal_category"] == "multi_timeframe"


def test_status_payload_falls_back_to_live_overlay_when_doctor_is_stale(tmp_path: Path, monkeypatch):
    log_dir = tmp_path / "logs"
    state_dir = tmp_path / "state"
    ops_status = tmp_path / "ops_guard_status.json"
    monkeypatch.setattr(dash.cfg, "LOG_DIR", log_dir)
    monkeypatch.setattr(dash.cfg, "STATE_DIR", state_dir)
    monkeypatch.setattr(dash.cfg, "OPS_GUARD_STATUS_FILE", ops_status)
    monkeypatch.setattr(dash.cfg, "OPS_GUARD_TARGETS", ["trade", "dashboard"])
    monkeypatch.setattr(dash.cfg, "TELEMETRY_SUMMARY_FILE", state_dir / "telemetry_summary.json")
    overlay_file = tmp_path / "q_signal_overlay.json"
    monkeypatch.setattr(dash.cfg, "EXT_SIGNAL_FILE", overlay_file)
    monkeypatch.setattr(dash.cfg, "EXT_SIGNAL_MAX_AGE_HOURS", 12.0)

    doctor_path = log_dir / "doctor_report.json"
    _write(
        doctor_path,
        {
            "ok": True,
            "checks": [
                {
                    "name": "external_overlay",
                    "ok": False,
                    "msg": "External overlay issues: quality_gate_not_ok",
                    "details": {"quality_gate_ok": False, "signals": 0, "risk_flags": ["fracture_alert"]},
                }
            ],
        },
    )
    old_ts = 946684800
    os.utime(doctor_path, (old_ts, old_ts))

    _write(log_dir / "runtime_monitor.json", {"alerts": [], "system_events": []})
    _write(log_dir / "performance_report.json", {"trade_metrics": {"closed_trades": 1}, "equity_metrics": {}})
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
    _write(state_dir / "runtime_controls.json", {})
    _write(
        overlay_file,
        {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "signals": {"AAPL": {"bias": 0.2, "confidence": 0.8}},
            "quality_gate": {"ok": True},
            "runtime_context": {"runtime_multiplier": 0.93, "risk_flags": []},
            "degraded_safe_mode": False,
        },
    )

    s = dash._status_payload()
    assert s["external_overlay_source"] == "live_check"
    assert s["external_overlay_ok"] is True
    assert "healthy" in str(s["external_overlay_msg"]).lower()
    assert s["external_overlay"]["quality_gate_ok"] is True


def test_signal_gate_summary_supports_legacy_reason_fields():
    rows = [
        {
            "decision": "HOLD",
            "reasons": "Intraday blocked (0.35) | MTF blocked (0.42)",
            "intraday_score": "",
        },
        {
            "decision": "BUY",
            "reasons": "Intraday align 0.81",
        },
    ]
    out = dash._signal_gate_summary(rows)
    assert out["rows"] == 2
    assert out["considered"] == 2
    assert out["blocked_intraday"] == 1
    assert out["blocked_mtf"] == 1
    assert out["blocked_total"] == 1
    assert out["passed"] == 1
    assert out["avg_intraday_score"] is not None


def test_tail_jsonl_returns_last_rows(tmp_path: Path):
    p = tmp_path / "trade_decisions.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps({"timestamp": "2026-01-01T10:00:00Z", "symbol": "AAPL", "decision": "ENTRY_LONG"}),
                "not-json",
                json.dumps({"timestamp": "2026-01-01T10:05:00Z", "symbol": "MSFT", "decision": "EXIT_TRAILING_STOP"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = dash._tail_jsonl(p, limit=1)
    assert len(rows) == 1
    assert rows[0]["symbol"] == "MSFT"
