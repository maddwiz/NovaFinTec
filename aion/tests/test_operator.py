import aion.exec.operator as op
import json
import os
from datetime import datetime, timezone


def test_operator_defaults_to_status(monkeypatch):
    seen = {"status": 0}

    def fake_status():
        seen["status"] += 1
        return 0

    monkeypatch.setattr(op, "_status", fake_status)
    rc = op.main([])
    assert rc == 0
    assert seen["status"] == 1


def test_operator_start_dispatch(monkeypatch):
    seen = {"tasks": None}

    def fake_start(tasks):
        seen["tasks"] = tasks
        return 0

    monkeypatch.setattr(op, "_start", fake_start)
    rc = op.main(["start", "--task", "trade"])
    assert rc == 0
    assert seen["tasks"] == ["trade"]


def test_operator_status_includes_runtime_controls_and_overlay(tmp_path, monkeypatch, capsys):
    log_dir = tmp_path / "logs"
    state_dir = tmp_path / "state"
    log_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "doctor_report.json").write_text('{"ok": true, "ib": {"configured_port": 4002}}', encoding="utf-8")
    (state_dir / "runtime_controls.json").write_text(
        (
            '{"max_trades_cap_runtime": 9, "overlay_block_new_entries": true, '
            '"overlay_block_reasons": ["critical_flag:fracture_alert"], '
            '"aion_feedback_active": true, "aion_feedback_status": "warn", '
            '"aion_feedback_source": "shadow_trades", "aion_feedback_source_selected": "shadow_trades", '
            '"aion_feedback_source_preference": "auto", '
            '"aion_feedback_risk_scale": 0.88, "aion_feedback_closed_trades": 12, '
            '"aion_feedback_age_hours": 80.0, "aion_feedback_max_age_hours": 72.0, '
            '"aion_feedback_stale": false, '
            '"memory_feedback_active": true, "memory_feedback_status": "ok", '
            '"memory_feedback_risk_scale": 0.93, "memory_feedback_trades_scale": 0.8, '
            '"memory_feedback_open_scale": 0.75, '
            '"memory_feedback_turnover_pressure": 0.51, "memory_feedback_turnover_dampener": 0.84, '
            '"memory_feedback_reasons": ["turnover_guard:warn"], '
            '"memory_feedback_block_new_entries": false, '
            '"memory_replay_enabled": true, "memory_replay_last_ok": true, '
            '"memory_replay_remaining_files": 6, "memory_replay_queued_files": 6, '
            '"memory_outbox_warn_files": 5, "memory_outbox_alert_files": 20}'
        ),
        encoding="utf-8",
    )
    ext = tmp_path / "overlay.json"
    ext.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "runtime_context": {"runtime_multiplier": 0.82, "risk_flags": ["exec_risk_tight"]},
            }
        ),
        encoding="utf-8",
    )
    old_ts = 946684800
    os.utime(ext, (old_ts, old_ts))
    monkeypatch.setattr(op.cfg, "LOG_DIR", log_dir)
    monkeypatch.setattr(op.cfg, "STATE_DIR", state_dir)
    monkeypatch.setattr(op.cfg, "EXT_SIGNAL_FILE", ext)
    monkeypatch.setattr(op.cfg, "OPS_GUARD_STATUS_FILE", tmp_path / "ops_guard_status.json")
    monkeypatch.setattr(op, "status_snapshot", lambda _tasks=None: {"trade": {"running": True, "pids": [123]}})

    rc = op.main(["status"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["runtime_controls"]["max_trades_cap_runtime"] == 9
    assert payload["runtime_controls"]["overlay_block_new_entries"] is True
    assert "critical_flag:fracture_alert" in payload["runtime_controls"]["overlay_block_reasons"]
    assert payload["aion_feedback_runtime"]["present"] is True
    assert payload["aion_feedback_runtime"]["source"] == "runtime_controls"
    assert payload["aion_feedback_runtime"]["state"] == "stale"
    assert payload["aion_feedback_runtime"]["stale"] is True
    assert payload["aion_feedback_runtime"]["feedback_source"] == "shadow_trades"
    assert payload["aion_feedback_runtime"]["feedback_source_selected"] == "shadow_trades"
    assert payload["aion_feedback_runtime"]["feedback_source_preference"] == "auto"
    assert payload["memory_feedback_runtime"]["present"] is True
    assert payload["memory_feedback_runtime"]["source"] == "runtime_controls"
    assert payload["memory_feedback_runtime"]["state"] == "warn"
    assert payload["memory_feedback_runtime"]["severity"] == 2
    assert payload["memory_feedback_runtime"]["turnover_pressure"] == 0.51
    assert payload["memory_feedback_runtime"]["turnover_dampener"] == 0.84
    assert payload["memory_feedback_runtime"]["max_trades_scale"] == 0.8
    assert payload["memory_feedback_runtime"]["max_open_scale"] == 0.75
    assert "turnover_guard:warn" in payload["memory_feedback_runtime"]["reasons"]
    assert payload["memory_outbox_runtime"]["present"] is True
    assert payload["memory_outbox_runtime"]["source"] == "runtime_controls"
    assert payload["memory_outbox_runtime"]["state"] == "warn"
    assert payload["memory_outbox_runtime"]["severity"] == 2
    assert payload["memory_outbox_runtime"]["remaining_files"] == 6
    assert payload["runtime_decision"]["entry_blocked"] is True
    assert any("external_overlay" in x for x in payload["runtime_decision"]["entry_block_reasons"])
    assert isinstance(payload["runtime_remediation"], list)
    assert len(payload["runtime_remediation"]) >= 1
    assert payload["external_runtime_context"]["runtime_multiplier"] == 0.82
    assert payload["external_overlay_runtime"]["exists"] is True
    assert payload["external_overlay_runtime"]["stale"] is False
    assert payload["external_overlay_runtime"]["age_source"] == "payload"
    assert payload["external_overlay_runtime"]["generated_at_utc"] is not None
    assert payload["external_overlay_runtime"]["runtime_context_present"] is True
    assert "exec_risk_tight" in payload["external_overlay_runtime"]["risk_flags"]
    assert payload["runtime_controls_age_sec"] is not None
    assert payload["runtime_controls_stale_threshold_sec"] >= 60
    assert payload["runtime_controls_stale"] is False
