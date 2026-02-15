import aion.exec.operator as op
import json


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
    (state_dir / "runtime_controls.json").write_text('{"max_trades_cap_runtime": 9}', encoding="utf-8")
    ext = tmp_path / "overlay.json"
    ext.write_text('{"runtime_context": {"runtime_multiplier": 0.82, "risk_flags": ["exec_risk_tight"]}}', encoding="utf-8")
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
    assert payload["external_runtime_context"]["runtime_multiplier"] == 0.82
    assert payload["runtime_controls_age_sec"] is not None
    assert payload["runtime_controls_stale_threshold_sec"] >= 60
    assert payload["runtime_controls_stale"] is False
