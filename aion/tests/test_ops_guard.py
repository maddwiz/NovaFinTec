import json
import os
from pathlib import Path

import aion.exec.ops_guard as og


def test_find_task_pids_from_ps_text():
    text = """
      123 /usr/bin/python -m aion.exec.paper_loop
      124 /usr/bin/python -m aion.exec.dashboard
      125 /usr/bin/python -m aion.exec.operator
    """
    assert og.find_task_pids("trade", ps_text=text) == [123]
    assert og.find_task_pids("dashboard", ps_text=text) == [124]


def test_can_restart_respects_limits():
    hist = [100.0, 200.0]
    ok, reason = og._can_restart(now_ts=250.0, history=hist, cooldown_sec=120.0, max_restarts_per_hour=5)
    assert ok is False
    assert reason == "cooldown"

    hist2 = [10.0, 20.0, 30.0]
    ok2, reason2 = og._can_restart(now_ts=3705.0, history=hist2, cooldown_sec=0.0, max_restarts_per_hour=2)
    assert ok2 is True
    assert reason2 == "ok"


def test_guard_cycle_restarts_missing_task(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(og.cfg, "OPS_GUARD_TARGETS", ["trade"])
    monkeypatch.setattr(og.cfg, "OPS_GUARD_STATUS_FILE", tmp_path / "ops_guard_status.json")
    monkeypatch.setattr(og.cfg, "OPS_GUARD_RESTART_COOLDOWN_SEC", 1)
    monkeypatch.setattr(og.cfg, "OPS_GUARD_MAX_RESTARTS_PER_HOUR", 3)
    monkeypatch.setattr(og.cfg, "OPS_GUARD_INCIDENT_LOG", tmp_path / "ops_incidents.log")

    calls = {"n": 0}

    def fake_status(_targets=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"trade": {"module": "aion.exec.paper_loop", "running": False, "pids": []}}
        return {"trade": {"module": "aion.exec.paper_loop", "running": True, "pids": [999]}}

    monkeypatch.setattr(og, "status_snapshot", fake_status)
    monkeypatch.setattr(og, "start_task", lambda task: True)

    payload = og.guard_cycle({"restart_history": {}})
    assert payload["restarts"]["trade"]["attempt"] == "started"
    assert payload["running"]["trade"]["running"] is True

    raw = json.loads((tmp_path / "ops_guard_status.json").read_text(encoding="utf-8"))
    assert raw["running"]["trade"]["running"] is True


def test_main_once_mode(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(og.cfg, "OPS_GUARD_TARGETS", ["trade"])
    monkeypatch.setattr(og.cfg, "OPS_GUARD_STATUS_FILE", tmp_path / "ops_guard_status.json")
    monkeypatch.setattr(og.cfg, "OPS_GUARD_INCIDENT_LOG", tmp_path / "ops_incidents.log")
    monkeypatch.setattr(og, "status_snapshot", lambda _targets=None: {"trade": {"module": "aion.exec.paper_loop", "running": True, "pids": [1]}})
    rc = og.main(["--once"])
    assert rc == 0
    assert (tmp_path / "ops_guard_status.json").exists()


def test_start_task_returns_false_when_process_not_detected(tmp_path: Path, monkeypatch):
    run_script = tmp_path / "run_aion.sh"
    run_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    run_script.chmod(0o755)
    monkeypatch.setattr(og, "_repo_root", lambda: tmp_path)
    class _ProcExit:
        returncode = 1

        def poll(self):
            return 1

    monkeypatch.setattr(og.subprocess, "Popen", lambda *args, **kwargs: _ProcExit())
    monkeypatch.setattr(og, "find_task_pids", lambda _task: [])
    monkeypatch.setattr(og.time, "sleep", lambda _x: None)
    assert og.start_task("trade", root=tmp_path, log_dir=tmp_path) is False


def test_start_task_returns_true_when_launcher_alive(tmp_path: Path, monkeypatch):
    run_script = tmp_path / "run_aion.sh"
    run_script.write_text("#!/usr/bin/env bash\nsleep 1\n", encoding="utf-8")
    run_script.chmod(0o755)
    monkeypatch.setattr(og, "_repo_root", lambda: tmp_path)

    class _ProcAlive:
        returncode = None

        def poll(self):
            return None

    monkeypatch.setattr(og.subprocess, "Popen", lambda *args, **kwargs: _ProcAlive())
    monkeypatch.setattr(og, "find_task_pids", lambda _task: [])
    monkeypatch.setattr(og.time, "sleep", lambda _x: None)
    assert og.start_task("trade", root=tmp_path, log_dir=tmp_path) is True


def test_start_task_returns_true_when_single_instance_already_running(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(og, "find_task_pids", lambda _task: [123])
    monkeypatch.setattr(og.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("should not launch")))
    assert og.start_task("trade", root=tmp_path, log_dir=tmp_path) is True


def test_trade_runtime_controls_stale_detection(tmp_path: Path, monkeypatch):
    st = tmp_path / "state"
    st.mkdir(parents=True, exist_ok=True)
    p = st / "runtime_controls.json"
    p.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(og.cfg, "STATE_DIR", st)
    monkeypatch.setattr(og.cfg, "OPS_GUARD_TRADE_STALE_SEC", 120)
    now = 1000.0
    os.utime(p, (now - 300, now - 300))
    assert og._trade_runtime_controls_stale(now_ts=now) is True


def test_trade_runtime_controls_stale_threshold_scales_with_watchlist(tmp_path: Path, monkeypatch):
    st = tmp_path / "state"
    st.mkdir(parents=True, exist_ok=True)
    p = st / "runtime_controls.json"
    p.write_text('{"loop_seconds": 30, "watchlist_size": 180}', encoding="utf-8")
    monkeypatch.setattr(og.cfg, "STATE_DIR", st)
    monkeypatch.setattr(og.cfg, "LOOP_SECONDS", 30)
    monkeypatch.setattr(og.cfg, "OPS_GUARD_TRADE_STALE_SEC", 120)
    now = 1000.0
    os.utime(p, (now - 300, now - 300))
    assert og._trade_runtime_controls_stale_threshold_sec(now_ts=now) > 300.0
    assert og._trade_runtime_controls_stale(now_ts=now) is False


def test_guard_cycle_restarts_trade_when_runtime_controls_stale(tmp_path: Path, monkeypatch):
    st = tmp_path / "state"
    st.mkdir(parents=True, exist_ok=True)
    p = st / "runtime_controls.json"
    p.write_text("{}", encoding="utf-8")
    now = 1000.0
    os.utime(p, (now - 500, now - 500))

    monkeypatch.setattr(og.cfg, "STATE_DIR", st)
    monkeypatch.setattr(og.cfg, "OPS_GUARD_TARGETS", ["trade"])
    monkeypatch.setattr(og.cfg, "OPS_GUARD_STATUS_FILE", tmp_path / "ops_guard_status.json")
    monkeypatch.setattr(og.cfg, "OPS_GUARD_INCIDENT_LOG", tmp_path / "ops_incidents.log")
    monkeypatch.setattr(og.cfg, "OPS_GUARD_RESTART_COOLDOWN_SEC", 1)
    monkeypatch.setattr(og.cfg, "OPS_GUARD_MAX_RESTARTS_PER_HOUR", 6)
    monkeypatch.setattr(og.cfg, "OPS_GUARD_TRADE_STALE_SEC", 120)
    monkeypatch.setattr(og.time, "time", lambda: now)

    calls = {"n": 0}

    def fake_status(_targets=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"trade": {"module": "aion.exec.paper_loop", "running": True, "pids": [111]}}
        return {"trade": {"module": "aion.exec.paper_loop", "running": True, "pids": [222]}}

    stopped = {"n": 0}
    monkeypatch.setattr(og, "status_snapshot", fake_status)
    monkeypatch.setattr(og, "start_task", lambda _task: True)
    monkeypatch.setattr(og, "stop_task", lambda _task: stopped.__setitem__("n", stopped["n"] + 1) or 1)

    payload = og.guard_cycle({"restart_history": {}})
    assert stopped["n"] == 1
    assert payload["restarts"]["trade"]["attempt"] == "started"
    assert payload["config"]["trade_stale_threshold_sec"] >= 60


def test_guard_cycle_restarts_trade_when_duplicate_instances(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(og.cfg, "OPS_GUARD_TARGETS", ["trade"])
    monkeypatch.setattr(og.cfg, "OPS_GUARD_STATUS_FILE", tmp_path / "ops_guard_status.json")
    monkeypatch.setattr(og.cfg, "OPS_GUARD_INCIDENT_LOG", tmp_path / "ops_incidents.log")
    monkeypatch.setattr(og.cfg, "OPS_GUARD_RESTART_COOLDOWN_SEC", 1)
    monkeypatch.setattr(og.cfg, "OPS_GUARD_MAX_RESTARTS_PER_HOUR", 6)
    monkeypatch.setattr(og.time, "time", lambda: 1000.0)
    monkeypatch.setattr(og, "_trade_runtime_controls_stale", lambda now_ts=None: False)

    calls = {"n": 0}

    def fake_status(_targets=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"trade": {"module": "aion.exec.paper_loop", "running": True, "pids": [111, 112]}}
        return {"trade": {"module": "aion.exec.paper_loop", "running": True, "pids": [222]}}

    stopped = {"n": 0}
    monkeypatch.setattr(og, "status_snapshot", fake_status)
    monkeypatch.setattr(og, "start_task", lambda _task: True)
    monkeypatch.setattr(og, "stop_task", lambda _task: stopped.__setitem__("n", stopped["n"] + 1) or 2)

    payload = og.guard_cycle({"restart_history": {}})
    assert stopped["n"] == 1
    assert payload["restarts"]["trade"]["attempt"] == "started"


def test_guard_cycle_forces_duplicate_cleanup_during_cooldown(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(og.cfg, "OPS_GUARD_TARGETS", ["trade"])
    monkeypatch.setattr(og.cfg, "OPS_GUARD_STATUS_FILE", tmp_path / "ops_guard_status.json")
    monkeypatch.setattr(og.cfg, "OPS_GUARD_INCIDENT_LOG", tmp_path / "ops_incidents.log")
    monkeypatch.setattr(og.cfg, "OPS_GUARD_RESTART_COOLDOWN_SEC", 120)
    monkeypatch.setattr(og.cfg, "OPS_GUARD_MAX_RESTARTS_PER_HOUR", 6)
    monkeypatch.setattr(og.time, "time", lambda: 1000.0)
    monkeypatch.setattr(og, "_trade_runtime_controls_stale", lambda now_ts=None: False)

    calls = {"n": 0}

    def fake_status(_targets=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"trade": {"module": "aion.exec.paper_loop", "running": True, "pids": [111, 112]}}
        return {"trade": {"module": "aion.exec.paper_loop", "running": True, "pids": [333]}}

    stopped = {"n": 0}
    monkeypatch.setattr(og, "status_snapshot", fake_status)
    monkeypatch.setattr(og, "start_task", lambda _task: True)
    monkeypatch.setattr(og, "stop_task", lambda _task: stopped.__setitem__("n", stopped["n"] + 1) or 2)

    # Last restart was very recent: should normally fail cooldown.
    payload = og.guard_cycle({"restart_history": {"trade": [950.0]}})
    assert stopped["n"] == 1
    assert payload["restarts"]["trade"]["attempt"] == "started"
    assert payload["restarts"]["trade"]["skip"] is None
