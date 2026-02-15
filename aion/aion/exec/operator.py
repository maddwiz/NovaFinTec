from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import config as cfg
from .ops_guard import find_task_pids, start_task, status_snapshot, stop_task, task_module
from .runtime_health import runtime_controls_stale_info


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _selected_tasks(raw_task: str) -> list[str]:
    if raw_task == "all":
        return ["trade", "dashboard", "ops-guard"]
    return [raw_task]


def _print(payload: dict):
    print(json.dumps(payload, indent=2))


def _status() -> int:
    tasks = ["trade", "dashboard", "ops-guard"]
    snap = status_snapshot(tasks)
    ops_status = _read_json(cfg.OPS_GUARD_STATUS_FILE, {})
    doctor = _read_json(cfg.LOG_DIR / "doctor_report.json", {})
    rc_info = runtime_controls_stale_info(
        cfg.STATE_DIR / "runtime_controls.json",
        default_loop_seconds=int(cfg.LOOP_SECONDS),
        base_stale_seconds=int(cfg.OPS_GUARD_TRADE_STALE_SEC),
    )
    runtime_controls = rc_info.get("payload", {})
    runtime_controls_age_sec = rc_info.get("age_sec")
    runtime_controls_stale = bool(rc_info.get("stale", False))
    runtime_controls_stale_threshold_sec = float(rc_info.get("threshold_sec", max(60, int(cfg.LOOP_SECONDS * 6))))
    ext_overlay = _read_json(cfg.EXT_SIGNAL_FILE, {})
    ext_ctx = ext_overlay.get("runtime_context", {}) if isinstance(ext_overlay, dict) else {}
    if not isinstance(ext_ctx, dict):
        ext_ctx = {}
    out = {
        "tasks": snap,
        "ops_guard_status": ops_status,
        "runtime_controls": runtime_controls if isinstance(runtime_controls, dict) else {},
        "runtime_controls_age_sec": runtime_controls_age_sec,
        "runtime_controls_stale_threshold_sec": runtime_controls_stale_threshold_sec,
        "runtime_controls_stale": runtime_controls_stale,
        "external_runtime_context": ext_ctx,
        "doctor_ok": bool(doctor.get("ok", False)) if isinstance(doctor, dict) else False,
        "ib": doctor.get("ib", {}) if isinstance(doctor, dict) else {},
    }
    _print(out)
    return 0


def _start(tasks: list[str]) -> int:
    ok_all = True
    out = {"action": "start", "results": []}
    for task in tasks:
        if not task_module(task):
            out["results"].append({"task": task, "ok": False, "msg": "unknown task"})
            ok_all = False
            continue
        running = bool(find_task_pids(task))
        if running:
            out["results"].append({"task": task, "ok": True, "msg": "already running"})
            continue
        ok = start_task(task)
        out["results"].append({"task": task, "ok": bool(ok), "msg": "started" if ok else "start failed"})
        if not ok:
            ok_all = False
    _print(out)
    return 0 if ok_all else 1


def _stop(tasks: list[str]) -> int:
    out = {"action": "stop", "results": []}
    for task in tasks:
        if not task_module(task):
            out["results"].append({"task": task, "ok": False, "stopped": 0, "msg": "unknown task"})
            continue
        stopped = stop_task(task)
        out["results"].append({"task": task, "ok": True, "stopped": int(stopped)})
    _print(out)
    return 0


def _restart(tasks: list[str]) -> int:
    ok_all = True
    out = {"action": "restart", "results": []}
    for task in tasks:
        if not task_module(task):
            out["results"].append({"task": task, "ok": False, "msg": "unknown task"})
            ok_all = False
            continue
        stopped = stop_task(task)
        ok = start_task(task)
        out["results"].append(
            {
                "task": task,
                "ok": bool(ok),
                "stopped": int(stopped),
                "msg": "restarted" if ok else "restart failed",
            }
        )
        if not ok:
            ok_all = False
    _print(out)
    return 0 if ok_all else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AION operator control plane")
    sub = parser.add_subparsers(dest="command")

    for cmd in ("start", "stop", "restart"):
        p = sub.add_parser(cmd)
        p.add_argument("--task", choices=["trade", "dashboard", "ops-guard", "all"], default="all")

    sub.add_parser("status")

    args = parser.parse_args(argv)
    command = args.command or "status"

    if command == "status":
        return _status()

    tasks = _selected_tasks(getattr(args, "task", "all"))
    if command == "start":
        return _start(tasks)
    if command == "stop":
        return _stop(tasks)
    if command == "restart":
        return _restart(tasks)

    return _status()


if __name__ == "__main__":
    raise SystemExit(main())
