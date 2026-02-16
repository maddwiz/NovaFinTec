from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .. import config as cfg
from .ops_guard import find_task_pids, start_task, status_snapshot, stop_task, task_module
from .runtime_decision import runtime_decision_summary
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


def _overlay_runtime_status(path: Path, max_age_hours: float) -> dict:
    out = {
        "exists": False,
        "age_hours": None,
        "age_source": "unknown",
        "generated_at_utc": None,
        "max_age_hours": float(max_age_hours),
        "stale": False,
        "runtime_context_present": False,
        "runtime_context": {},
        "risk_flags": [],
    }
    if not isinstance(path, Path) or not path.exists():
        return out
    out["exists"] = True
    payload = _read_json(path, {})
    mtime_age = None
    try:
        ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        mtime_age = float((datetime.now(timezone.utc) - ts).total_seconds() / 3600.0)
    except Exception:
        mtime_age = None

    if isinstance(payload, dict):
        raw_ts = payload.get("generated_at_utc", payload.get("generated_at"))
        s = str(raw_ts or "").strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s) if s else None
            if dt is not None:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                dt = dt.astimezone(timezone.utc)
                out["generated_at_utc"] = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                out["age_hours"] = float((datetime.now(timezone.utc) - dt).total_seconds() / 3600.0)
                out["age_source"] = "payload"
        except Exception:
            pass

    if out["age_hours"] is None and isinstance(mtime_age, float):
        out["age_hours"] = float(max(0.0, mtime_age))
        out["age_source"] = "mtime"
    if isinstance(out["age_hours"], float):
        out["age_hours"] = float(max(0.0, out["age_hours"]))
    if isinstance(out["age_hours"], float) and float(max_age_hours) > 0:
        out["stale"] = bool(out["age_hours"] > float(max_age_hours))

    if not isinstance(payload, dict):
        return out
    ext_ctx = payload.get("runtime_context", {})
    if isinstance(ext_ctx, dict):
        out["runtime_context_present"] = len(ext_ctx) > 0
        out["runtime_context"] = ext_ctx
        flags = ext_ctx.get("risk_flags", [])
        if isinstance(flags, list):
            out["risk_flags"] = [str(x).strip().lower() for x in flags if str(x).strip()]
    return out


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
    ext_rt = _overlay_runtime_status(cfg.EXT_SIGNAL_FILE, max_age_hours=float(cfg.EXT_SIGNAL_MAX_AGE_HOURS))
    ext_ctx = ext_rt.get("runtime_context", {}) if isinstance(ext_rt, dict) else {}
    if not isinstance(ext_ctx, dict):
        ext_ctx = {}
    decision = runtime_decision_summary(
        runtime_controls if isinstance(runtime_controls, dict) else {},
        ext_rt,
        ext_rt.get("risk_flags", []) if isinstance(ext_rt, dict) else [],
    )
    out = {
        "tasks": snap,
        "ops_guard_status": ops_status,
        "runtime_controls": runtime_controls if isinstance(runtime_controls, dict) else {},
        "runtime_decision": decision,
        "runtime_remediation": decision.get("recommended_actions", []),
        "runtime_controls_age_sec": runtime_controls_age_sec,
        "runtime_controls_stale_threshold_sec": runtime_controls_stale_threshold_sec,
        "runtime_controls_stale": runtime_controls_stale,
        "external_runtime_context": ext_ctx,
        "external_overlay_runtime": ext_rt,
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
