from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from .. import config as cfg

TASK_TO_MODULE = {
    "trade": "aion.exec.paper_loop",
    "dashboard": "aion.exec.dashboard",
    "ops-guard": "aion.exec.ops_guard",
}


def _now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _repo_root() -> Path:
    # /.../aion/aion/exec/ops_guard.py -> /.../aion
    return Path(__file__).resolve().parents[2]


def _incident(level: str, msg: str, task: str | None = None):
    cfg.OPS_GUARD_INCIDENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {level.upper()}"
    if task:
        line += f" task={task}"
    line += f" {msg}\n"
    with cfg.OPS_GUARD_INCIDENT_LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def task_module(task: str) -> str | None:
    return TASK_TO_MODULE.get(str(task or "").strip().lower())


def find_task_pids(task: str, ps_text: str | None = None) -> list[int]:
    module = task_module(task)
    if not module:
        return []

    if ps_text is None:
        try:
            ps_text = subprocess.check_output(["ps", "-axo", "pid=,command="], text=True)
        except Exception:
            return []

    out: list[int] = []
    for raw in ps_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        pid_raw, cmd = parts
        try:
            pid = int(pid_raw)
        except Exception:
            continue
        if f"-m {module}" not in cmd:
            continue
        out.append(pid)
    return sorted(set(out))


def start_task(task: str, *, root: Path | None = None, log_dir: Path | None = None) -> bool:
    t = str(task or "").strip().lower()
    if t not in TASK_TO_MODULE:
        return False

    root = root or _repo_root()
    log_dir = log_dir or cfg.LOG_DIR
    run_script = root / "run_aion.sh"
    if not run_script.exists():
        _incident("error", f"run_aion.sh missing at {run_script}", task=t)
        return False

    log_dir.mkdir(parents=True, exist_ok=True)
    out_log = log_dir / f"{t}.out"
    try:
        env = dict(os.environ)
        env["AION_TASK"] = t
        with out_log.open("ab") as fout:
            proc = subprocess.Popen(
                [str(run_script)],
                cwd=str(root),
                env=env,
                stdout=fout,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        for _ in range(80):
            if find_task_pids(t):
                return True
            # Trade can spend significant time in preflight before paper_loop appears.
            if proc.poll() is not None:
                break
            time.sleep(0.25)
        if proc.poll() is None:
            _incident("info", "task launcher alive; treating as booting", task=t)
            if find_task_pids(t):
                return True
            return True
        _incident("error", f"launch exited rc={proc.returncode} before task detected", task=t)
        return False
    except Exception as exc:
        _incident("error", f"failed to start task ({exc})", task=t)
        return False


def stop_task(task: str, *, timeout_sec: float = 6.0) -> int:
    pids = find_task_pids(task)
    if not pids:
        return 0

    for pid in pids:
        try:
            import os

            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass

    deadline = time.time() + max(0.0, float(timeout_sec))
    while time.time() < deadline:
        alive = find_task_pids(task)
        if not alive:
            return len(pids)
        time.sleep(0.2)

    for pid in find_task_pids(task):
        try:
            import os

            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
    return len(pids)


def _can_restart(now_ts: float, history: list[float], cooldown_sec: float, max_restarts_per_hour: int):
    keep = [ts for ts in history if (now_ts - ts) <= 3600.0]
    history[:] = keep
    if history and (now_ts - history[-1]) < max(0.0, float(cooldown_sec)):
        return False, "cooldown"
    if len(history) >= max(1, int(max_restarts_per_hour)):
        return False, "hourly_limit"
    return True, "ok"


def status_snapshot(targets: list[str] | None = None) -> dict:
    tgs = [str(t).strip().lower() for t in (targets or cfg.OPS_GUARD_TARGETS) if str(t).strip()]
    out = {}
    for t in tgs:
        module = task_module(t)
        if not module:
            continue
        pids = find_task_pids(t)
        out[t] = {
            "module": module,
            "running": bool(pids),
            "pids": pids,
        }
    return out


def _write_status(payload: dict):
    cfg.OPS_GUARD_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    cfg.OPS_GUARD_STATUS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def guard_cycle(state: dict) -> dict:
    now_ts = time.time()
    now_iso = _now_iso()

    targets = [str(t).strip().lower() for t in cfg.OPS_GUARD_TARGETS if str(t).strip()]
    targets = [t for t in targets if task_module(t)]

    restart_history = state.setdefault("restart_history", {})
    restarted: dict[str, str] = {}
    skipped: dict[str, str] = {}

    snapshot = status_snapshot(targets)
    for task, meta in snapshot.items():
        if meta.get("running"):
            continue

        hist = restart_history.setdefault(task, [])
        allowed, reason = _can_restart(
            now_ts=now_ts,
            history=hist,
            cooldown_sec=cfg.OPS_GUARD_RESTART_COOLDOWN_SEC,
            max_restarts_per_hour=cfg.OPS_GUARD_MAX_RESTARTS_PER_HOUR,
        )
        if not allowed:
            skipped[task] = reason
            continue

        ok = start_task(task)
        if ok:
            hist.append(now_ts)
            restarted[task] = "started"
            _incident("warn", "task not running, auto-restarted", task=task)
        else:
            restarted[task] = "start_failed"
            _incident("error", "task not running, restart failed", task=task)

    post = status_snapshot(targets)
    restarts = {}
    for task in targets:
        hist = restart_history.setdefault(task, [])
        hist[:] = [ts for ts in hist if (now_ts - ts) <= 3600.0]
        restarts[task] = {
            "restarts_last_hour": len(hist),
            "last_restart_ts": dt.datetime.fromtimestamp(hist[-1]).isoformat(timespec="seconds") if hist else None,
            "attempt": restarted.get(task),
            "skip": skipped.get(task),
        }

    payload = {
        "ts": now_iso,
        "targets": targets,
        "running": post,
        "restarts": restarts,
        "config": {
            "interval_sec": int(cfg.OPS_GUARD_INTERVAL_SEC),
            "restart_cooldown_sec": int(cfg.OPS_GUARD_RESTART_COOLDOWN_SEC),
            "max_restarts_per_hour": int(cfg.OPS_GUARD_MAX_RESTARTS_PER_HOUR),
        },
    }
    _write_status(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AION ops guard")
    parser.add_argument("--once", action="store_true", help="run a single guard cycle and exit")
    args = parser.parse_args(argv)

    _incident("info", "ops_guard started")
    state: dict = {"restart_history": {}}
    interval = max(5, int(cfg.OPS_GUARD_INTERVAL_SEC))

    if args.once:
        guard_cycle(state)
        _incident("info", "ops_guard cycle complete")
        return 0

    try:
        while True:
            guard_cycle(state)
            time.sleep(interval)
    except KeyboardInterrupt:
        _incident("info", "ops_guard stopped")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
