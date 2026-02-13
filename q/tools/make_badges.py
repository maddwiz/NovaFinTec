#!/usr/bin/env python3
import json, os, platform, subprocess, sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"

def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "nogit"

def count_symbols():
    # try to infer from council.json or dna_drift.json
    c = RUNS / "council.json"
    d = RUNS / "dna_drift.json"
    if c.exists():
        try:
            data = json.loads(c.read_text())
            return len(data.get("final_weights", {}))
        except Exception:
            pass
    if d.exists():
        try:
            data = json.loads(d.read_text())
            return len(data.get("dna_drift", {}))
        except Exception:
            pass
    return None

def mtime(p: Path):
    return datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds") if p.exists() else None

def main():
    badges = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "git": git_hash(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cwd": str(ROOT),
        "symbols": count_symbols(),
        "artifacts": {
            "report_plus.html": mtime(ROOT / "report_plus.html"),
            "council.json": mtime(RUNS / "council.json"),
            "dna_drift.json": mtime(RUNS / "dna_drift.json"),
            "dna_drift.png": mtime(RUNS / "dna_drift.png"),
            "heartbeat.json": mtime(RUNS / "heartbeat.json"),
            "heartbeat.png": mtime(RUNS / "heartbeat.png"),
            "walk_forward_table.csv": mtime(RUNS / "walk_forward_table.csv"),
        }
    }
    (RUNS / "badges.json").write_text(json.dumps(badges, indent=2))
    print("✅ Wrote", (RUNS / "badges.json").as_posix())

if __name__ == "__main__":
    main()
