#!/usr/bin/env python3
# Runs the all-in-one, then prints WF table, then opens report.

import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

def run(rel):
    p = ROOT/rel
    print(f"\n▶ {rel}")
    subprocess.run([PY, str(p)], check=False, cwd=str(ROOT))

if __name__ == "__main__":
    run("tools/run_all_in_one_plus.py")
    run("tools/print_wf_results.py")
    try:
        subprocess.run(["open", str(ROOT/"report_best_plus.html")], check=False)
    except Exception:
        pass
