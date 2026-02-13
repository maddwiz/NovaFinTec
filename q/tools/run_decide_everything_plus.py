#!/usr/bin/env python3
import sys, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

def run(cmd, env=None):
    print("▶", " ".join(map(str, cmd)))
    subprocess.check_call(cmd, env=env)

if __name__ == "__main__":
    # build base artifacts
    run([sys.executable, str(TOOLS / "run_all_plus.py")])
    # ensure oos exist
    run([sys.executable, str(TOOLS / "make_oos_all.py")])
    # run WF+ ablation & best pick
    run([sys.executable, str(TOOLS / "eval_addons_plus.py")])
    print("\nOpen one of these:")
    print("  - report_all.html         (current flags)")
    print("  - report_best_plus.html   (best combo from WF+ ablation)")
