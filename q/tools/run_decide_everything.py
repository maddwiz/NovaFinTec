#!/usr/bin/env python3
import sys, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

def run(cmd):
    print("▶", " ".join(map(str, cmd)))
    subprocess.check_call(cmd)

if __name__ == "__main__":
    # 1) Walk-forward batch (produces runs_plus/walk_forward_table.csv)
    if (TOOLS/"walk_forward_batch.py").exists():
        run([sys.executable, str(TOOLS/"walk_forward_batch.py")])
    else:
        print("(!) tools/walk_forward_batch.py not found — skipping WF step.")
    # 2) Auto-judge (runs full builds internally, prints matrix, writes report_best.html)
    run([sys.executable, str(TOOLS/"auto_judge_addons.py")])
    print("\nDone. Open one of these:")
    print("  - report_all.html  (everything ON)")
    print("  - report_best.html (best config chosen automatically)")
