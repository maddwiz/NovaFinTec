#!/usr/bin/env python3
import sys, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

def run(cmd):
    print("▶", " ".join(map(str, cmd)))
    subprocess.check_call(cmd)

if __name__ == "__main__":
    # 0) Build base report + artifacts
    run([sys.executable, str(TOOLS / "run_all_plus.py")])

    # 1) Ensure every asset has oos.csv (day-by-day scoreboard)
    run([sys.executable, str(TOOLS / "make_oos_all.py")])

    # 2) Recompute WF with add-on hooks (WF+)
    run([sys.executable, str(TOOLS / "walk_forward_plus.py")])

    # 3) Inject WF+ card into the HTML
    run([sys.executable, str(TOOLS / "add_wfplus_card.py")])

    # 4) Print baseline vs WF+ tables in terminal
    run([sys.executable, str(TOOLS / "show_results_cli.py")])

    print("\n✅ Done. Open report_all.html for the WF+ card.")
