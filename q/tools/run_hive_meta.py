#!/usr/bin/env python3
import sys, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

def run(cmd, env=None):
    print("▶", " ".join(map(str, cmd)))
    subprocess.check_call(cmd, env=env)

if __name__ == "__main__":
    run([sys.executable, str(TOOLS / "run_all_plus.py")])
    run([sys.executable, str(TOOLS / "make_hive_council.py")])
    run([sys.executable, str(TOOLS / "make_oos_all.py")])
    run([sys.executable, str(TOOLS / "walk_forward_plus.py")])
    # build portfolio + inject all cards
    run([sys.executable, str(TOOLS / "build_portfolio_plus.py")])
    run([sys.executable, str(TOOLS / "add_wfplus_card_any.py")])
    run([sys.executable, str(TOOLS / "add_hive_meta_card.py")])
    run([sys.executable, str(TOOLS / "add_portfolio_card.py")])
    run([sys.executable, str(TOOLS / "show_results_cli.py")])
    print("\nOpen: report_best_plus.html (and/or report_all.html)")
