#!/usr/bin/env python3
import os, sys, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
RUNS  = ROOT / "runs_plus"

def run(cmd, env=None):
    print("▶", " ".join(map(str, cmd)))
    subprocess.check_call(cmd, env=env)

if __name__ == "__main__":
    tb = RUNS / "tuning_best.json"
    if not tb.exists():
        print("(!) No tuning_best.json. Run: python tools/tune_addons.py")
        sys.exit(1)
    best = json.loads(tb.read_text())

    env = os.environ.copy()
    # ensure all add-ons are ON
    env["SKIP_DNA"] = "0"
    env["SKIP_HEARTBEAT"] = "0"
    env["SKIP_SYMBOLIC"] = "0"
    env["SKIP_REFLEXIVE"] = "0"
    # set tuned knobs
    env["DNA_THRESH"]     = str(best.get("DNA_THRESH", 0.05))
    env["REFLEXIVE_CLIP"] = str(best.get("REFLEXIVE_CLIP", 0.2))
    env["SYMBOLIC_TILT"]  = str(best.get("SYMBOLIC_TILT", 0.1))
    env["HB_MULT"]        = str(best.get("HB_MULT", 1.0))

    # build + recompute + inject + print
    run([sys.executable, str(TOOLS / "run_all_plus.py")], env=env)
    run([sys.executable, str(TOOLS / "make_oos_all.py")], env=env)
    run([sys.executable, str(TOOLS / "walk_forward_plus.py")], env=env)
    run([sys.executable, str(TOOLS / "add_wfplus_card_any.py")], env=env)
    run([sys.executable, str(TOOLS / "show_results_cli.py")], env=env)

    print("\n✅ Used tuned best settings from runs_plus/tuning_best.json")
    print("Open: report_best_plus.html")
