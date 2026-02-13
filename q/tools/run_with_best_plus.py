#!/usr/bin/env python3
import os, sys, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
RUNS  = ROOT / "runs_plus"

CONFIGS = {
    "baseline (Council only)":             {"dna":False,"heartbeat":False,"symbolic":False,"reflexive":False,"hive":False},
    "+ DNA":                               {"dna":True, "heartbeat":False,"symbolic":False,"reflexive":False,"hive":False},
    "+ Heartbeat":                         {"dna":True, "heartbeat":True, "symbolic":False,"reflexive":False,"hive":False},
    "+ Symbolic":                          {"dna":True, "heartbeat":True, "symbolic":True, "reflexive":False,"hive":False},
    "+ Reflexive":                         {"dna":True, "heartbeat":True, "symbolic":True, "reflexive":True, "hive":False},
    "+ Hive (all current)":                {"dna":True, "heartbeat":True, "symbolic":True, "reflexive":True, "hive":True},
}

def env_from(flags):
    e = os.environ.copy()
    e["SKIP_DNA"]       = "0" if flags.get("dna")       else "1"
    e["SKIP_HEARTBEAT"] = "0" if flags.get("heartbeat") else "1"
    e["SKIP_SYMBOLIC"]  = "0" if flags.get("symbolic")  else "1"
    e["SKIP_REFLEXIVE"] = "0" if flags.get("reflexive") else "1"
    e["SKIP_HIVE"]      = "0" if flags.get("hive")      else "1"
    return e

def run(cmd, env=None):
    print("▶", " ".join(map(str, cmd)))
    subprocess.check_call(cmd, env=env)

if __name__ == "__main__":
    choice_p = RUNS / "addon_choice_plus.json"
    if not choice_p.exists():
        print("(!) No addon_choice_plus.json yet. Run: python tools/run_decide_everything_plus.py")
        sys.exit(1)

    choice = json.loads(choice_p.read_text())
    label  = choice.get("best", {}).get("config", "+ Reflexive")
    flags  = CONFIGS.get(label, CONFIGS["+ Reflexive"])
    env    = env_from(flags)

    # 0) Build artifacts under best flags
    run([sys.executable, str(TOOLS / "run_all_plus.py")], env=env)

    # 1) Ensure oos.csv, recompute WF+ under best flags
    run([sys.executable, str(TOOLS / "make_oos_all.py")], env=env)
    run([sys.executable, str(TOOLS / "walk_forward_plus.py")], env=env)

    # 2) Inject WF+ card into both reports
    run([sys.executable, str(TOOLS / "add_wfplus_card_any.py")], env=env)

    # 3) Print terminal tables
    run([sys.executable, str(TOOLS / "show_results_cli.py")], env=env)

    print("\n✅ Best config used:", label)
    print("Open:")
    print("  - report_best_plus.html")
