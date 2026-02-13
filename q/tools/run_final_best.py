#!/usr/bin/env python3
import os, sys, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
RUNS  = ROOT / "runs_plus"

def run(cmd, env=None):
    print("▶", " ".join(map(str, cmd)))
    subprocess.check_call(cmd, env=env)

def load_addon_env(env):
    # ensure add-ons ON
    env["SKIP_DNA"] = "0"
    env["SKIP_HEARTBEAT"] = "0"
    env["SKIP_SYMBOLIC"] = "0"
    env["SKIP_REFLEXIVE"] = "0"
    # tuned add-on knobs
    tb = RUNS / "tuning_best.json"
    if tb.exists():
        best = json.loads(tb.read_text())
        env["DNA_THRESH"]     = str(best.get("DNA_THRESH", 0.05))
        env["REFLEXIVE_CLIP"] = str(best.get("REFLEXIVE_CLIP", 0.2))
        env["SYMBOLIC_TILT"]  = str(best.get("SYMBOLIC_TILT", 0.1))
        env["HB_MULT"]        = str(best.get("HB_MULT", 1.0))
    else:
        env.setdefault("DNA_THRESH", "0.05")
        env.setdefault("REFLEXIVE_CLIP", "0.2")
        env.setdefault("SYMBOLIC_TILT", "0.1")
        env.setdefault("HB_MULT", "1.0")
    # meta-council defaults (will be overridden by meta_tuning_best if present)
    env.setdefault("META_STRENGTH", "0.5")
    env.setdefault("META_SIGN_THRESH", "0.05")
    env.setdefault("META_REQUIRE_AGREE", "1")
    # if meta tuner exists, override
    mb = RUNS / "meta_tuning_best.json"
    if mb.exists():
        mbest = json.loads(mb.read_text())
        env["META_STRENGTH"] = str(mbest.get("META_STRENGTH", float(env["META_STRENGTH"])))
        env["META_SIGN_THRESH"] = str(mbest.get("META_SIGN_THRESH", float(env["META_SIGN_THRESH"])))
        env["META_REQUIRE_AGREE"] = "1" if int(mbest.get("META_REQUIRE_AGREE", int(env["META_REQUIRE_AGREE"]))) == 1 else "0"
    return env

def load_portfolio_env(env):
    pb = RUNS / "portfolio_tuning_best.json"
    if pb.exists():
        bestp = json.loads(pb.read_text())
        env["CAP_PER"]  = str(bestp.get("CAP_PER", 0.10))
        env["COST_BPS"] = str(bestp.get("COST_BPS", 2.0))
        env["LOOKBACK"] = str(int(bestp.get("LOOKBACK", 63)))
    else:
        env.setdefault("CAP_PER", "0.10")
        env.setdefault("COST_BPS", "2.0")
        env.setdefault("LOOKBACK", "63")
    return env

if __name__ == "__main__":
    env = load_addon_env(os.environ.copy())

    run([sys.executable, str(TOOLS / "run_all_plus.py")], env=env)
    run([sys.executable, str(TOOLS / "make_hive_council.py")], env=env)
    run([sys.executable, str(TOOLS / "make_oos_all.py")], env=env)
    run([sys.executable, str(TOOLS / "walk_forward_plus.py")], env=env)

    env = load_portfolio_env(env)
    run([sys.executable, str(TOOLS / "build_portfolio_plus.py")], env=env)

    run([sys.executable, str(TOOLS / "add_wfplus_card_any.py")], env=env)
    run([sys.executable, str(TOOLS / "add_hive_meta_card.py")], env=env)
    run([sys.executable, str(TOOLS / "add_portfolio_card.py")], env=env)
    run([sys.executable, str(TOOLS / "add_portfolio_equity_chart.py")], env=env)

    run([sys.executable, str(TOOLS / "show_results_cli.py")], env=env)
    print("\n✅ All done. Open: report_best_plus.html (or report_all.html)")
