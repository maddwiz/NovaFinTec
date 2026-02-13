#!/usr/bin/env python3
# tools/apply_portfolio_micro_best.py
# Reads runs_plus/sweep_portfolio_micro.csv (top row), writes TOP_K and CAP_PER
# into tools/portfolio_from_runs_plus.py, rebuilds Main + Regime(+DNA) + VolTarget,
# re-adds cards, and echoes final OOS metrics.

from pathlib import Path
import pandas as pd, re, subprocess, sys, json

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
SWEEP = RUNS / "sweep_portfolio_micro.csv"
PORT  = ROOT / "tools" / "portfolio_from_runs_plus.py"

def run(cmd):
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(r.returncode)

def read_json(p):
    p = RUNS/p
    return json.loads(p.read_text()) if p.exists() else {}

if __name__ == "__main__":
    if not SWEEP.exists():
        sys.exit("Missing runs_plus/sweep_portfolio_micro.csv (run tools/sweep_portfolio_micro.py first).")
    df = pd.read_csv(SWEEP)
    if df.empty:
        sys.exit("sweep_portfolio_micro.csv is empty.")
    best = df.iloc[0]
    topk = int(best["top_k"])
    cap  = float(best["cap_per"])

    txt = PORT.read_text()
    txt = re.sub(r"(TOP_K\s*=\s*)[0-9]+", rf"\g<1>{topk}", txt)
    txt = re.sub(r"(CAP_PER\s*=\s*)[0-9.]+", rf"\g<1>{cap}", txt)
    PORT.write_text(txt)
    print(f"✅ Applied TOP_K={topk}, CAP_PER={cap} to portfolio_from_runs_plus.py")

    # Rebuild main + report
    run(["python", "tools/portfolio_from_runs_plus.py"])
    run(["python", "tools/build_report_plus.py"])

    # Rebuild regime + DNA and update card
    run(["python", "tools/make_regime.py"])
    run(["python", "tools/apply_regime_governor.py"])
    run(["python", "tools/patch_regime_weights_with_dna.py"])
    run(["python", "tools/apply_regime_governor_dna.py"])
    run(["python", "tools/add_regime_final_card_triple.py"])

    # Rebuild vol-target and update card
    run(["python", "tools/portfolio_vol_target.py"])
    run(["python", "tools/add_vol_target_card.py"])

    # Echo scoreboard
    main = read_json("final_portfolio_summary.json").get("out_sample", {})
    reg  = read_json("final_portfolio_regime_summary.json").get("out_sample", {})
    dna  = read_json("final_portfolio_regime_dna_summary.json").get("out_sample", {})
    vt   = read_json("final_portfolio_vt_summary.json").get("out_sample", {})
    print("Main   OOS: Sharpe=%.3f MaxDD=%.3f" % (main.get("sharpe",float("nan")), main.get("maxdd",float("nan"))))
    print("Regime OOS: Sharpe=%.3f MaxDD=%.3f" % (reg.get("sharpe",float("nan")),  reg.get("maxdd",float("nan"))))
    print("DNA    OOS: Sharpe=%.3f MaxDD=%.3f" % (dna.get("sharpe",float("nan")),  dna.get("maxdd",float("nan"))))
    print("VTarget OOS: Sharpe=%.3f MaxDD=%.3f" % (vt.get("sharpe",float("nan")), vt.get("maxdd",float("nan"))))
    print("✅ Rebuilt portfolio + regime + dna + vol-target and updated cards.")
