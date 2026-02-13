#!/usr/bin/env python3
# tools/apply_portfolio_knobs_best.py
# Reads runs_plus/sweep_portfolio_knobs.csv (top row),
# rewrites TOP_K and CAP_PER in tools/portfolio_from_runs_plus.py,
# then rebuilds portfolio, report, regime, and regime+DNA.

from pathlib import Path
import pandas as pd, re, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
SWEEP = RUNS / "sweep_portfolio_knobs.csv"
PORT = ROOT / "tools" / "portfolio_from_runs_plus.py"

def run(cmd):
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(r.returncode)

if __name__ == "__main__":
    if not SWEEP.exists():
        sys.exit("Missing runs_plus/sweep_portfolio_knobs.csv (run tools/sweep_portfolio_knobs.py first).")
    df = pd.read_csv(SWEEP)
    if df.empty:
        sys.exit("sweep_portfolio_knobs.csv is empty.")
    best = df.iloc[0]
    topk = int(best["top_k"])
    cap  = float(best["cap_per"])

    txt = PORT.read_text()
    txt = re.sub(r"(TOP_K\s*=\s*)[0-9]+", rf"\g<1>{topk}", txt)
    txt = re.sub(r"(CAP_PER\s*=\s*)[0-9.]+", rf"\g<1>{cap}", txt)
    PORT.write_text(txt)
    print(f"✅ Applied TOP_K={topk}, CAP_PER={cap} to portfolio_from_runs_plus.py")

    # Rebuild portfolio + report
    run(["python", "tools/portfolio_from_runs_plus.py"])
    run(["python", "tools/build_report_plus.py"])

    # Rebuild regime + DNA and update card
    run(["python", "tools/make_regime.py"])
    run(["python", "tools/apply_regime_governor.py"])
    run(["python", "tools/patch_regime_weights_with_dna.py"])
    run(["python", "tools/apply_regime_governor_dna.py"])
    run(["python", "tools/add_regime_final_card_triple.py"])
    print("✅ Rebuilt portfolio + regime + dna and updated report.")
