#!/usr/bin/env python3
# tools/apply_vol_target_best.py
# Reads runs_plus/sweep_vol_target.csv (top row), writes TARGET_ANN in tools/portfolio_vol_target.py,
# rebuilds the vol-target portfolio and updates the report card.

from pathlib import Path
import pandas as pd, re, subprocess, sys, json

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
SWEEP = RUNS / "sweep_vol_target.csv"
SCRIPT = ROOT / "tools" / "portfolio_vol_target.py"
ADD_CARD = ROOT / "tools" / "add_vol_target_card.py"

def run(cmd):
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(r.returncode)

if __name__ == "__main__":
    if not SWEEP.exists():
        sys.exit("Missing runs_plus/sweep_vol_target.csv (run tools/sweep_vol_target.py first).")
    df = pd.read_csv(SWEEP)
    if df.empty:
        sys.exit("sweep_vol_target.csv is empty.")
    best = df.iloc[0]
    target = float(best["target"])

    txt = SCRIPT.read_text()
    txt = re.sub(r"(TARGET_ANN\s*=\s*)[0-9.]+", rf"\g<1>{target}", txt)
    SCRIPT.write_text(txt)
    print(f"✅ Set TARGET_ANN={target} in portfolio_vol_target.py")

    run(["python", "tools/portfolio_vol_target.py"])
    # re-add compare card
    run(["python", str(ADD_CARD)])

    # Echo final numbers
    summ = json.loads((RUNS/"final_portfolio_vt_summary.json").read_text())
    oos = summ.get("out_sample", {})
    print("OOS Sharpe=%.3f  OOS MaxDD=%.3f" % (oos.get("sharpe", float("nan")), oos.get("maxdd", float("nan"))))
    print("✅ Vol-target rebuilt and card updated.")
