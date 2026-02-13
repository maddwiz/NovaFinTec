#!/usr/bin/env python3
"""
tune_portfolio.py v2 (with HIVE_CAP)

Sweeps CAP_PER, COST_BPS, LOOKBACK, HIVE_CAP.
For each combo:
  - runs build_portfolio_plus.py with env overrides
  - reads portfolio_summary.json
Outputs:
  runs_plus/portfolio_tuning.csv
  runs_plus/portfolio_tuning_best.json
"""

import os, sys, json, subprocess, itertools
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
RUNS  = ROOT / "runs_plus"

CAPS    = [0.04, 0.05, 0.06, 0.08, 0.10]
COSTS   = [0.5, 1.0, 2.0]
LOOKS   = [63, 84, 105]
HIVECAP = [0.25, 0.30, 0.35, 0.40, 0.50]

def run(cmd, env=None):
    print("▶", " ".join(map(str, cmd)))
    subprocess.check_call(cmd, env=env)

def read_summary():
    p = RUNS / "portfolio_summary.json"
    if not p.exists(): return None
    try: return json.loads(p.read_text())
    except: return None

def main():
    results = []
    for cap, cost, look, hcap in itertools.product(CAPS, COSTS, LOOKS, HIVECAP):
        env = os.environ.copy()
        env["CAP_PER"]  = str(cap)
        env["COST_BPS"] = str(cost)
        env["LOOKBACK"] = str(look)
        env["HIVE_CAP"] = str(hcap)
        run([sys.executable, str(TOOLS / "build_portfolio_plus.py")], env=env)
        s = read_summary() or {}
        results.append({
            "CAP_PER": cap,
            "COST_BPS": cost,
            "LOOKBACK": look,
            "HIVE_CAP": hcap,
            "sharpe": s.get("sharpe"),
            "hit": s.get("hit"),
            "maxDD": s.get("maxDD"),
        })

    df = pd.DataFrame(results)
    out_csv = RUNS / "portfolio_tuning.csv"
    df.to_csv(out_csv, index=False)
    dff = df.dropna(subset=["sharpe"]).copy()
    dff["dd_score"] = dff["maxDD"]
    dff = dff.sort_values(["sharpe","dd_score","hit"], ascending=[False, False, False])
    best = dff.iloc[0].to_dict()
    (RUNS / "portfolio_tuning_best.json").write_text(json.dumps(best, indent=2))
    print("\nBEST PORTFOLIO SETTINGS")
    print(json.dumps(best, indent=2))
    print(f"\n✅ Saved {out_csv.as_posix()} and portfolio_tuning_best.json")

if __name__ == "__main__":
    main()
