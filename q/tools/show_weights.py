#!/usr/bin/env python3
# tools/show_weights.py
from pathlib import Path
import pandas as pd
RUNS = Path(__file__).resolve().parents[1] / "runs_plus"
p = RUNS / "portfolio_weights.csv"
if not p.exists():
    raise SystemExit("Missing runs_plus/portfolio_weights.csv (run portfolio_from_runs_plus.py first).")
df = pd.read_csv(p)
df = df.sort_values("weight", ascending=False).reset_index(drop=True)
wsum = df["weight"].sum()
print("PORTFOLIO WEIGHTS (sum=%.3f)" % wsum)
print(df.to_string(index=False, formatters={"weight":lambda x: f"{x:0.4f}"}))
