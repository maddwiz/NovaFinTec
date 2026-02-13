#!/usr/bin/env python3
from pathlib import Path
import pandas as pd, json
p = Path("runs_plus/portfolio_plus.csv")
if not p.exists():
    print("Missing runs_plus/portfolio_plus.csv"); raise SystemExit
df = pd.read_csv(p)
print("COLUMNS:", list(df.columns))
print(df.head(3).to_string(index=False))
