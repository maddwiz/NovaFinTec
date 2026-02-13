#!/usr/bin/env python3
# Builds minimal inputs so Phase-2 can run:
# - runs_plus/daily_returns.csv   (portfolio daily returns, 1 column)
# - portfolio_weights.csv         (T x N equal-weights path)
# Also writes duplicates under runs_plus/ for convenience.

import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNS = ROOT / "runs_plus"; RUNS.mkdir(exist_ok=True)

def read_close_series(fp):
    dates, close = [], []
    with open(fp) as f:
        r = csv.DictReader(f)
        for row in r:
            dates.append(row["Date"])
            # allow common schemas
            c = row.get("Adj Close") or row.get("Close") or row.get("close")
            close.append(float(c))
    return np.array(dates), np.array(close, float)

if __name__ == "__main__":
    files = sorted(DATA.glob("*.csv"))
    if not files:
        print("(!) No data/*.csv found — cannot build returns."); raise SystemExit(0)

    # Load all assets, align to shortest length
    series = []
    for fp in files:
        try:
            d,c = read_close_series(fp)
            r = np.diff(c) / (c[:-1] + 1e-12)
            series.append(r)
        except Exception as e:
            print(f"skip {fp.name}: {e}")
    if not series:
        print("(!) No usable assets."); raise SystemExit(0)

    T = min(len(s) for s in series)
    series = [s[-T:] for s in series]  # align tails
    R = np.stack(series, axis=1)       # [T, N]
    # Equal-weight portfolio daily returns:
    N = R.shape[1]
    port = np.mean(R, axis=1)          # simple average
    np.savetxt(RUNS/"daily_returns.csv", port, delimiter=",")
    (ROOT/"daily_returns.csv").write_text("\n".join(str(x) for x in port))

    # Equal-weights path (constant through time)
    W = np.ones((T, N), float) / N
    np.savetxt(ROOT/"portfolio_weights.csv", W, delimiter=",")
    np.savetxt(RUNS/"portfolio_weights.csv", W, delimiter=",")
    print(f"✅ Wrote {RUNS/'daily_returns.csv'}  and portfolio_weights.csv (T={T}, N={N})")
