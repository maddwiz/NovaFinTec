#!/usr/bin/env python3
# Builds minimal inputs so Phase-2 can run:
# - runs_plus/daily_returns.csv   (portfolio daily returns, 1 column)
# - portfolio_weights.csv         (T x N equal-weights path)
# Also writes duplicates under runs_plus/ for convenience.

import csv
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNS = ROOT / "runs_plus"; RUNS.mkdir(exist_ok=True)


def sanitize_returns(r: np.ndarray, clip_abs: float) -> tuple[np.ndarray, int]:
    arr = np.asarray(r, float).ravel()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    # Do not allow <= -100% and clip extreme positive spikes (bad ticks/splits).
    lo = -0.95
    hi = float(max(0.01, clip_abs))
    clipped = np.clip(arr, lo, hi)
    n_clipped = int(np.sum(np.abs(clipped - arr) > 1e-12))
    return clipped, n_clipped

def read_close_series(fp):
    dates, close = [], []
    date_keys = ("Date", "DATE", "date", "timestamp", "Timestamp", "datetime", "Datetime")
    close_keys = ("Adj Close", "adj_close", "AdjClose", "Close", "close", "price", "Price")
    with open(fp) as f:
        r = csv.DictReader(f)
        for row in r:
            d = None
            for k in date_keys:
                if k in row and row.get(k) not in (None, ""):
                    d = row.get(k)
                    break
            c = None
            for k in close_keys:
                if k in row and row.get(k) not in (None, ""):
                    c = row.get(k)
                    break
            if d is None or c is None:
                continue
            try:
                close_val = float(c)
            except Exception:
                continue
            dates.append(d)
            close.append(close_val)
    if len(close) < 2:
        raise ValueError("not enough usable close rows")
    return np.array(dates), np.array(close, float)

if __name__ == "__main__":
    files = sorted(DATA.glob("*.csv"))
    if not files:
        print("(!) No data/*.csv found — cannot build returns."); raise SystemExit(0)

    # Load all assets, align to shortest length
    series = []
    names = []
    clip_abs = float(np.clip(float(os.getenv("Q_ASSET_RET_CLIP", "0.35")), 0.01, 5.0))
    clip_events = 0
    for fp in files:
        try:
            d,c = read_close_series(fp)
            r = np.diff(c) / (c[:-1] + 1e-12)
            r, n_clip = sanitize_returns(r, clip_abs)
            clip_events += int(n_clip)
            series.append(r)
            names.append(fp.stem)
        except Exception as e:
            print(f"skip {fp.name}: {e}")
    if not series:
        print("(!) No usable assets."); raise SystemExit(0)

    T = min(len(s) for s in series)
    series = [s[-T:] for s in series]  # align tails
    R = np.stack(series, axis=1)       # [T, N]
    np.savetxt(RUNS/"asset_returns.csv", R, delimiter=",")
    (RUNS/"asset_names.csv").write_text("\n".join(names), encoding="utf-8")
    (RUNS/"asset_returns_info.json").write_text(
        json.dumps(
            {
                "rows": int(R.shape[0]),
                "assets": int(R.shape[1]),
                "asset_return_clip_abs": float(clip_abs),
                "clip_events": int(clip_events),
                "asset_return_min": float(np.min(R)),
                "asset_return_max": float(np.max(R)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
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
