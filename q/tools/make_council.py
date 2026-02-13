#!/usr/bin/env python3
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmods.council import run_council

def load_prices():
    frames = []
    for p in Path("data").glob("*.csv"):
        sym = p.stem.replace("_prices","")
        df = pd.read_csv(p)
        # find a date column
        for dc in ("date","Date","DATE"):
            if dc in df.columns:
                df[dc] = pd.to_datetime(df[dc], errors="coerce")
                df = df.set_index(dc).sort_index()
                break
        else:
            continue
        # find a price-like column
        val_col = None
        for vc in ("close","Close","adj_close","Adj Close","AdjClose","price","Price","PX_LAST"):
            if vc in df.columns:
                val_col = vc; break
        if val_col is None:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                continue
            val_col = num_cols[-1]
        sub = df[[val_col]].rename(columns={val_col: sym})
        frames.append(sub)
    if not frames:
        raise SystemExit("No usable CSVs in data/ (need date + price column).")
    prices = pd.concat(frames, axis=1).dropna(how="all").ffill()
    return prices

if __name__ == "__main__":
    prices = load_prices()
    run_council(prices)
    print("✅ Wrote runs_plus/council.json")
