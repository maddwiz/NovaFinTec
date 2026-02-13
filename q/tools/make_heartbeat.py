#!/usr/bin/env python3
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmods.heartbeat import compute_heartbeat

def load_prices():
    frames = []
    for p in Path("data").glob("*.csv"):
        sym = p.stem.replace("_prices","")
        df = pd.read_csv(p)

        # date column (flexible)
        date_col = None
        for dc in ("date","Date","DATE"):
            if dc in df.columns:
                date_col = dc
                break
        if date_col is None:
            continue
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col).sort_index()

        # price-like column (flexible)
        val_col = None
        for vc in ("close","Close","adj_close","Adj Close","AdjClose","price","Price","PX_LAST"):
            if vc in df.columns:
                val_col = vc
                break
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
    out_json, out_png = compute_heartbeat(prices)
    print("✅ Wrote", out_json)
    print("✅ Wrote", out_png)
