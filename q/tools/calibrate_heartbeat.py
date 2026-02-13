#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd
from qmods.heartbeat import realized_vol, HBConfig

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"

def load_prices():
    frames = []
    for p in Path(ROOT/"data").glob("*.csv"):
        sym = p.stem.replace("_prices","")
        df = pd.read_csv(p)
        date_col = next((c for c in ("date","Date","DATE") if c in df.columns), None)
        if not date_col: continue
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col).sort_index()
        price_col = next((c for c in ("close","Close","adj_close","Adj Close","AdjClose","price","Price","PX_LAST") if c in df.columns), None)
        if not price_col:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols: continue
            price_col = num_cols[-1]
        frames.append(df[[price_col]].rename(columns={price_col:sym}))
    if not frames:
        raise SystemExit("No usable CSVs in data/")
    return pd.concat(frames, axis=1).dropna(how="all").ffill()

if __name__ == "__main__":
    prices = load_prices()
    cfg = HBConfig()
    vol = realized_vol(prices, cfg.window).dropna()
    lo = float(vol.quantile(0.20)) if not vol.empty else 0.01
    hi = float(vol.quantile(0.90)) if not vol.empty else 0.05
    out = {"suggested_vol_lo": lo, "suggested_vol_hi": hi, "window": cfg.window}
    (RUNS/"heartbeat_calibration.json").write_text(json.dumps(out, indent=2))
    print("Suggested Heartbeat vol bounds:")
    print(f"  vol_lo ≈ {lo:.4f}")
    print(f"  vol_hi ≈ {hi:.4f}")
    print(f"Saved -> { (RUNS/'heartbeat_calibration.json').as_posix() }")
