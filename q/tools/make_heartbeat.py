#!/usr/bin/env python3
from pathlib import Path
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmods.heartbeat import compute_heartbeat, compute_heartbeat_from_returns

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

def load_returns():
    for rel in ["runs_plus/daily_returns.csv", "daily_returns.csv", "runs_plus/wf_oos_returns.csv"]:
        p = ROOT / rel
        if not p.exists():
            continue
        try:
            a = np.loadtxt(p, delimiter=",").ravel()
        except Exception:
            try:
                a = np.loadtxt(p, delimiter=",", skiprows=1).ravel()
            except Exception:
                continue
        a = np.asarray(a, float).ravel()
        if a.size >= 20:
            idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=a.size, freq="B")
            return pd.Series(a, index=idx)
    return None

if __name__ == "__main__":
    try:
        prices = load_prices()
        out_json, out_png = compute_heartbeat(prices)
        print("✅ Wrote", out_json)
        print("✅ Wrote", out_png)
        print("✅ Wrote runs_plus/heartbeat_bpm.csv")
        print("✅ Wrote runs_plus/heartbeat_exposure_scaler.csv")
    except (Exception, SystemExit):
        rets = load_returns()
        if rets is None:
            raise SystemExit("No usable data/ prices or returns found for heartbeat.")
        out_json, out_png = compute_heartbeat_from_returns(rets)
        print("✅ Wrote", out_json)
        print("✅ Wrote", out_png)
        print("✅ Wrote runs_plus/heartbeat_bpm.csv")
        print("✅ Wrote runs_plus/heartbeat_exposure_scaler.csv")
