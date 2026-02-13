#!/usr/bin/env python3
# tools/vol_utils.py
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def load_series(csv_path: Path):
    df = pd.read_csv(csv_path)
    if "DATE" not in df.columns or "VALUE" not in df.columns:
        cols = {c.lower(): c for c in df.columns}
        dcol = cols.get("date", list(df.columns)[0])
        vcol = cols.get("value", list(df.columns)[-1])
        df = pd.DataFrame({"DATE": df[dcol], "VALUE": df[vcol]})
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna().sort_values("DATE")
    return df

def daily_returns(price: pd.Series):
    return price.pct_change().fillna(0.0)

def realized_vol(r, window=21):
    return r.rolling(window).std() * np.sqrt(252)

def ann_sharpe(r):
    r = pd.Series(r).dropna()
    s = r.std()
    if s == 0 or np.isnan(s): return 0.0
    return float((r.mean() / s) * np.sqrt(252.0))
