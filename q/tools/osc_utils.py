#!/usr/bin/env python3
# tools/osc_utils.py
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def load_series(csv_path: Path):
    df = pd.read_csv(csv_path)
    # Expecting DATE,VALUE. Try to coerce if not exact.
    if "DATE" not in df.columns or "VALUE" not in df.columns:
        # best effort normalize
        cols = {c.lower(): c for c in df.columns}
        dcol = cols.get("date", list(df.columns)[0])
        vcol = cols.get("value", list(df.columns)[-1])
        df = pd.DataFrame({"DATE": df[dcol], "VALUE": df[vcol]})
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna().sort_values("DATE")
    return df

def rsi(series: pd.Series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    signal_line = line.ewm(span=signal, adjust=False).mean()
    hist = line - signal_line
    return line, signal_line, hist

def stoch(series: pd.Series, n=14, d=3):
    low_n = series.rolling(n).min()
    high_n = series.rolling(n).max()
    k = (series - low_n) / (high_n - low_n + 1e-12) * 100.0
    d_ = k.rolling(d).mean()
    return k, d_

def daily_returns(price: pd.Series):
    return price.pct_change().fillna(0.0)

def ann_sharpe(ret: pd.Series):
    m = ret.mean()
    s = ret.std()
    if s == 0 or np.isnan(s):
        return 0.0
    return float((m / s) * np.sqrt(252.0))
