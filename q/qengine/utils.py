
import hashlib, json, os, numpy as np, pandas as pd, datetime as dt
from pathlib import Path

def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def sha1_file(path: str) -> str:
    with open(path, "rb") as f:
        return sha1_bytes(f.read())

def ensure_datetime_index(df: pd.DataFrame, date_col="Date"):
    if date_col in df.columns:
        df = df.rename(columns={date_col:"Date"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Could not create DatetimeIndex")
    return df

def to_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff().fillna(0.0)

def annualize_sharpe(daily_ret: np.ndarray, eps=1e-12):
    mu = daily_ret.mean()
    sd = daily_ret.std(ddof=1) + eps
    return (mu * 252.0) / (sd * (252.0 ** 0.5))

def max_drawdown(equity: np.ndarray) -> float:
    if len(equity)==0: return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity/peak - 1.0
    return dd.min() if len(dd)>0 else 0.0
