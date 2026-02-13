# qmods/io.py — smart CSV loader that finds a "Close" series
import pandas as pd
import numpy as np
from pathlib import Path

CANDIDATES = [
    "Close","Adj Close","AdjClose","PX_LAST","LAST","Last","Settle","Price",
    "CLOSE","close","value","VALUE"
]

def _pick_price_column(df: pd.DataFrame) -> str | None:
    # 1) try known names
    for c in CANDIDATES:
        if c in df.columns:
            return c
    # 2) if there is exactly one non-date numeric column, use it
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) == 1:
        return num_cols[0]
    # 3) FRED style: last column is the series
    if num_cols:
        return num_cols[-1]
    return None

def load_close(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    # Clean infinities/empties
    df = df.replace([np.inf,-np.inf], np.nan)
    col = _pick_price_column(df)
    if col is None:
        raise KeyError(f"No price-like column found in {path.name}. Columns={list(df.columns)}")
    s = df[col].astype(float)
    s.name = "Close"
    s = s.dropna()
    return s
