
import os, re
import pandas as pd
from .utils import ensure_datetime_index

PRICE_CANDIDATES = ["Close","Adj Close","Adj_Close","PX_LAST","value","Close/Last","Price","last","Settle","Settlement Price","close"]

def load_csv(path: str, date_col_guess="Date"):
    df = pd.read_csv(path)
    # find date col
    date_col = None
    cols = [c for c in df.columns]
    for c in cols:
        if str(c).strip().lower() in ["date","time","timestamp"]:
            date_col = c; break
    if date_col is None:
        for c in cols:
            if "date" in str(c).lower():
                date_col = c; break
    # find price col
    price_col = None
    for cand in PRICE_CANDIDATES:
        for c in cols:
            if str(c).strip().lower() == cand.lower():
                price_col = c; break
        if price_col: break
    if price_col is None:
        # fallback: choose first numeric column that isn't open/high/low/volume
        for c in cols:
            if str(c).lower() in ["open","high","low","volume","vol","amount"]:
                continue
            try:
                if pd.api.types.is_numeric_dtype(df[c]):
                    price_col = c; break
            except Exception:
                pass

    if date_col is None or price_col is None:
        raise ValueError(f"Could not find date/close in {os.path.basename(path)}; cols={list(df.columns)[:6]}...")

    out = df[[date_col, price_col]].rename(columns={date_col:"Date", price_col:"Close"}).copy()
    if out["Close"].dtype == object:
        out["Close"] = (out["Close"].astype(str)
                        .str.replace(r"[^0-9\.\-]","", regex=True))
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna().copy()
    out = ensure_datetime_index(out)
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    return out

def align_series(dfs, how="inner"):
    # align multiple dataframes on common dates
    idx = None
    for df in dfs:
        idx = df.index if idx is None else idx.intersection(df.index)
    aligned = [df.reindex(idx).ffill().dropna() for df in dfs]
    return aligned
