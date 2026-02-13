#!/usr/bin/env python3
# tools/fix_close_in_data.py
# Ensures every CSV in data/ and data_new/ has a 'Close' column.
# If not present, it copies from first available among:
# 'Adj Close','Adjusted Close','close','adj_close','Close*','Price','Last'
# If none found, it tries the right-most numeric column.

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIRS = [ROOT/"data", ROOT/"data_new"]
CANDIDATE_COLS = [
    "Close", "Adj Close", "Adjusted Close", "close", "adj_close", "Close*", "Price", "Last"
]

def pick_price_col(df):
    # 1) Direct candidates in order (case-sensitive first)
    for c in CANDIDATE_COLS:
        if c in df.columns:
            return c
    # 2) Case-insensitive fallbacks
    lower = {c.lower(): c for c in df.columns}
    for c in ["close","adj close","adjusted close","price","last","close*","adj_close"]:
        if c in lower:
            return lower[c]
    # 3) Right-most numeric column as last resort
    numerics = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numerics:
        return numerics[-1]
    return None

def ensure_close(p: Path):
    try:
        df = pd.read_csv(p)
    except Exception:
        return (False, f"skip: {p.name} (unreadable)")

    if "Close" in df.columns:
        return (False, f"ok: {p.name} (Close exists)")

    src = pick_price_col(df)
    if not src:
        return (False, f"warn: {p.name} (no numeric price-like column found)")

    try:
        df["Close"] = pd.to_numeric(df[src], errors="coerce")
        df.to_csv(p, index=False)
        return (True, f"fixed: {p.name} -> Close from '{src}'")
    except Exception as e:
        return (False, f"error: {p.name} ({e})")

if __name__ == "__main__":
    total_changed = 0
    for d in CANDIDATE_DIRS:
        if not d.exists():
            print("skip dir:", d)
            continue
        for p in d.rglob("*.csv"):
            changed, msg = ensure_close(p)
            print(msg)
            if changed:
                total_changed += 1
    print("done. files changed:", total_changed)
