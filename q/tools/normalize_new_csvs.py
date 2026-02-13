#!/usr/bin/env python3
"""
normalize_new_csvs.py  (robust)

Goal:
- Make EVERY file in ./data/*.csv look like two columns:
    DATE,VALUE
- Handles:
  * FRED downloads (DATE,VALUE)
  * Stooq OHLCV (Date,Open,High,Low,Close,Volume) -> Close
  * Mixed/odd CSVs with extra columns, bad lines, weird headers

It reads with a safe fallback (engine='python', on_bad_lines='skip'),
then picks (date-like column, numeric value column) and overwrites in place.

Skips files that are clearly not timeseries (e.g., news.csv).
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# Heuristics
DATE_CANDIDATES = {"date","DATE","Date","timestamp","Timestamp"}
VALUE_FIRST_CHOICES = ["VALUE","value","Close","close","Adj Close","AdjClose","Price","price","PX_LAST","CLOSE"]

NON_TS_FILES = {"news.csv"}  # leave alone

def safe_read(p: Path):
    # try fast C engine first
    try:
        return pd.read_csv(p)
    except Exception:
        # robust fallback
        try:
            return pd.read_csv(p, engine="python", on_bad_lines="skip")
        except Exception:
            return None

def choose_date_col(cols):
    # pick first date-like column
    for c in cols:
        if c in DATE_CANDIDATES:
            return c
    # try anything named like date
    for c in cols:
        if "date" in c.lower() or "time" in c.lower():
            return c
    # last resort: first column
    return cols[0] if cols else None

def choose_value_col(df):
    cols = list(df.columns)
    # prefer known price/value names
    for name in VALUE_FIRST_CHOICES:
        if name in cols:
            return name
        # case-insensitive match
        for c in cols:
            if c.lower() == name.lower():
                return c
    # otherwise pick the first numeric-ish column that is NOT the date col
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(10, int(0.2*len(s))):  # has some numbers
            return c
    return None

def is_timeseries(df):
    if df is None or df.empty:
        return False
    # needs at least one column that can be parsed as dates
    for c in df.columns:
        d = pd.to_datetime(df[c], errors="coerce")
        if d.notna().sum() >= max(10, int(0.2*len(d))):
            return True
    return False

def main():
    if not DATA.exists():
        print("data/ not found")
        return
    changed = 0
    for p in sorted(DATA.glob("*.csv")):
        if p.name in NON_TS_FILES:
            print(f"left {p.name} unchanged (non timeseries)")
            continue

        df = safe_read(p)
        if df is None:
            print(f"skip {p.name}: unreadable even with fallback")
            continue

        # Already perfect FRED?
        up = [c.upper() for c in df.columns]
        if "DATE" in up and "VALUE" in up and len(df.columns) == 2:
            # small cleanup: ensure parseable & numeric
            dcol = df.columns[up.index("DATE")]
            vcol = df.columns[up.index("VALUE")]
            out = pd.DataFrame({
                "DATE": pd.to_datetime(df[dcol], errors="coerce"),
                "VALUE": pd.to_numeric(df[vcol], errors="coerce"),
            }).dropna(subset=["DATE","VALUE"]).sort_values("DATE")
            out.to_csv(p, index=False)
            continue

        if not is_timeseries(df):
            print(f"left {p.name} unchanged (no clear dates)")
            continue

        # Pick date + value columns
        dcol = choose_date_col(list(df.columns))
        if dcol is None:
            print(f"left {p.name} unchanged (no date column)")
            continue

        # Normalize column names for easier matching
        # (Handle Stooq casing: Date, Open, High, Low, Close)
        df.columns = [str(c) for c in df.columns]

        vcol = None
        # If this looks like Stooq OHLCV, prefer Close
        stooq_like = set(c.lower() for c in df.columns) >= {"date","open","high","low","close"}
        if stooq_like:
            vcol = "Close" if "Close" in df.columns else "close"

        if vcol is None:
            vcol = choose_value_col(df)

        if vcol is None:
            print(f"left {p.name} unchanged (no numeric value column)")
            continue

        out = pd.DataFrame({
            "DATE": pd.to_datetime(df[dcol], errors="coerce"),
            "VALUE": pd.to_numeric(df[vcol], errors="coerce"),
        }).dropna(subset=["DATE","VALUE"]).sort_values("DATE")

        if out.empty:
            print(f"left {p.name} unchanged (no valid rows after cleaning)")
            continue

        out.to_csv(p, index=False)
        print(f"converted {p.name} -> DATE,VALUE (from {dcol}, {vcol})")
        changed += 1

    print(f"\n✅ normalization done. files changed: {changed}")

if __name__ == "__main__":
    main()
