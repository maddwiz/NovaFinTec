#!/usr/bin/env python3
"""
clean_data_folder.py

Goal:
- Scan ./data/*.csv
- Keep files that parse into a clean two-column time series:
    DATE,VALUE   (DATE is parseable datetime, VALUE numeric)
- If a file can't be coerced to that shape, move it to ./data_bad/
  so the pipeline won't crash on it (e.g., corrupted downloads / HTML).

Safe to run anytime before your pipeline.
"""

from pathlib import Path
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA_BAD = ROOT / "data_bad"

DATE_NAMES = {"DATE","Date","date","timestamp","Timestamp"}
VALUE_PREF = ["VALUE","value","Close","close","Adj Close","AdjClose","Price","price","PX_LAST","CLOSE"]

def robust_read(p: Path):
    try:
        return pd.read_csv(p)
    except Exception:
        try:
            return pd.read_csv(p, engine="python", on_bad_lines="skip")
        except Exception:
            return None

def pick_date_col(cols):
    for c in cols:
        if c in DATE_NAMES:
            return c
    for c in cols:
        if "date" in str(c).lower() or "time" in str(c).lower():
            return c
    return cols[0] if cols else None

def pick_value_col(df):
    cols = list(df.columns)
    for name in VALUE_PREF:
        if name in cols:
            return name
        for c in cols:
            if str(c).lower() == name.lower():
                return c
    # fallback: first numeric-ish column not equal to date
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(10, int(0.2*len(s))):
            return c
    return None

def to_two_col(df):
    up = [str(c).upper() for c in df.columns]
    if "DATE" in up and "VALUE" in up and len(df.columns) == 2:
        dcol = df.columns[up.index("DATE")]
        vcol = df.columns[up.index("VALUE")]
        out = pd.DataFrame({
            "DATE": pd.to_datetime(df[dcol], errors="coerce"),
            "VALUE": pd.to_numeric(df[vcol], errors="coerce"),
        }).dropna(subset=["DATE","VALUE"]).sort_values("DATE")
        return out if not out.empty else None

    dcol = pick_date_col(list(df.columns))
    if dcol is None:
        return None
    vcol = pick_value_col(df)
    if vcol is None:
        return None

    out = pd.DataFrame({
        "DATE": pd.to_datetime(df[dcol], errors="coerce"),
        "VALUE": pd.to_numeric(df[vcol], errors="coerce"),
    }).dropna(subset=["DATE","VALUE"]).sort_values("DATE")
    return out if not out.empty else None

def main():
    DATA_BAD.mkdir(exist_ok=True)
    moved = 0
    fixed = 0
    for p in sorted(DATA.glob("*.csv")):
        df = robust_read(p)
        if df is None or df.empty:
            shutil.move(p, DATA_BAD / p.name)
            print(f"🚫 moved unreadable -> data_bad/{p.name}")
            moved += 1
            continue
        out = to_two_col(df)
        if out is None:
            shutil.move(p, DATA_BAD / p.name)
            print(f"🚫 moved uncoercible -> data_bad/{p.name}")
            moved += 1
            continue
        # overwrite with clean two-column
        out.to_csv(p, index=False)
        fixed += 1
    print(f"\n✅ clean complete. fixed={fixed}, moved_bad={moved}, keep={fixed}")

if __name__ == "__main__":
    main()
