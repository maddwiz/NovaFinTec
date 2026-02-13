#!/usr/bin/env python3
# tools/fix_dates_in_data.py
# Ensures every CSV in data/ and data_new/ has a 'Date' column.
# If only 'DATE'/'date'/'timestamp' exists, it copies it into 'Date' (no other changes).

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIRS = [ROOT/"data", ROOT/"data_new"]

def pick_date_col(cols):
    lower = {c.lower(): c for c in cols}
    if "date" in lower: return lower["date"]
    if "timestamp" in lower: return lower["timestamp"]
    for c in cols:
        if c.lower() in ("date","timestamp","dt","time"):
            return c
    return None

def fix_dir(d: Path):
    if not d.exists(): 
        print("skip dir (not found):", d)
        return 0
    changed = 0
    for p in d.rglob("*.csv"):
        try:
            df = pd.read_csv(p, nrows=5)
        except Exception:
            continue
        cols = list(df.columns)
        if "Date" in cols:
            continue
        dcol = pick_date_col(cols)
        if not dcol:
            continue
        try:
            df_full = pd.read_csv(p)
            df_full["Date"] = df_full[dcol]
            df_full.to_csv(p, index=False)
            changed += 1
            print("fixed:", p.relative_to(ROOT), "added Date from", dcol)
        except Exception as e:
            print("skip:", p.relative_to(ROOT), e)
    return changed

if __name__ == "__main__":
    total = 0
    for d in CANDIDATE_DIRS:
        total += fix_dir(d)
    print("done. files changed:", total)
