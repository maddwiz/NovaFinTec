#!/usr/bin/env python3
# tools/fix_dates_in_runs_plus.py
# Adds a 'Date' column when only 'DATE'/'date'/'timestamp' exists in runs_plus/*.csv

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"

def pick_date_col(cols):
    lower = {c.lower(): c for c in cols}
    if "date" in lower: return lower["date"]
    if "timestamp" in lower: return lower["timestamp"]
    for c in cols:
        if c.lower() in ("date","timestamp","dt","time"):
            return c
    return None

def main():
    if not RUNS.exists():
        print("no runs_plus/")
        return
    changed = 0
    for p in RUNS.glob("*.csv"):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        cols = list(df.columns)
        if "Date" in cols:
            continue
        dcol = pick_date_col(cols)
        if not dcol:
            continue
        df["Date"] = df[dcol]
        try:
            df.to_csv(p, index=False)
            changed += 1
            print("fixed:", p.name, "added Date from", dcol)
        except Exception as e:
            print("skip:", p.name, e)
    print("done. files changed:", changed)

if __name__ == "__main__":
    main()
