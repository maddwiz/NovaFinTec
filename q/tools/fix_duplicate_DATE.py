#!/usr/bin/env python3
# tools/fix_duplicate_DATE.py
# Normalizes every CSV in runs_plus/ to have exactly one datetime 'DATE' column.
# - If both 'Date' and 'DATE' exist, keep a single 'DATE' and drop 'Date'.
# - If multiple 'DATE' columns exist (duplicate headers), keep the FIRST, drop the rest.
# - Coerce 'DATE' to datetime and sort.

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"

def normalize_one(p: Path) -> str:
    try:
        df = pd.read_csv(p)
    except Exception as e:
        return f"skip: {p.name} (unreadable: {e})"

    cols = list(df.columns)

    # 1) If both 'Date' and 'DATE' exist: prefer 'DATE'. If 'DATE' missing but 'Date' present, promote.
    has_Date = "Date" in cols
    has_DATE = "DATE" in cols

    if has_Date and not has_DATE:
        df["DATE"] = df["Date"]
        has_DATE = True

    # 2) Drop duplicate 'DATE' columns beyond the first
    if has_DATE:
        idxs = [i for i, c in enumerate(df.columns) if c == "DATE"]
        # If there are multiple DATE columns, keep the first
        if len(idxs) > 1:
            # Build a new frame with unique DATE
            keep = idxs[0]
            drop_idxs = [i for i in idxs[1:]]
            # Drop by position:
            df = df.drop(df.columns[drop_idxs], axis=1)

    # 3) Drop any lingering 'Date' (lowercase D) if 'DATE' now exists
    if "DATE" in df.columns and "Date" in df.columns:
        df = df.drop(columns=["Date"])

    # 4) If neither exists, try to find a date-like column to promote
    if "DATE" not in df.columns:
        lower = {c.lower(): c for c in df.columns}
        candidate = None
        for key in ("date","timestamp","dt","time"):
            if key in lower:
                candidate = lower[key]
                break
        if candidate is not None:
            df["DATE"] = df[candidate]
        else:
            return f"warn: {p.name} (no DATE/Date found)"

    # 5) Coerce to datetime (best effort) and sort
    try:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)
    except Exception as e:
        return f"warn: {p.name} (failed to coerce DATE: {e})"

    try:
        df.to_csv(p, index=False)
    except Exception as e:
        return f"error: {p.name} (save failed: {e})"

    return f"fixed: {p.name}"

if __name__ == "__main__":
    if not RUNS.exists():
        raise SystemExit("No runs_plus/ directory found.")
    changed = 0
    for p in RUNS.glob("*.csv"):
        msg = normalize_one(p)
        print(msg)
        if msg.startswith("fixed"):
            changed += 1
    print("done. files changed:", changed)
