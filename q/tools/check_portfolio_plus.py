#!/usr/bin/env python3
# tools/check_portfolio_plus.py
# Prints columns, sample rows, detected date/return columns, and basic stats.

from pathlib import Path
import pandas as pd
import numpy as np

RUNS = Path(__file__).resolve().parents[1] / "runs_plus"
P = RUNS / "portfolio_plus.csv"

def safe_series(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)

if __name__ == "__main__":
    if not P.exists():
        raise SystemExit("Missing runs_plus/portfolio_plus.csv")

    df = pd.read_csv(P)
    print("COLUMNS:", list(df.columns))
    print("SHAPE:", df.shape)

    # Head / Tail
    try:
        print("\nHEAD:\n", df.head(5).to_string(index=False))
        print("\nTAIL:\n", df.tail(5).to_string(index=False))
    except Exception as e:
        print("\n(head/tail print error)", e)

    # Detect date column
    lowers = {c.lower(): c for c in df.columns}
    date_col = (
        lowers.get("date")
        or lowers.get("timestamp")
        or lowers.get("time")
        or df.columns[0]
    )
    print("\nDetected date_col:", date_col)

    # Detect return column
    ret_candidates = [
        "ret_net","ret","ret_plus","ret_gross","return",
        "pnl","pnl_plus","daily_ret","portfolio_ret","port_ret"
    ]
    ret_col = None
    for c in ret_candidates:
        if c in df.columns:
            ret_col = c
            break

    if ret_col is not None:
        s = safe_series(df[ret_col]).fillna(0.0)
        print("Detected return_col:", ret_col)
        try:
            desc = s.describe(percentiles=[0.01,0.05,0.25,0.5,0.75,0.95,0.99])
            print("\nReturn summary:\n", desc.to_string())
        except Exception as e:
            print("\n(summary error)", e)
    else:
        print("Detected return_col: NONE (will fall back to equity pct-change)")
        # Try equity-based fallback summary
        eq_candidates = [
            "eq_net","eq","equity","equity_curve","portfolio_eq","equity_index","port_equity"
        ]
        eq_col = None
        for c in eq_candidates:
            if c in df.columns:
                eq_col = c
                break
        if eq_col is not None:
            eq = safe_series(df[eq_col])
            r = eq.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
            try:
                desc = r.describe(percentiles=[0.01,0.05,0.25,0.5,0.75,0.95,0.99])
                print(f"\nFallback returns from {eq_col} pct-change summary:\n", desc.to_string())
            except Exception as e:
                print("\n(fallback summary error)", e)
        else:
            print("No equity column found either; blender will be unable to compute returns.")
