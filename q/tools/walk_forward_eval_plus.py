#!/usr/bin/env python3
"""
walk_forward_eval_plus.py

Wrapper around baseline walk_forward_eval that:
- runs the usual evaluation
- saves per-asset daily OOS returns into runs_plus/<asset>/oos.csv
"""

import sys, json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def max_drawdown(equity):
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())

def eval_one(asset: str, df: pd.DataFrame, out_dir: Path):
    """
    df must have columns: date, ret (raw returns), pos (position signal), price
    """
    df = df.copy()
    df["pnl"] = df["pos"].shift(1).fillna(0) * df["ret"]
    df["equity"] = (1.0 + df["pnl"]).cumprod()

    hit = (np.sign(df["pnl"]) == np.sign(df["ret"])).mean()
    sharpe = df["pnl"].mean() / (df["pnl"].std() + 1e-9) * np.sqrt(252)
    mdd = max_drawdown(df["equity"])

    out = {
        "asset": asset,
        "hit": float(hit),
        "sharpe": float(sharpe),
        "maxDD": float(mdd),
    }

    safe_mkdir(out_dir)
    # save summary
    (out_dir / "summary_plus.json").write_text(json.dumps(out, indent=2))
    # save daily oos
    df.to_csv(out_dir / "oos.csv", index=False)

    return out

def main():
    """
    Example driver.
    In practice, replace this with your pipeline’s loop over assets.
    """
    # Dummy example — replace with your real WF loop
    example_asset = "SPY"
    dates = pd.date_range("2020-01-01", periods=100)
    df = pd.DataFrame({
        "date": dates,
        "ret": np.random.normal(0, 0.01, size=len(dates)),
        "pos": np.random.choice([-1,0,1], size=len(dates)),
        "price": np.linspace(300, 350, len(dates))
    })
    out_dir = RUNS / example_asset
    out = eval_one(example_asset, df, out_dir)
    print("✅ Saved", out)

if __name__ == "__main__":
    main()
