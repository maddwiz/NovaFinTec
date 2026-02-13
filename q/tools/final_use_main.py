#!/usr/bin/env python3
# tools/final_use_main.py
# Simple, safe reset: use Main (portfolio_plus.csv) as the FINAL portfolio.
# Writes:
#   runs_plus/final_portfolio.csv
#   runs_plus/final_portfolio_summary.json
# and updates the HTML card via tools/add_final_card.py

from pathlib import Path
import pandas as pd, numpy as np, json

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
P_MAIN = RUNS / "portfolio_plus.csv"

def safe(s):
    return pd.Series(pd.to_numeric(s, errors="coerce")).replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(-0.5,0.5)

def ann_sharpe(r):
    r = pd.Series(r).replace([np.inf,-np.inf], np.nan).dropna()
    s = r.std()
    if s == 0 or np.isnan(s): return 0.0
    return float((r.mean()/s)*(252.0**0.5))

def dd_min(r):
    eq = (1.0 + r).cumprod()
    peak = pd.concat([eq, pd.Series(1.0, index=eq.index)], axis=1).max(axis=1).cummax()
    dd = (eq/peak - 1.0)
    return float(dd.min())

if __name__ == "__main__":
    if not P_MAIN.exists():
        raise SystemExit("Missing runs_plus/portfolio_plus.csv")

    df = pd.read_csv(P_MAIN)
    lowers = {c.lower(): c for c in df.columns}
    dcol = lowers.get("date") or lowers.get("timestamp") or lowers.get("time") or df.columns[0]
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).sort_values(dcol)

    # Detect returns or derive from equity
    r = None
    for c in ["ret_net","ret","ret_plus","ret_gross","return","pnl","pnl_plus","daily_ret","portfolio_ret","port_ret"]:
        if c in df.columns:
            r = safe(df[c]); break
    if r is None:
        for c in ["eq_net","eq","equity","equity_curve","portfolio_eq","equity_index","port_equity"]:
            if c in df.columns:
                eq = pd.to_numeric(df[c], errors="coerce")
                r = safe(eq.pct_change()); break
    if r is None:
        raise SystemExit("Could not find returns in portfolio_plus.csv")

    # Build final CSV (Main only)
    out = pd.DataFrame({"DATE": df[dcol], "ret": r})
    out["eq"] = (1.0 + out["ret"]).cumprod()
    out["w_main"] = 1.0
    out["w_vol"] = 0.0
    out["w_osc"] = 0.0
    out.to_csv(RUNS/"final_portfolio.csv", index=False)

    # 75/25 split just for IS/OOS display
    n = len(out)
    split_idx = int(n*0.75)
    r_is  = out["ret"].iloc[:split_idx]
    r_oos = out["ret"].iloc[split_idx:]

    summary = {
        "weights": {"w_main": 1.0, "w_vol": 0.0, "w_osc": 0.0},
        "in_sample":  {"sharpe": ann_sharpe(r_is),  "hit": float((r_is>0).mean()),  "maxdd": dd_min(r_is)},
        "out_sample": {"sharpe": ann_sharpe(r_oos), "hit": float((r_oos>0).mean()), "maxdd": dd_min(r_oos)},
        "sleeve_sharpes_test": {"main": ann_sharpe(r_oos), "vol_net": 0.0, "osc_net": 0.0},
        "split": {"train_until": str(out['DATE'].iloc[split_idx].date()),
                  "n_train": int(split_idx), "n_test": int(n - split_idx)},
        "guard": "MAIN_ONLY (manual reset)"
    }
    (RUNS/"final_portfolio_summary.json").write_text(json.dumps(summary, indent=2))

    print("✅ Final reset to Main-only.")
    print("Saved: runs_plus/final_portfolio.csv and final_portfolio_summary.json")
