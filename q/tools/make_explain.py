#!/usr/bin/env python3
# tools/make_explain.py
# Builds a simple, human-readable explanation of Q's current stance.

from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"

def _load_csv(p, date_col="DATE"):
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, parse_dates=[date_col])
    except Exception:
        return pd.read_csv(p)

def _latest(df, col):
    if df.empty or col not in df.columns:
        return None
    return df[col].iloc[-1]

def run_make_explain():
    RUNS.mkdir(parents=True, exist_ok=True)

    # Try loading final portfolio summary
    try:
        finalj = json.loads((RUNS/"final_portfolio_summary.json").read_text())
    except Exception:
        finalj = {}

    # Regime summary
    try:
        regimej = json.loads((RUNS/"regime_summary.json").read_text())
    except Exception:
        regimej = {}

    sym  = _load_csv(RUNS/"symbolic_signal.csv")
    refl = _load_csv(RUNS/"reflexive_signal.csv")
    regw = _load_csv(RUNS/"regime_weights.csv")

    # Pull latest values
    cur_reg = (regimej.get("current") or {}).get("regime")
    cur_wts = (regimej.get("current") or {}).get("suggested_weights") or {}

    sym_v = None
    if not sym.empty:
        if "ASSET" in sym.columns and sym["ASSET"].nunique() > 1:
            sym_v = sym.groupby("DATE")["sym_signal"].mean().iloc[-1]
        elif "sym_signal" in sym.columns:
            sym_v = _latest(sym, "sym_signal")

    refl_v = None
    if not refl.empty:
        if "ASSET" in refl.columns and refl["ASSET"].nunique() > 1:
            refl_v = refl.groupby("DATE")["reflexive_signal"].mean().iloc[-1]
        elif "reflexive_signal" in refl.columns:
            refl_v = _latest(refl, "reflexive_signal")

    # Final portfolio weights if present
    fin_wts = (finalj.get("weights") or {})

    # Build narrative
    bullets = []
    if cur_reg:
        bullets.append(f"Regime detector classifies the market as **{cur_reg}**.")
    if cur_wts:
        txt = ", ".join([f"{k.replace('w_','')}: {cur_wts.get(k,0):.2f}" for k in ["w_main","w_vol","w_osc","w_sym","w_reflex"] if k in cur_wts])
        bullets.append(f"Governor suggests sleeve mix → {txt} (observer mode; not applied).")
    if sym_v is not None:
        bullets.append(f"Symbolic sentiment (avg) is **{sym_v:+.2f}** (tanh-z).")
    if refl_v is not None:
        bullets.append(f"Reflexive signal (avg) is **{refl_v:+.2f}** (tanh-z).")
    if fin_wts:
        txt2 = ", ".join([f"{k.replace('w_','')}: {fin_wts.get(k,0):.2f}" for k in fin_wts])
        bullets.append(f"Final portfolio weights (now) → {txt2}")

    explain = {
        "date": (regw["DATE"].iloc[-1].strftime("%Y-%m-%d") if not regw.empty else None),
        "bullets": bullets,
        "provenance": {
            "used_files": [
                "runs_plus/final_portfolio_summary.json",
                "runs_plus/regime_summary.json",
                "runs_plus/symbolic_signal.csv",
                "runs_plus/reflexive_signal.csv",
                "runs_plus/regime_weights.csv"
            ]
        }
    }
    (RUNS/"explain_latest.json").write_text(json.dumps(explain, indent=2))
    return explain

if __name__ == "__main__":
    ex = run_make_explain()
    print("✅ Wrote runs_plus/explain_latest.json")
    print("Bullets:")
    for b in ex.get("bullets", []):
        print(" -", b)
