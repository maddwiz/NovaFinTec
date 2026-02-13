#!/usr/bin/env python3
# tools/apply_best_osc_from_sweep.py
# Reads runs_plus/osc_cost_sweep.csv, picks best blend_sharpe,
# re-runs the costed oscillator and blend using those knobs,
# updates report cards, and prints a tiny summary.

from pathlib import Path
import os, json, subprocess, shlex, sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
CSV  = RUNS / "osc_cost_sweep.csv"

def run(cmd, env=None):
    res = subprocess.run(shlex.split(cmd), cwd=ROOT, env=env or os.environ.copy(),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    if res.returncode != 0:
        raise SystemExit(f"FAILED: {cmd}")

def sharpe_of_csv(p: Path, ret_cols=("ret_net","ret","return","ret_gross","pnl","daily_ret"),
                  eq_cols=("eq_net","eq","equity","equity_curve")):
    if not p.exists(): return None
    df = pd.read_csv(p)
    lowers={c.lower():c for c in df.columns}
    # Try returns first
    for c in ret_cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(0.0)
            if s.notna().sum()>10:
                mu = s.mean(); sd = s.std()
                return 0.0 if sd==0 or np.isnan(sd) else float((mu/sd)*np.sqrt(252))
    # Else derive from equity
    for c in eq_cols:
        if c in df.columns:
            eq = pd.to_numeric(df[c], errors="coerce")
            ret = eq.pct_change().replace([np.inf,-np.inf], np.nan).fillna(0.0)
            sd = ret.std(); 
            return 0.0 if sd==0 or np.isnan(sd) else float((ret.mean()/sd)*np.sqrt(252))
    return None

if __name__ == "__main__":
    if not CSV.exists():
        raise SystemExit("Missing runs_plus/osc_cost_sweep.csv — run tools/sweep_osc_cost_grid.py first.")

    df = pd.read_csv(CSV)
    df = df[df["blend_sharpe"].notna()]
    if df.empty:
        raise SystemExit("No valid rows in osc_cost_sweep.csv")

    best = df.sort_values("blend_sharpe", ascending=False).iloc[0]
    cost_bps = float(best["cost_bps"])
    max_dpos = float(best["max_dpos"])
    print(f"Using best from sweep: COST_BPS={cost_bps} bps, MAX_DPOS={max_dpos} (blend_sharpe={best['blend_sharpe']:.3f})")

    env = os.environ.copy()
    env["OSC_COST_BPS"] = str(cost_bps)
    env["OSC_MAX_DPOS"] = str(max_dpos)

    # Rebuild oscillator (costed) and cards
    run("python tools/run_osc_portfolio_costed.py", env)
    run("python tools/add_osc_cost_card.py", env)

    # Blend with Main (NET osc) and card
    run("python tools/blend_main_with_osc_costed.py", env)
    run("python tools/add_blend_osc_cost_card.py", env)

    # Print quick sanity and main Sharpe
    main_sh = sharpe_of_csv(RUNS/"portfolio_plus.csv")
    blend_meta = json.loads((RUNS/"blend_main_osc_costed_summary.json").read_text())
    print("\nSUMMARY")
    print("=======")
    print(f"Main Sharpe (approx): {main_sh if main_sh is not None else 'n/a'}")
    print(f"Blend alpha: {blend_meta.get('alpha'):.2f} | Blend Sharpe: {blend_meta.get('sharpe'):.3f}")
    print("Updated report_best_plus.html and report_all.html")
