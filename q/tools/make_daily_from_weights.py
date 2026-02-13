#!/usr/bin/env python3
# Builds runs_plus/daily_returns.csv by multiplying weights × asset_returns
# Chooses best available weights automatically.

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"; RUNS.mkdir(exist_ok=True)

def load_mat(rel):
    p = ROOT/rel
    if not p.exists(): return None
    try:
        a = np.loadtxt(p, delimiter=",")
    except:
        a = np.loadtxt(p, delimiter=",", skiprows=1)
    if a.ndim == 1: a = a.reshape(-1,1)
    return a

def first_mat(paths):
    for rel in paths:
        a = load_mat(rel)
        if a is not None: return a, rel
    return None, None

if __name__ == "__main__":
    A = load_mat("runs_plus/asset_returns.csv")
    if A is None:
        print("(!) runs_plus/asset_returns.csv missing. Run tools/rebuild_asset_matrix.py first.")
        raise SystemExit(0)

    W, src = first_mat([
        "runs_plus/portfolio_weights_final.csv",
        "runs_plus/tune_best_weights.csv",
        "runs_plus/weights_regime.csv",
        "runs_plus/weights_tail_blend.csv",
        "runs_plus/portfolio_weights.csv",
        "portfolio_weights.csv",
    ])
    if W is None:
        print("(!) No weights found."); raise SystemExit(0)

    # Align T and N
    T = min(A.shape[0], W.shape[0])
    if A.shape[1] != W.shape[1]:
        print(f"(!) Col mismatch: asset_returns N={A.shape[1]} vs weights N={W.shape[1]}.")
        raise SystemExit(0)

    pnl = np.sum(W[:T] * A[:T], axis=1)
    np.savetxt(RUNS/"daily_returns.csv", pnl, delimiter=",")
    print(f"✅ Wrote runs_plus/daily_returns.csv (T={T}) from weights='{src}'")
