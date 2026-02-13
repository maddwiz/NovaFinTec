#!/usr/bin/env python3
# Trims common files in runs_plus/ to the same T as asset_returns.csv
# Writes *_aligned.csv next to originals (non-destructive).

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"; RUNS.mkdir(exist_ok=True)

FILES_SERIES = [
    "daily_returns.csv",
    "meta_stack_pred.csv",
    "synapses_pred.csv",
    "meta_mix.csv",
    "dna_drift.csv",
    "heartbeat_bpm.csv",
    "symbolic_latent.csv",
    "reflex_latent.csv",
    "shock_mask.csv",
    "legacy_exposure.csv",
]

FILES_MATRIX = [
    "portfolio_weights.csv",
    "weights_tail_blend.csv",
    "risk_parity_weights.csv",
    "weights_capped.csv",
    "weights_cluster_capped.csv",
    "weights_regime.csv",
    "tune_best_weights.csv",
    "portfolio_weights_final.csv",
    "asset_returns.csv",   # reference
]

def load_series(p: Path):
    try:
        return np.loadtxt(p, delimiter=",").ravel()
    except:
        try:
            return np.loadtxt(p, delimiter=",", skiprows=1).ravel()
        except:
            return None

def load_matrix(p: Path):
    try:
        a = np.loadtxt(p, delimiter=",")
    except:
        try:
            a = np.loadtxt(p, delimiter=",", skiprows=1)
        except:
            return None
    if a.ndim == 1: a = a.reshape(-1,1)
    return a

if __name__ == "__main__":
    ref = load_matrix(RUNS/"asset_returns.csv")
    if ref is None:
        print("(!) runs_plus/asset_returns.csv missing. Run tools/rebuild_asset_matrix.py first.")
        raise SystemExit(0)
    T = ref.shape[0]

    # series
    for name in FILES_SERIES:
        p = RUNS/name
        if not p.exists(): continue
        s = load_series(p)
        if s is None: continue
        s2 = s[:T]
        np.savetxt(RUNS/(name.replace(".csv","_aligned.csv")), s2, delimiter=",")
    # matrices
    for name in FILES_MATRIX:
        p = RUNS/name
        if not p.exists(): continue
        M = load_matrix(p)
        if M is None: continue
        M2 = M[:T]
        np.savetxt(RUNS/(name.replace(".csv","_aligned.csv")), M2, delimiter=",")

    print(f"✅ Wrote *_aligned.csv (T={T}) for common runs_plus/ files.")
