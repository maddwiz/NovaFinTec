#!/usr/bin/env python3
# tools/sweep_regime_weights.py
# Sweeps tiny multipliers on regime sleeve weights (vol/osc/sym/reflex),
# caps sleeves, renormalizes per-row, rebuilds Regime + Regime+DNA,
# and records OOS performance.
#
# Writes: runs_plus/sweep_regime_weights.csv

from pathlib import Path
import pandas as pd
import numpy as np
import json
import subprocess
import sys
import itertools

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
WCSV = RUNS / "regime_weights.csv"

MAKE_REGIME = ROOT / "tools" / "make_regime.py"              # not called in sweep (weights already exist)
APPLY_REG   = ROOT / "tools" / "apply_regime_governor.py"
PATCH_DNA   = ROOT / "tools" / "patch_regime_weights_with_dna.py"
APPLY_DNA   = ROOT / "tools" / "apply_regime_governor_dna.py"

# small, safe multipliers
VOL_M = [0.90, 1.00, 1.10]
OSC_M = [0.90, 1.00, 1.10]
SYM_M = [0.80, 1.00, 1.20]
RFX_M = [0.80, 1.00, 1.20]

SLEEVES_CAP = 0.65  # total (vol+osc+sym+reflex) max
MAIN_MIN    = 0.25  # keep some main everywhere

def run(cmd):
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(r.returncode)

def read_json(name):
    p = RUNS / name
    return json.loads(p.read_text()) if p.exists() else {}

def metric_or(d, path, default=np.nan):
    cur = d
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    try:
        return float(cur)
    except Exception:
        return default

def adjust_weights(df, mv, mo, ms, mr):
    out = df.copy()
    for c in ["w_main", "w_vol", "w_osc", "w_sym", "w_reflex"]:
        if c not in out.columns:
            out[c] = 0.0

    # multiply sleeves
    out["w_vol"]    = out["w_vol"]    * mv
    out["w_osc"]    = out["w_osc"]    * mo
    out["w_sym"]    = out["w_sym"]    * ms
    out["w_reflex"] = out["w_reflex"] * mr

    # cap sleeves total
    sleeves = out[["w_vol", "w_osc", "w_sym", "w_reflex"]].sum(axis=1)
    over = sleeves > SLEEVES_CAP
    scale = np.where(over, SLEEVES_CAP / sleeves.replace(0, np.nan), 1.0)
    for c in ["w_vol", "w_osc", "w_sym", "w_reflex"]:
        out[c] = out[c] * scale

    # set w_main as remainder, enforce MAIN_MIN
    out["w_main"] = 1.0 - (out["w_vol"] + out["w_osc"] + out["w_sym"] + out["w_reflex"])
    need = out["w_main"] < MAIN_MIN
    if need.any():
        deficit = (MAIN_MIN - out["w_main"]).clip(lower=0.0)
        sleeves_sum = out[["w_vol", "w_osc", "w_sym", "w_reflex"]].sum(axis=1).replace(0, np.nan)
        shrink = deficit / sleeves_sum
        for c in ["w_vol", "w_osc", "w_sym", "w_reflex"]:
            out.loc[need, c] = out.loc[need, c] * (1.0 - shrink.loc[need].fillna(0.0))
        out["w_main"] = 1.0 - (out["w_vol"] + out["w_osc"] + out["w_sym"] + out["w_reflex"])

    return out.fillna(0.0)

if __name__ == "__main__":
    if not WCSV.exists():
        sys.exit("Missing runs_plus/regime_weights.csv (run tools/make_regime.py first).")

    base = pd.read_csv(WCSV)
    original = base.copy()

    rows = []
    try:
        for mv, mo, ms, mr in itertools.product(VOL_M, OSC_M, SYM_M, RFX_M):
            tweaked = adjust_weights(base, mv, mo, ms, mr)
            tweaked.to_csv(WCSV, index=False)

            # rebuild governed portfolios
            run(["python", "tools/apply_regime_governor.py"])
            run(["python", "tools/patch_regime_weights_with_dna.py"])
            run(["python", "tools/apply_regime_governor_dna.py"])

            reg = read_json("final_portfolio_regime_summary.json")
            dna = read_json("final_portfolio_regime_dna_summary.json")

            rows.append({
                "mv": mv, "mo": mo, "ms": ms, "mr": mr,
                "reg_oos": metric_or(reg, ["out_sample", "sharpe"]),
                "dna_oos": metric_or(dna, ["out_sample", "sharpe"]),
                "reg_dd":  metric_or(reg, ["out_sample", "maxdd"]),
                "dna_dd":  metric_or(dna, ["out_sample", "maxdd"]),
            })

        df = pd.DataFrame(rows).sort_values(["dna_oos", "reg_oos"], ascending=False)
        outp = RUNS / "sweep_regime_weights.csv"
        df.to_csv(outp, index=False)
        print(df.head(15).to_string(index=False))
        print("Saved:", outp)
    finally:
        # always restore original weights file
        original.to_csv(WCSV, index=False)
