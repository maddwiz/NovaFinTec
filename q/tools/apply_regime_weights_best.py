#!/usr/bin/env python3
# tools/apply_regime_weights_best.py
# Reads runs_plus/sweep_regime_weights.csv (top row), multiplies sleeves in
# runs_plus/regime_weights.csv by mv/mo/ms/mr, caps sleeves, preserves MAIN_MIN,
# rebuilds Regime + Regime+DNA, and updates the triple card.

from pathlib import Path
import pandas as pd
import numpy as np
import subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
WCSV = RUNS / "regime_weights.csv"
SWEEP = RUNS / "sweep_regime_weights.csv"

APPLY_REG = ROOT/"tools"/"apply_regime_governor.py"
PATCH_DNA = ROOT/"tools"/"patch_regime_weights_with_dna.py"
APPLY_DNA = ROOT/"tools"/"apply_regime_governor_dna.py"
ADD_CARD  = ROOT/"tools"/"add_regime_final_card_triple.py"

SLEEVES_CAP = 0.65
MAIN_MIN    = 0.25

def run(cmd):
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(r.returncode)

def adjust_weights(df, mv, mo, ms, mr):
    out = df.copy()
    for c in ["w_main","w_vol","w_osc","w_sym","w_reflex"]:
        if c not in out.columns:
            out[c] = 0.0
    # multiply sleeves
    out["w_vol"]    = out["w_vol"]    * mv
    out["w_osc"]    = out["w_osc"]    * mo
    out["w_sym"]    = out["w_sym"]    * ms
    out["w_reflex"] = out["w_reflex"] * mr

    # cap sleeves total per row
    sleeves = out[["w_vol","w_osc","w_sym","w_reflex"]].sum(axis=1)
    over = sleeves > SLEEVES_CAP
    scale = np.where(over, SLEEVES_CAP / sleeves.replace(0, np.nan), 1.0)
    for c in ["w_vol","w_osc","w_sym","w_reflex"]:
        out[c] = out[c] * scale

    # keep some main everywhere
    out["w_main"] = 1.0 - (out["w_vol"] + out["w_osc"] + out["w_sym"] + out["w_reflex"])
    need = out["w_main"] < MAIN_MIN
    if need.any():
        deficit = (MAIN_MIN - out["w_main"]).clip(lower=0.0)
        sleeves_sum = out[["w_vol","w_osc","w_sym","w_reflex"]].sum(axis=1).replace(0, np.nan)
        shrink = deficit / sleeves_sum
        for c in ["w_vol","w_osc","w_sym","w_reflex"]:
            out.loc[need, c] = out.loc[need, c] * (1.0 - shrink.loc[need].fillna(0.0))
        out["w_main"] = 1.0 - (out["w_vol"] + out["w_osc"] + out["w_sym"] + out["w_reflex"])
    return out.fillna(0.0)

if __name__ == "__main__":
    if not SWEEP.exists():
        sys.exit("Missing runs_plus/sweep_regime_weights.csv (run tools/sweep_regime_weights.py first).")
    if not WCSV.exists():
        sys.exit("Missing runs_plus/regime_weights.csv (run tools/make_regime.py first).")

    best = pd.read_csv(SWEEP).iloc[0]
    mv, mo, ms, mr = float(best["mv"]), float(best["mo"]), float(best["ms"]), float(best["mr"])

    base = pd.read_csv(WCSV)
    tweaked = adjust_weights(base, mv, mo, ms, mr)
    tweaked.to_csv(WCSV, index=False)
    print(f"✅ Applied multipliers: vol×{mv} osc×{mo} sym×{ms} reflex×{mr}")

    run(["python", "tools/apply_regime_governor.py"])
    run(["python", "tools/patch_regime_weights_with_dna.py"])
    run(["python", "tools/apply_regime_governor_dna.py"])
    run(["python", "tools/add_regime_final_card_triple.py"])
    print("✅ Rebuilt regime + regime+DNA and updated report.")
