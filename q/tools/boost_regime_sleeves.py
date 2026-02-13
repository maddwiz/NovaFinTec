#!/usr/bin/env python3
# tools/boost_regime_sleeves.py
# Bumps sleeve weights in runs_plus/regime_weights.csv by a small factor, caps, renormalizes.
# Creates a backup: runs_plus/regime_weights.backup.csv

from pathlib import Path
import pandas as pd
import shutil

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
SRC  = RUNS / "regime_weights.csv"
BAK  = RUNS / "regime_weights.backup.csv"

BOOST = 1.15    # +15% sleeves
CAP   = 0.65    # max total non-main (vol+osc+sym+reflex) after boost

if __name__ == "__main__":
    if not SRC.exists():
        raise SystemExit("Missing runs_plus/regime_weights.csv (run tools/make_regime.py first).")
    shutil.copyfile(SRC, BAK)
    df = pd.read_csv(SRC)
    for c in ["w_vol","w_osc","w_sym","w_reflex"]:
        if c not in df.columns:
            df[c] = 0.0

    # boost sleeves
    df["w_vol"]    = df["w_vol"]    * BOOST
    df["w_osc"]    = df["w_osc"]    * BOOST
    df["w_sym"]    = df["w_sym"]    * BOOST
    df["w_reflex"] = df["w_reflex"] * BOOST

    # cap combined sleeves
    sleeves = df[["w_vol","w_osc","w_sym","w_reflex"]].sum(axis=1)
    over = (sleeves > CAP)
    if over.any():
        scale = CAP / sleeves
        scale = scale.where(~over, other=scale)  # only scale where over
        for c in ["w_vol","w_osc","w_sym","w_reflex"]:
            df[c] = df[c] * scale.fillna(1.0)

    # reallocate to main so rows sum to 1
    df["w_main"] = 1.0 - (df["w_vol"] + df["w_osc"] + df["w_sym"] + df["w_reflex"])
    df.to_csv(SRC, index=False)
    print("✅ Updated regime_weights.csv (backup at regime_weights.backup.csv)")
