#!/usr/bin/env python3
# tools/apply_sweep_best.py
# Reads runs_plus/sweep_regime_small.csv (top row), rewrites:
#  - tools/build_min_sleeves.py -> VOL_SCALE, OSC_SCALE
#  - tools/patch_regime_weights_with_dna.py -> STRENGTH, ALPHA_MIN
# Then rebuilds sleeves, regime, regime+DNA, and updates the report.

from pathlib import Path
import pandas as pd, re, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
SWEEP = RUNS / "sweep_regime_small.csv"

BUILD_SLEEVES = ROOT / "tools" / "build_min_sleeves.py"
PATCH_DNA     = ROOT / "tools" / "patch_regime_weights_with_dna.py"
MAKE_REGIME   = ROOT / "tools" / "make_regime.py"
APPLY_REG     = ROOT / "tools" / "apply_regime_governor.py"
APPLY_DNA     = ROOT / "tools" / "apply_regime_governor_dna.py"
ADD_CARD      = ROOT / "tools" / "add_regime_final_card_triple.py"

def run(cmd):
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(r.returncode)

if __name__ == "__main__":
    if not SWEEP.exists():
        sys.exit("Missing runs_plus/sweep_regime_small.csv (run sweep first).")
    df = pd.read_csv(SWEEP)
    if df.empty:
        sys.exit("sweep_regime_small.csv is empty.")
    best = df.iloc[0]

    vol = float(best["vol"])
    osc = float(best["osc"])
    strength = float(best["dna_strength"])
    alpha_min = float(best["alpha_min"]) if "alpha_min" in best.index else 0.60

    # rewrite build_min_sleeves.py (VOL/OSC)
    txt = BUILD_SLEEVES.read_text()
    txt = re.sub(r"(VOL_SCALE\s*=\s*)[0-9.]+", rf"\g<1>{vol}", txt)
    txt = re.sub(r"(OSC_SCALE\s*=\s*)[0-9.]+", rf"\g<1>{osc}", txt)
    BUILD_SLEEVES.write_text(txt)

    # rewrite patch_regime_weights_with_dna.py (STRENGTH/ALPHA_MIN)
    txt = PATCH_DNA.read_text()
    txt = re.sub(r"(STRENGTH\s*=\s*)[0-9.]+", rf"\g<1>{strength}", txt)
    txt = re.sub(r"(ALPHA_MIN\s*=\s*)[0-9.]+", rf"\g<1>{alpha_min}", txt)
    PATCH_DNA.write_text(txt)

    print(f"✅ Applied best sweep: VOL_SCALE={vol} OSC_SCALE={osc} DNA_STRENGTH={strength} ALPHA_MIN={alpha_min}")

    # rebuild sleeves + regime + dna + card
    run(["python", str(BUILD_SLEEVES)])
    run(["python", str(MAKE_REGIME)])
    run(["python", str(APPLY_REG)])
    run(["python", str(PATCH_DNA)])
    run(["python", str(APPLY_DNA)])
    run(["python", str(ADD_CARD)])
    print("✅ Rebuilt regime + regime+DNA and updated report card.")
