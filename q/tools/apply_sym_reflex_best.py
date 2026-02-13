#!/usr/bin/env python3
# tools/apply_sym_reflex_best.py
# Reads runs_plus/sweep_sym_reflex.csv (top row), rewrites SCALE_SYM/SCALE_REFLEX
# in both governors, then rebuilds and updates the report card.

from pathlib import Path
import re, subprocess, sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
SWEEP = RUNS / "sweep_sym_reflex.csv"

GOV  = ROOT / "tools" / "apply_regime_governor.py"
GOVD = ROOT / "tools" / "apply_regime_governor_dna.py"
ADD  = ROOT / "tools" / "add_regime_final_card_triple.py"

def run(cmd):
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(r.returncode)

def rewrite_scales(sym_scale, rfx_scale):
    for p in (GOV, GOVD):
        txt = p.read_text()
        txt = re.sub(r"(SCALE_SYM\s*=\s*)[0-9.]+", rf"\g<1>{sym_scale}", txt)
        txt = re.sub(r"(SCALE_REFLEX\s*=\s*)[0-9.]+", rf"\g<1>{rfx_scale}", txt)
        p.write_text(txt)

if __name__ == "__main__":
    if not SWEEP.exists():
        sys.exit("Missing runs_plus/sweep_sym_reflex.csv (run tools/sweep_sym_reflex.py first).")
    df = pd.read_csv(SWEEP)
    if df.empty:
        sys.exit("sweep_sym_reflex.csv is empty.")
    best = df.iloc[0]
    sym = float(best["sym_scale"])
    rfx = float(best["reflex_scale"])
    rewrite_scales(sym, rfx)
    print(f"✅ Applied Symbolic scale={sym}, Reflexive scale={rfx}")

    run(["python", "tools/make_regime.py"])
    run(["python", "tools/apply_regime_governor.py"])
    run(["python", "tools/patch_regime_weights_with_dna.py"])
    run(["python", "tools/apply_regime_governor_dna.py"])
    run(["python", "tools/add_regime_final_card_triple.py"])
    print("✅ Rebuilt regime + regime+DNA and updated report.")
