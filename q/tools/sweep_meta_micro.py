#!/usr/bin/env python3
# tools/sweep_meta_micro.py
# Fine sweep around regime sleeve multipliers (vol/osc/sym/reflex).
# Optimizes primarily for Vol-Target OOS Sharpe (vt_oos),
# tie-breakers: Regime+DNA OOS (dna_oos), Main OOS (main_oos).
#
# Writes: runs_plus/sweep_meta_micro.csv (best rows first).

from pathlib import Path
import pandas as pd, numpy as np, json, subprocess, sys, itertools, re

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
WCSV = RUNS / "regime_weights.csv"

APPLY_REG   = ROOT/"tools"/"apply_regime_governor.py"
PATCH_DNA   = ROOT/"tools"/"patch_regime_weights_with_dna.py"
APPLY_DNA   = ROOT/"tools"/"apply_regime_governor_dna.py"
PORTF_BUILDER = ROOT/"tools"/"portfolio_from_runs_plus.py"
PORTF_VT   = ROOT/"tools"/"portfolio_vol_target.py"

# --- knobs: center multipliers and offsets for fine grid ---
# You can adjust these sets wider/narrower if needed.
VOL_M = [0.95, 1.00, 1.05]
OSC_M = [0.85, 0.90, 0.95, 1.00]
SYM_M = [0.90, 1.00, 1.10, 1.20]
RFX_M = [0.90, 1.00, 1.10, 1.20]

SLEEVES_CAP = 0.65
MAIN_MIN    = 0.25

def run(cmd):
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(r.returncode)

def jload(rel):
    p = RUNS/rel
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
    for c in ["w_main","w_vol","w_osc","w_sym","w_reflex"]:
        if c not in out.columns:
            out[c] = 0.0
    # multiply sleeves
    out["w_vol"]    = out["w_vol"]    * mv
    out["w_osc"]    = out["w_osc"]    * mo
    out["w_sym"]    = out["w_sym"]    * ms
    out["w_reflex"] = out["w_reflex"] * mr
    # cap sleeves sum
    sleeves = out[["w_vol","w_osc","w_sym","w_reflex"]].sum(axis=1)
    over = sleeves > SLEEVES_CAP
    scale = np.where(over, SLEEVES_CAP / sleeves.replace(0, np.nan), 1.0)
    for c in ["w_vol","w_osc","w_sym","w_reflex"]:
        out[c] = out[c] * scale
    # leave room for main
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
    if not WCSV.exists():
        sys.exit("Missing runs_plus/regime_weights.csv (run tools/make_regime.py first).")

    base = pd.read_csv(WCSV)
    original = base.copy()
    rows = []
    try:
        for mv, mo, ms, mr in itertools.product(VOL_M, OSC_M, SYM_M, RFX_M):
            tweaked = adjust_weights(base, mv, mo, ms, mr)
            tweaked.to_csv(WCSV, index=False)

            # Rebuild governed portfolios
            run(["python", str(APPLY_REG)])
            run(["python", str(PATCH_DNA)])
            run(["python", str(APPLY_DNA)])

            # Rebuild Main & Vol-Target so metrics line up
            run(["python", str(PORTF_BUILDER)])
            run(["python", str(PORTF_VT)])

            # Read metrics
            main = jload("final_portfolio_summary.json")
            reg  = jload("final_portfolio_regime_summary.json")
            dna  = jload("final_portfolio_regime_dna_summary.json")
            vt   = jload("final_portfolio_vt_summary.json")

            rows.append(dict(
                mv=mv, mo=mo, ms=ms, mr=mr,
                main_oos=metric_or(main, ["out_sample","sharpe"]),
                reg_oos =metric_or(reg,  ["out_sample","sharpe"]),
                dna_oos =metric_or(dna,  ["out_sample","sharpe"]),
                vt_oos  =metric_or(vt,   ["out_sample","sharpe"]),
                vt_dd   =metric_or(vt,   ["out_sample","maxdd"]),
            ))

        df = pd.DataFrame(rows).sort_values(["vt_oos","dna_oos","main_oos"], ascending=False)
        outp = RUNS/"sweep_meta_micro.csv"
        df.to_csv(outp, index=False)
        print(df.head(20).to_string(index=False))
        print("Saved:", outp)
    finally:
        # Restore original
        original.to_csv(WCSV, index=False)
