#!/usr/bin/env python3
# tools/sweep_regime_small.py
# Sweeps a few safe values for sleeve scales (VOL/OSC) and DNA STRENGTH.
# Prints a results table and writes runs_plus/sweep_regime_small.csv.
from pathlib import Path
import pandas as pd, numpy as np, json, shutil, itertools, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"

# candidate knobs (tiny, safe)
VOL_CAND = [0.0015, 0.0020, 0.0025]
OSC_CAND = [0.0012, 0.0016, 0.0020]
DNA_STRENGTH = [0.25, 0.40, 0.55]  # attenuation strength

BUILD_SLEEVES = ROOT/"tools"/"build_min_sleeves.py"
PATCH_DNA     = ROOT/"tools"/"patch_regime_weights_with_dna.py"
MAKE_REGIME   = ROOT/"tools"/"make_regime.py"
APPLY_REG     = ROOT/"tools"/"apply_regime_governor.py"
APPLY_DNA     = ROOT/"tools"/"apply_regime_governor_dna.py"

def read_json(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else {}

def metric_or(d, path, default=np.nan):
    cur = d
    for k in path:
        if isinstance(cur, dict) and k in cur: cur = cur[k]
        else: return default
    try:
        return float(cur)
    except:
        return default

def rewrite_scales(vol_scale, osc_scale):
    # rewrite tools/build_min_sleeves.py constants VOL_SCALE/OSC_SCALE
    p = BUILD_SLEEVES
    txt = p.read_text()
    import re
    txt = re.sub(r"VOL_SCALE\s*=\s*[0-9.]+", f"VOL_SCALE = {vol_scale}", txt)
    txt = re.sub(r"OSC_SCALE\s*=\s*[0-9.]+", f"OSC_SCALE = {osc_scale}", txt)
    p.write_text(txt)

def rewrite_strength(strength):
    # rewrite tools/patch_regime_weights_with_dna.py STRENGTH
    p = PATCH_DNA
    txt = p.read_text()
    import re
    txt = re.sub(r"STRENGTH\s*=\s*[0-9.]+", f"STRENGTH  = {strength}", txt)
    p.write_text(txt)

def run(cmd):
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        print("ERR:", " ".join(cmd))
        print(r.stdout)
        print(r.stderr)
    return r.returncode == 0

def ann_sharpe_csv(pth, ret_col):
    df = pd.read_csv(pth)
    if "DATE" in df.columns:
        df = df.sort_values("DATE")
    r = pd.to_numeric(df[ret_col], errors="coerce").replace([np.inf,-np.inf], np.nan).dropna()
    if len(r) < 50: return np.nan
    s = r.std()
    if not np.isfinite(s) or s == 0: return np.nan
    return float((r.mean()/s)*np.sqrt(252.0))

if __name__ == "__main__":
    results = []
    # baseline (for context)
    base = read_json(RUNS/"final_portfolio_summary.json")
    base_oos = metric_or(base, ["out_sample","sharpe"], np.nan)

    for vol, osc, s in itertools.product(VOL_CAND, OSC_CAND, DNA_STRENGTH):
        print(f"TRY vol={vol} osc={osc} dna_strength={s}")
        rewrite_scales(vol, osc)
        run(["python", str(BUILD_SLEEVES)])
        # fresh regime weights from features
        run(["python", str(MAKE_REGIME)])
        run(["python", str(APPLY_REG)])
        rewrite_strength(s)
        run(["python", str(PATCH_DNA)])
        run(["python", str(APPLY_DNA)])

        reg  = read_json(RUNS/"final_portfolio_regime_summary.json")
        dna  = read_json(RUNS/"final_portfolio_regime_dna_summary.json")
        reg_oos = metric_or(reg, ["out_sample","sharpe"], np.nan)
        dna_oos = metric_or(dna, ["out_sample","sharpe"], np.nan)
        reg_dd  = metric_or(reg, ["out_sample","maxdd"], np.nan)
        dna_dd  = metric_or(dna, ["out_sample","maxdd"], np.nan)

        results.append(dict(vol=vol, osc=osc, dna_strength=s,
                            base_oos=base_oos, reg_oos=reg_oos, dna_oos=dna_oos,
                            reg_maxdd=reg_dd, dna_maxdd=dna_dd))

    df = pd.DataFrame(results).sort_values(["dna_oos","reg_oos"], ascending=False)
    outp = RUNS/"sweep_regime_small.csv"
    df.to_csv(outp, index=False)
    print("=== SWEEP DONE ===")
    print(df.head(12).to_string(index=False))
    print("Saved:", outp)
