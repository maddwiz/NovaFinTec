#!/usr/bin/env python3
# tools/sweep_regime_fine.py
# Fine sweep around sleeve scales and DNA strength.
# Writes results to runs_plus/sweep_regime_small.csv (same filename as before).

from pathlib import Path
import pandas as pd, numpy as np, json, re, subprocess, sys, itertools

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"

BUILD_SLEEVES = ROOT/"tools"/"build_min_sleeves.py"
PATCH_DNA     = ROOT/"tools"/"patch_regime_weights_with_dna.py"
MAKE_REGIME   = ROOT/"tools"/"make_regime.py"
APPLY_REG     = ROOT/"tools"/"apply_regime_governor.py"
APPLY_DNA     = ROOT/"tools"/"apply_regime_governor_dna.py"

# Centered around your good zone
VOL_CAND = [0.0014, 0.0016, 0.0018]
OSC_CAND = [0.0018, 0.0020, 0.0022]
DNA_STRENGTH = [0.40, 0.50, 0.60]
ALPHA_MIN = [0.60, 0.70]  # optional floor to keep sleeves alive

def read_json(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else {}

def metric_or(d, path, default=np.nan):
    cur = d
    for k in path:
        if isinstance(cur, dict) and k in cur: cur = cur[k]
        else: return default
    try: return float(cur)
    except: return default

def rewrite_scales(vol_scale, osc_scale):
    p = BUILD_SLEEVES
    txt = p.read_text()
    txt = re.sub(r"(VOL_SCALE\s*=\s*)[0-9.]+", rf"\g<1>{vol_scale}", txt)
    txt = re.sub(r"(OSC_SCALE\s*=\s*)[0-9.]+", rf"\g<1>{osc_scale}", txt)
    p.write_text(txt)

def rewrite_dna(strength, alpha_min):
    p = PATCH_DNA
    txt = p.read_text()
    txt = re.sub(r"(STRENGTH\s*=\s*)[0-9.]+", rf"\g<1>{strength}", txt)
    txt = re.sub(r"(ALPHA_MIN\s*=\s*)[0-9.]+", rf"\g<1>{alpha_min}", txt)
    p.write_text(txt)

def run(cmd):
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(r.returncode)

if __name__ == "__main__":
    rows = []
    for vol in VOL_CAND:
        for osc in OSC_CAND:
            for s in DNA_STRENGTH:
                for a in ALPHA_MIN:
                    rewrite_scales(vol, osc)
                    run(["python", "tools/build_min_sleeves.py"])
                    run(["python", "tools/make_regime.py"])
                    run(["python", "tools/apply_regime_governor.py"])
                    rewrite_dna(s, a)
                    run(["python", "tools/patch_regime_weights_with_dna.py"])
                    run(["python", "tools/apply_regime_governor_dna.py"])

                    reg = read_json(RUNS/"final_portfolio_regime_summary.json")
                    dna = read_json(RUNS/"final_portfolio_regime_dna_summary.json")
                    rows.append(dict(
                        vol=vol, osc=osc, dna_strength=s, alpha_min=a,
                        reg_oos=metric_or(reg, ["out_sample","sharpe"]),
                        dna_oos=metric_or(dna, ["out_sample","sharpe"]),
                        reg_dd =metric_or(reg, ["out_sample","maxdd"]),
                        dna_dd =metric_or(dna, ["out_sample","maxdd"]),
                    ))
    df = pd.DataFrame(rows).sort_values(["dna_oos","reg_oos"], ascending=False)
    outp = RUNS/"sweep_regime_small.csv"
    df.to_csv(outp, index=False)
    print(df.head(12).to_string(index=False))
    print("Saved:", outp)
