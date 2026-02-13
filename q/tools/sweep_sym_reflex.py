#!/usr/bin/env python3
# tools/sweep_sym_reflex.py
# Sweeps tiny scales for Symbolic & Reflexive sleeves by rewriting
# tools/apply_regime_governor.py and tools/apply_regime_governor_dna.py
# constants (SCALE_SYM, SCALE_REFLEX), then rebuilding Regime + Regime+DNA.
# Writes: runs_plus/sweep_sym_reflex.csv (best rows at the top).

from pathlib import Path
import re, json, subprocess, sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"

GOV  = ROOT / "tools" / "apply_regime_governor.py"
GOVD = ROOT / "tools" / "apply_regime_governor_dna.py"

# Try very small → small values only (safe)
SYM_SET = [0.0005, 0.0010, 0.0015]
RFX_SET = [0.0003, 0.0008, 0.0012]

def run(cmd):
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(r.returncode)

def read_json(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else {}

def metric_or(d, path, default=np.nan):
    cur = d
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    try:
        return float(cur)
    except:
        return default

def rewrite_scales(sym_scale, rfx_scale):
    # apply to both governors (with & without DNA)
    for p in (GOV, GOVD):
        txt = p.read_text()
        txt = re.sub(r"(SCALE_SYM\s*=\s*)[0-9.]+", rf"\g<1>{sym_scale}", txt)
        txt = re.sub(r"(SCALE_REFLEX\s*=\s*)[0-9.]+", rf"\g<1>{rfx_scale}", txt)
        p.write_text(txt)

if __name__ == "__main__":
    rows = []
    # keep originals in memory to restore at the end
    orig_gov  = GOV.read_text()
    orig_govd = GOVD.read_text()
    try:
        for s in SYM_SET:
            for r in RFX_SET:
                rewrite_scales(s, r)
                run(["python", "tools/make_regime.py"])
                run(["python", "tools/apply_regime_governor.py"])
                run(["python", "tools/patch_regime_weights_with_dna.py"])
                run(["python", "tools/apply_regime_governor_dna.py"])

                reg = read_json(RUNS/"final_portfolio_regime_summary.json")
                dna = read_json(RUNS/"final_portfolio_regime_dna_summary.json")
                rows.append(dict(
                    sym_scale=s,
                    reflex_scale=r,
                    reg_oos=metric_or(reg, ["out_sample","sharpe"]),
                    dna_oos=metric_or(dna, ["out_sample","sharpe"]),
                    reg_dd =metric_or(reg, ["out_sample","maxdd"]),
                    dna_dd =metric_or(dna, ["out_sample","maxdd"]),
                ))
        df = pd.DataFrame(rows).sort_values(["dna_oos","reg_oos"], ascending=False)
        outp = RUNS/"sweep_sym_reflex.csv"
        df.to_csv(outp, index=False)
        print(df.to_string(index=False))
        print("Saved:", outp)
    finally:
        # restore originals
        GOV.write_text(orig_gov)
        GOVD.write_text(orig_govd)
