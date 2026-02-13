#!/usr/bin/env python3
# tools/sweep_osc_micro.py
# Micro-sweep oscillator params: lookback and thresholds.
# Optimizes for Vol-Target OOS Sharpe, tie-breaker DNA OOS, then Main OOS.
# Writes runs_plus/sweep_osc_micro.csv

from pathlib import Path
import re, subprocess, sys, json, pandas as pd, numpy as np
import itertools

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"

# Try these small ranges
LOOKBACKS = [10, 14, 20]
UP_BANDS  = [0.7, 0.8]
DN_BANDS  = [0.2, 0.3]

# Where the oscillator constants live:
OSC_FILE_CANDIDATES = [
    ROOT/"qmods"/"osc.py",
    ROOT/"qengine"/"signals.py",
]

def find_osc_file():
    for p in OSC_FILE_CANDIDATES:
        if p.exists(): return p
    raise SystemExit("Could not find oscillator module (expected qmods/osc.py or qengine/signals.py).")

def run(cmd):
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode!=0: sys.exit(r.returncode)

def j(rel):
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
    except: return default

def patch_params(txt, look, up, dn):
    txt = re.sub(r"(OSC_LOOKBACK\s*=\s*)\d+", rf"\g<1>{look}", txt)
    txt = re.sub(r"(OSC_UP\s*=\s*)[0-9.]+",   rf"\g<1>{up}",  txt)
    txt = re.sub(r"(OSC_DOWN\s*=\s*)[0-9.]+", rf"\g<1>{dn}",  txt)
    return txt

if __name__ == "__main__":
    osc_path = find_osc_file()
    original = osc_path.read_text()
    rows = []
    try:
        for look, up, dn in itertools.product(LOOKBACKS, UP_BANDS, DN_BANDS):
            osc_path.write_text(patch_params(original, look, up, dn))

            # Rebuild sleeves and governed portfolios so oscillator feeds through
            run(["python","tools/build_min_sleeves.py"])
            run(["python","tools/make_regime.py"])
            run(["python","tools/apply_regime_governor.py"])
            run(["python","tools/patch_regime_weights_with_dna.py"])
            run(["python","tools/apply_regime_governor_dna.py"])

            # Rebuild Main and Vol-Target
            run(["python","tools/portfolio_from_runs_plus.py"])
            run(["python","tools/portfolio_vol_target.py"])

            main = j("final_portfolio_summary.json")
            dna  = j("final_portfolio_regime_dna_summary.json")
            vt   = j("final_portfolio_vt_summary.json")

            rows.append(dict(
                lookback=look, up=up, down=dn,
                main_oos=metric_or(main, ["out_sample","sharpe"]),
                dna_oos =metric_or(dna,  ["out_sample","sharpe"]),
                vt_oos  =metric_or(vt,   ["out_sample","sharpe"]),
                vt_dd   =metric_or(vt,   ["out_sample","maxdd"]),
            ))

        df = pd.DataFrame(rows).sort_values(["vt_oos","dna_oos","main_oos"], ascending=False)
        outp = RUNS/"sweep_osc_micro.csv"
        df.to_csv(outp, index=False)
        print(df.head(15).to_string(index=False))
        print("Saved:", outp)
    finally:
        osc_path.write_text(original)
