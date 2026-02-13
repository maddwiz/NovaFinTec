#!/usr/bin/env python3
# tools/sweep_overlay_micro.py
# Sweeps overlay parameters for make_overlay_alpha.py and measures VT+Overlay OOS Sharpe.
# Writes runs_plus/sweep_overlay_micro.csv (best rows first).

from pathlib import Path
import re, subprocess, sys, json, pandas as pd, numpy as np
import itertools

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"
MAKER = ROOT/"tools"/"make_overlay_alpha.py"
APPLIER = ROOT/"tools"/"apply_overlay_to_vt.py"

# Small, safe grid
RET_WINDOWS = [14, 20, 30]
ALPHA_MINS  = [0.75, 0.80]
ALPHA_MAXS  = [1.20, 1.30]
BREADTH_WS  = [0.5, 0.6, 0.7]   # calm weight will be 1 - breadth

def run(cmd):
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0: sys.exit(r.returncode)

def read_json(rel):
    p = RUNS/rel
    return json.loads(p.read_text()) if p.exists() else {}

def patch(txt, ret_win, a_min, a_max, bw):
    cw = 1.0 - bw
    txt = re.sub(r"(RET_WINDOW\s*=\s*)\d+",         rf"\g<1>{ret_win}", txt)
    txt = re.sub(r"(ALPHA_MIN\s*=\s*)[0-9.]+",      rf"\g<1>{a_min}",   txt)
    txt = re.sub(r"(ALPHA_MAX\s*=\s*)[0-9.]+",      rf"\g<1>{a_max}",   txt)
    txt = re.sub(r"(BREADTH_W\s*=\s*)[0-9.]+",      rf"\g<1>{bw}",      txt)
    txt = re.sub(r"(CALM_W\s*=\s*)[0-9.]+",         rf"\g<1>{cw}",      txt)
    return txt

if __name__ == "__main__":
    original = MAKER.read_text()
    rows = []
    try:
        for ret_win, a_min, a_max, bw in itertools.product(RET_WINDOWS, ALPHA_MINS, ALPHA_MAXS, BREADTH_WS):
            MAKER.write_text(patch(original, ret_win, a_min, a_max, bw))
            run(["python", str(MAKER)])
            run(["python", str(APPLIER)])
            summ = read_json("final_portfolio_vt_overlay_summary.json")
            rows.append(dict(
                ret_window=ret_win, alpha_min=a_min, alpha_max=a_max, breadth_w=bw, calm_w=1.0-bw,
                vt_ov_oos=summ.get("out_sample",{}).get("sharpe", np.nan),
                vt_ov_dd =summ.get("out_sample",{}).get("maxdd",  np.nan),
            ))
        df = pd.DataFrame(rows).sort_values(["vt_ov_oos"], ascending=False)
        outp = RUNS/"sweep_overlay_micro.csv"
        df.to_csv(outp, index=False)
        print(df.head(15).to_string(index=False))
        print("Saved:", outp)
    finally:
        MAKER.write_text(original)
