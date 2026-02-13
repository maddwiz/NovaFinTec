#!/usr/bin/env python3
# tools/apply_overlay_micro_best.py
# Applies the top row from sweep_overlay_micro.csv into make_overlay_alpha.py,
# rebuilds VT+Overlay, updates the Overlay card, and prints final OOS metrics.

from pathlib import Path
import re, pandas as pd, subprocess, sys, json, numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"
SWEEP = RUNS/"sweep_overlay_micro.csv"
MAKER = ROOT/"tools"/"make_overlay_alpha.py"
APPLIER = ROOT/"tools"/"apply_overlay_to_vt.py"
CARD = ROOT/"tools"/"add_overlay_card.py"

def run(cmd):
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0: sys.exit(r.returncode)

def read_json(rel):
    p = RUNS/rel
    return json.loads(p.read_text()) if p.exists() else {}

def set_params(ret_win, a_min, a_max, bw):
    cw = 1.0 - bw
    txt = MAKER.read_text()
    txt = re.sub(r"(RET_WINDOW\s*=\s*)\d+",         rf"\g<1>{ret_win}", txt)
    txt = re.sub(r"(ALPHA_MIN\s*=\s*)[0-9.]+",      rf"\g<1>{a_min}",   txt)
    txt = re.sub(r"(ALPHA_MAX\s*=\s*)[0-9.]+",      rf"\g<1>{a_max}",   txt)
    txt = re.sub(r"(BREADTH_W\s*=\s*)[0-9.]+",      rf"\g<1>{bw}",      txt)
    txt = re.sub(r"(CALM_W\s*=\s*)[0-9.]+",         rf"\g<1>{cw}",      txt)
    MAKER.write_text(txt)

if __name__ == "__main__":
    if not SWEEP.exists(): sys.exit("Missing sweep_overlay_micro.csv (run tools/sweep_overlay_micro.py first).")
    df = pd.read_csv(SWEEP)
    if df.empty: sys.exit("sweep_overlay_micro.csv is empty.")
    best = df.iloc[0]

    # Compare to current VT+Overlay
    cur = read_json("final_portfolio_vt_overlay_summary.json")
    cur_oos = float(cur.get("out_sample",{}).get("sharpe", np.nan))

    best_oos = float(best["vt_ov_oos"])
    if np.isfinite(cur_oos) and best_oos <= cur_oos:
        print(f"No apply: best overlay OOS {best_oos:.3f} <= current {cur_oos:.3f}")
        sys.exit(0)

    # Apply best
    set_params(int(best["ret_window"]), float(best["alpha_min"]), float(best["alpha_max"]), float(best["breadth_w"]))
    print("✅ Applied best overlay parameters.")

    run(["python", str(MAKER)])
    run(["python", str(APPLIER)])
    try:
        run(["python", str(CARD)])
    except Exception:
        pass

    new = read_json("final_portfolio_vt_overlay_summary.json")
    oos = new.get("out_sample", {})
    print("VT+Overlay OOS: Sharpe=%.3f  MaxDD=%.3f" % (oos.get("sharpe", np.nan), oos.get("maxdd", np.nan)))
    print("✅ Overlay updated and report refreshed.")
