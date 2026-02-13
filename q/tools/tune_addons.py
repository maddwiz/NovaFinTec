#!/usr/bin/env python3
"""
tune_addons.py

Sweeps add-on knobs and picks best by WF+ avg Sharpe:
  - HB_MULT          (heartbeat exposure scaler)
  - REFLEXIVE_CLIP   (bounds reflexive overlay)
  - SYMBOLIC_TILT    (add/subtract tilt when symbolic score extreme)
  - DNA_THRESH       (drift gate threshold)

Writes:
  runs_plus/tuning_addons.csv
  runs_plus/tuning_best.json
"""

import os, sys, json, subprocess, itertools
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
RUNS  = ROOT / "runs_plus"

HB_MULT          = [0.8, 1.0, 1.2]
REFLEXIVE_CLIP   = [0.10, 0.15, 0.20, 0.25]
SYMBOLIC_TILT    = [0.05, 0.10, 0.15]
DNA_THRESH       = [0.03, 0.05, 0.07]

def run(cmd, env=None):
    print("▶", " ".join(map(str, cmd)))
    subprocess.check_call(cmd, env=env)

def parse_plus_csv():
    p = RUNS / "walk_forward_table_plus.csv"
    if not p.exists(): return None
    df = pd.read_csv(p)
    s = pd.to_numeric(df.get("sharpe"), errors="coerce").dropna()
    h = pd.to_numeric(df.get("hit"), errors="coerce").dropna()
    dd = pd.to_numeric(df.get("maxDD"), errors="coerce").dropna() if "maxDD" in df.columns else None
    return {
        "assets": len(df),
        "hit": float(h.mean()) if len(h) else None,
        "sharpe": float(s.mean()) if len(s) else None,
        "maxDD": float(dd.mean()) if dd is not None and len(dd) else None
    }

if __name__=="__main__":
    results=[]
    # ensure artifacts
    run([sys.executable, str(TOOLS / "run_all_plus.py")])
    run([sys.executable, str(TOOLS / "make_hive_council.py")])
    run([sys.executable, str(TOOLS / "make_oos_all.py")])

    grid = list(itertools.product(HB_MULT, REFLEXIVE_CLIP, SYMBOLIC_TILT, DNA_THRESH))
    for hb, rc, st, dna in grid:
        env=os.environ.copy()
        env["HB_MULT"]=str(hb)
        env["REFLEXIVE_CLIP"]=str(rc)
        env["SYMBOLIC_TILT"]=str(st)
        env["DNA_THRESH"]=str(dna)
        run([sys.executable, str(TOOLS / "walk_forward_plus.py")], env=env)
        m = parse_plus_csv() or {}
        results.append({
            "HB_MULT": hb, "REFLEXIVE_CLIP": rc, "SYMBOLIC_TILT": st, "DNA_THRESH": dna,
            "sharpe": m.get("sharpe"), "hit": m.get("hit"), "maxDD": m.get("maxDD")
        })
    df = pd.DataFrame(results)
    df.to_csv(RUNS/"tuning_addons.csv", index=False)
    dff=df.dropna(subset=["sharpe"]).copy()
    dff["dd_score"]=dff["maxDD"]
    dff=dff.sort_values(["sharpe","dd_score","hit"], ascending=[False,False,False])
    best=dff.iloc[0].to_dict()
    (RUNS/"tuning_best.json").write_text(json.dumps(best, indent=2))
    print("\nBEST ADD-ON SETTINGS")
    print(json.dumps(best, indent=2))
    print("✅ Saved tuning_addons.csv and tuning_best.json")
