#!/usr/bin/env python3
# tools/diagnose_sleeves.py
# Prints average effective weights and OOS contribution for Main/Vol/Osc/Sym/Reflex (and DNA)
# so you can see if Oscillators are actually doing work.

from pathlib import Path
import json, pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"

def j(rel):
    p = RUNS/rel
    return json.loads(p.read_text()) if p.exists() else {}

def f(x): 
    try: return float(x)
    except: return np.nan

if __name__ == "__main__":
    # regime weights file holds effective sleeves by date
    wcsv = RUNS/"regime_weights.csv"
    if not wcsv.exists():
        raise SystemExit("Missing runs_plus/regime_weights.csv (run tools/make_regime.py first).")
    w = pd.read_csv(wcsv)
    cols = [c for c in ["w_main","w_vol","w_osc","w_sym","w_reflex"] if c in w.columns]
    avg_w = w[cols].mean().to_dict()

    main = j("final_portfolio_summary.json")
    reg  = j("final_portfolio_regime_summary.json")
    dna  = j("final_portfolio_regime_dna_summary.json")

    out = pd.DataFrame([
        {"track":"Main",  "oos_sharpe": f(main.get("out_sample",{}).get("sharpe")), "oos_maxdd": f(main.get("out_sample",{}).get("maxdd"))},
        {"track":"Regime","oos_sharpe": f(reg.get("out_sample",{}).get("sharpe")),  "oos_maxdd": f(reg.get("out_sample",{}).get("maxdd"))},
        {"track":"DNA",   "oos_sharpe": f(dna.get("out_sample",{}).get("sharpe")),  "oos_maxdd": f(dna.get("out_sample",{}).get("maxdd"))},
    ])

    print("Average effective sleeve weights (across dates):")
    for k,v in avg_w.items():
        print(f"  {k:8s} = {v:0.3f}")
    print("\nOOS scoreboard:")
    print(out.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))
    print("\nIf w_osc ≈ 0.00 or tiny vs w_main, the oscillator sleeve is being ignored.")
