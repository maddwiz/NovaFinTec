#!/usr/bin/env python3
# tools/make_scoreboard.py
# Prints a compact scoreboard to terminal and writes runs_plus/scoreboard.csv
# Tracks: Main, Regime, Regime+DNA, Vol-Target (gross & net if available)

from pathlib import Path
import json, pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"

def jload(name):
    p = RUNS/name
    if not p.exists(): return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def row(name, gross, net):
    def g(path): 
        try: return float(np.round(eval(path, {}, {"d":gross, "np":np}), 6))
        except: return np.nan
    def n(path):
        if net is None: return np.nan
        try: return float(np.round(eval(path, {}, {"d":net, "np":np}), 6))
        except: return np.nan
    return dict(
        track=name,
        oos_sharpe_g = g("d['out_sample']['sharpe']"),
        oos_maxdd_g  = g("d['out_sample']['maxdd']"),
        is_sharpe_g  = g("d['in_sample']['sharpe']"),
        oos_sharpe_n = n("d['out_sample']['sharpe']"),
        oos_maxdd_n  = n("d['out_sample']['maxdd']")
    )

if __name__ == "__main__":
    main_g = jload("final_portfolio_summary.json")
    reg_g  = jload("final_portfolio_regime_summary.json")
    dna_g  = jload("final_portfolio_regime_dna_summary.json")
    vt_g   = jload("final_portfolio_vt_summary.json")

    main_n = jload("final_portfolio_summary_costs.json")
    reg_n  = jload("final_portfolio_regime_summary_costs.json")
    dna_n  = jload("final_portfolio_regime_dna_summary_costs.json")
    vt_n   = jload("final_portfolio_vt_summary_costs.json")

    rows=[]
    if main_g: rows.append(row("Main", main_g, main_n))
    if reg_g:  rows.append(row("Regime", reg_g, reg_n))
    if dna_g:  rows.append(row("Regime+DNA", dna_g, dna_n))
    if vt_g:   rows.append(row("Vol-Target", vt_g, vt_n))

    df = pd.DataFrame(rows)
    outp = RUNS/"scoreboard.csv"
    df.to_csv(outp, index=False)
    # Pretty print
    show = df.copy()
    for c in ["oos_sharpe_g","oos_sharpe_n","oos_maxdd_g","oos_maxdd_n","is_sharpe_g"]:
        if c in show.columns:
            show[c] = show[c].map(lambda x: f"{x:0.3f}" if pd.notna(x) else "—")
    print(show.to_string(index=False))
    print("Saved:", outp)
