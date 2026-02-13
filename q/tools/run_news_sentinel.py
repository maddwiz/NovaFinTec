#!/usr/bin/env python3
# Builds a shock mask from vol (and optional news flags) and gates a signal.

import numpy as np
from pathlib import Path
from qmods.news_shock_guard import shock_mask, apply_shock_guard

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"; RUNS.mkdir(exist_ok=True)

def load_series(rel):
    p = ROOT/rel
    if not p.exists(): return None
    try:    return np.loadtxt(p, delimiter=",").ravel()
    except: return np.loadtxt(p, delimiter=",", skiprows=1).ravel()

def first_series(paths):
    for rel in paths:
        a = load_series(rel)
        if a is not None: return a
    return None

def append_card(title, html):
    for name in ["report_all.html","report_best_plus.html","report_plus.html"]:
        f = ROOT/name
        if not f.exists(): continue
        txt = f.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        f.write_text(txt.replace("</body>", card+"</body>") if "</body>" in txt else txt+card, encoding="utf-8")

if __name__ == "__main__":
    r = first_series(["runs_plus/daily_returns.csv","daily_returns.csv"])
    if r is None:
        print("(!) No returns; skipping."); raise SystemExit(0)
    vol = np.abs(r)

    mask = shock_mask(vol, z=2.5, min_len=2)
    news = load_series("runs_plus/news_events.csv")
    if news is not None:
        L = min(len(mask), len(news))
        mask[:L] = np.maximum(mask[:L], (news[:L] > 0).astype(int))

    np.savetxt(RUNS/"shock_mask.csv", mask, delimiter=",")

    # Gate a common signal so you see output now
    sig = first_series(["runs_plus/meta_stack_pred.csv","runs_plus/synapses_pred.csv","runs_plus/reflex_signal.csv"])
    if sig is not None:
        L = min(len(sig), len(mask))
        gated = apply_shock_guard(sig[:L], mask[:L], alpha=0.5)
        np.savetxt(RUNS/"signal_shock_gated.csv", gated, delimiter=",")

    append_card("Shock/News Sentinel ✔",
                f"<p>Saved shock_mask.csv{' + signal_shock_gated.csv' if sig is not None else ''}</p>")
    print("✅ Wrote runs_plus/shock_mask.csv")
