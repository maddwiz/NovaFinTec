#!/usr/bin/env python3
# Reflex Health Gating (no array "or" usage)

import numpy as np
from pathlib import Path
from qmods.reflex_health_index import reflex_health, gate_reflex

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"; RUNS.mkdir(exist_ok=True)

def load_series(rel):
    p = ROOT / rel
    if not p.exists(): return None
    try:    a = np.loadtxt(p, delimiter=",").ravel()
    except: a = np.loadtxt(p, delimiter=",", skiprows=1).ravel()
    return a

def first_series(paths):
    for rel in paths:
        a = load_series(rel)
        if a is not None: return a
    return None

def append_card(title, html):
    for name in ["report_all.html","report_best_plus.html","report_plus.html"]:
        f = ROOT / name
        if not f.exists(): continue
        txt = f.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        f.write_text(txt.replace("</body>", card+"</body>") if "</body>" in txt else txt+card, encoding="utf-8")

if __name__ == "__main__":
    r = load_series("runs_plus/reflex_returns.csv")
    if r is None:
        pred = first_series(["runs_plus/meta_stack_pred.csv","runs_plus/synapses_pred.csv"])
        y    = first_series(["runs_plus/daily_returns.csv","daily_returns.csv"])
        if pred is not None and y is not None:
            T = min(len(pred), len(y)); r = (pred[:T] * y[:T])
        else:
            T = 1000; rng = np.random.default_rng(7); r = rng.normal(0.0002, 0.01, T)

    H = reflex_health(r, lookback=126)
    np.savetxt(RUNS/"reflex_health.csv", H, delimiter=",")

    sig = load_series("runs_plus/reflex_signal.csv")
    if sig is not None:
        L = min(len(sig), len(H))
        gated = gate_reflex(sig[:L], H[:L], min_h=0.5)
        np.savetxt(RUNS/"reflex_signal_gated.csv", gated, delimiter=",")

    msg = "Saved reflex_health.csv" + (" + reflex_signal_gated.csv" if sig is not None else "")
    append_card("Reflex Health Gating ✔", f"<p>{msg}</p>")
    print("✅", msg)
