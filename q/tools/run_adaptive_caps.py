#!/usr/bin/env python3
# Adaptive Caps (no array "or" usage)

import numpy as np
from pathlib import Path
from qmods.adaptive_caps import adaptive_cap, apply_caps

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"; RUNS.mkdir(exist_ok=True)

def load_matrix(rel):
    p = ROOT / rel
    if not p.exists(): return None
    try:
        a = np.loadtxt(p, delimiter=",")
    except:
        a = np.loadtxt(p, delimiter=",", skiprows=1)
    if a.ndim == 1: a = a.reshape(-1,1)
    return a

def load_series(rel):
    p = ROOT / rel
    if not p.exists(): return None
    try:    return np.loadtxt(p, delimiter=",").ravel()
    except: return np.loadtxt(p, delimiter=",", skiprows=1).ravel()

def first_matrix(paths):
    for rel in paths:
        a = load_matrix(rel)
        if a is not None: return a
    return None

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
    W = first_matrix(["portfolio_weights.csv","runs_plus/portfolio_weights.csv"])
    r = first_series(["runs_plus/daily_returns.csv","daily_returns.csv"])
    if W is None or r is None:
        print("(!) Missing weights or returns; skipping."); raise SystemExit(0)

    T = min(W.shape[0], len(r))
    W = W[:T]; r = r[:T]

    vol = np.abs(r)
    caps = adaptive_cap(vol, cap_min=0.05, cap_max=0.15)[:T]
    np.savetxt(RUNS/"adaptive_caps.csv", caps, delimiter=",")

    Wc = apply_caps(W, caps)
    np.savetxt(RUNS/"weights_capped.csv", Wc, delimiter=",")

    append_card("Adaptive Caps ✔", f"<p>Saved adaptive_caps.csv + weights_capped.csv (T={T}, N={W.shape[1]})</p>")
    print("✅ Saved adaptive_caps.csv and weights_capped.csv")
