#!/usr/bin/env python3
# Bags a prediction series over random subwindows → meta_bagged.csv

import numpy as np
from pathlib import Path
from qmods.bag_time import time_bags, bag_mean

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
    pred = first_series(["runs_plus/meta_stack_pred.csv","runs_plus/synapses_pred.csv"])
    if pred is None:
        print("(!) No meta/synapses predictions; skipping."); raise SystemExit(0)

    T = len(pred)
    idxs = time_bags(T, bag_size=0.7, bags=20, seed=42)

    bags = []
    for idx in idxs:
        p = np.full(T, np.nan)
        p[idx] = pred[idx]
        bags.append(p)
    bagged = bag_mean(np.vstack(bags))
    np.savetxt(RUNS/"meta_bagged.csv", bagged, delimiter=",")

    append_card("Bagging by Time ✔", f"<p>Saved meta_bagged.csv (bags=20, size=70%).</p>")
    print("✅ Wrote runs_plus/meta_bagged.csv")
