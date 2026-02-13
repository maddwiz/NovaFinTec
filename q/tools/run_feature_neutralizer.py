#!/usr/bin/env python3
# Feature Neutralization (no array "or" usage)

import numpy as np
from pathlib import Path
from qmods.feature_neutralizer import neutralize

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"; RUNS.mkdir(exist_ok=True)

def load_mat(rel):
    p = ROOT / rel
    if not p.exists(): return None
    try:    A = np.loadtxt(p, delimiter=",")
    except: A = np.loadtxt(p, delimiter=",", skiprows=1)
    if A.ndim == 1: A = A.reshape(-1,1)
    return A

def first_mat(paths):
    for rel in paths:
        A = load_mat(rel)
        if A is not None: return A
    return None

def append_card(title, html):
    for name in ["report_all.html","report_best_plus.html","report_plus.html"]:
        f = ROOT / name
        if not f.exists(): continue
        txt = f.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        f.write_text(txt.replace("</body>", card+"</body>") if "</body" in txt else txt+card, encoding="utf-8")

if __name__ == "__main__":
    A = first_mat(["runs_plus/council_votes.csv","runs_plus/meta_stack_pred.csv"])
    B = first_mat(["runs_plus/osc_signals.csv","runs_plus/synapses_pred.csv"])
    if A is None or B is None:
        print("(!) Missing features; skipping."); raise SystemExit(0)

    T = min(A.shape[0], B.shape[0])
    A = A[:T]; B = B[:T]
    A_neu = neutralize(A, B, strength=1.0)
    np.savetxt(RUNS/"feature_A_neutral.csv", A_neu, delimiter=",")

    append_card("Feature Neutralization ✔", f"<p>Saved feature_A_neutral.csv (T={T}).</p>")
    print("✅ Saved feature_A_neutral.csv")
