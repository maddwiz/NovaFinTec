#!/usr/bin/env python3
# Nonlinear fusion of councils (tiny MLP)
# Reads:  runs_plus/council_votes.csv, returns like run_meta_stack
# Writes: runs_plus/synapses_pred.csv
# Appends a small card.

import numpy as np
from pathlib import Path
from qmods.synapses_small import SynapseSmall

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"; RUNS.mkdir(exist_ok=True)

def load_first(paths):
    for p in paths:
        f = ROOT/p
        if f.exists():
            try:
                return np.loadtxt(f, delimiter=",")
            except:
                try:
                    return np.loadtxt(f, delimiter=",", skiprows=1)
                except:
                    pass
    return None

def append_card(title, html):
    for name in ["report_all.html", "report_best_plus.html", "report_plus.html"]:
        p = ROOT/name
        if not p.exists(): continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        txt = txt.replace("</body>", card + "</body>") if "</body>" in txt else txt+card
        p.write_text(txt, encoding="utf-8")
        print(f"✅ Appended card to {name}")

if __name__ == "__main__":
    V = load_first(["runs_plus/council_votes.csv"])
    y = load_first([
        "runs_plus/target_returns.csv",
        "runs_plus/daily_returns.csv",
        "daily_returns.csv",
        "portfolio_daily_returns.csv"
    ])
    if V is None or y is None:
        print("(!) Missing council_votes or returns. Skipping."); raise SystemExit(0)
    if V.ndim == 1: V = V.reshape(-1,1)
    y = np.asarray(y).ravel()
    T = min(len(y), V.shape[0]); V = V[:T]; y = y[:T]

    nn = SynapseSmall(hidden=8, lr=0.01, reg=1e-3, epochs=200).fit(V, y)
    pred = nn.predict(V)
    np.savetxt(RUNS/"synapses_pred.csv", pred, delimiter=",")
    html = f"<p>SynapsesSmall trained on {T} rows, K={V.shape[1]}. Saved synapses_pred.csv</p>"
    append_card("Neural Synapses (Tiny MLP) ✔", html)
    print(f"✅ Wrote {RUNS/'synapses_pred.csv'}")
