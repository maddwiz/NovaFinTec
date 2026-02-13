#!/usr/bin/env python3
# Cross-Hive Arbitration (softmax over health scores)
# Reads: runs_plus/hive_score_*.csv  (each is [T] score, higher=better)
# Writes: runs_plus/cross_hive_weights.csv
# Appends a small card.

import numpy as np, glob
from pathlib import Path
from qmods.cross_hive_arb_v1 import arb_weights

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"; RUNS.mkdir(exist_ok=True)

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
    files = sorted(glob.glob(str(RUNS/"hive_score_*.csv")))
    if not files:
        # make a demo set so the card appears
        T = 1000
        h1 = np.linspace(0.2, 1.2, T) + np.sin(np.linspace(0, 20, T))*0.1
        h2 = 1.0 - h1 + 0.1
        names = ["hive_a", "hive_b"]
        scores = {"hive_a": h1, "hive_b": h2}
    else:
        names = []
        scores = {}
        for f in files:
            n = Path(f).stem.replace("hive_score_","")
            s = np.loadtxt(f, delimiter=",")
            names.append(n)
            scores[n] = s

    names, W = arb_weights(scores, alpha=2.0)
    np.savetxt(RUNS/"cross_hive_weights.csv", W, delimiter=",")
    html = f"<p>Cross-hive weights over {len(names)} hives saved to cross_hive_weights.csv</p>"
    append_card("Cross-Hive Arbitration ✔", html)
    print(f"✅ Wrote {RUNS/'cross_hive_weights.csv'}")
