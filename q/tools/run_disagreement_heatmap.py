#!/usr/bin/env python3
# Saves a simple time-series dispersion (std) of council votes.
# Writes: runs_plus/disagreement_std.csv
# Appends a small table to report_all.html if present.

import numpy as np, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"; RUNS.mkdir(exist_ok=True)

def append_report_table(rows):
    for name in ["report_all.html", "report_best_plus.html"]:
        p = ROOT / name
        if not p.exists(): 
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        table = "<table border='1' cellpadding='4'><tr><th>t</th><th>std(votes)</th></tr>" + \
                "".join(f"<tr><td>{i}</td><td>{v:.3f}</td></tr>" for i,v in rows[-50:]) + "</table>"
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>Council Disagreement (std)</h3>{table}</div>\n'
        if "</body>" in txt:
            txt = txt.replace("</body>", card + "</body>")
        else:
            txt += card
        p.write_text(txt, encoding="utf-8")
        print(f"✅ Appended Disagreement table to {name}")

if __name__ == "__main__":
    votes_p = RUNS / "council_votes.csv"
    if not votes_p.exists():
        print("No runs_plus/council_votes.csv found. Skipping.")
        raise SystemExit(0)
    V = np.loadtxt(votes_p, delimiter=",")
    if V.ndim == 1: V = V.reshape(-1,1)
    stds = V.std(axis=1)
    outp = RUNS/"disagreement_std.csv"
    np.savetxt(outp, stds, delimiter=",")
    print(f"✅ Wrote {outp}")
    append_report_table(list(enumerate(stds)))
