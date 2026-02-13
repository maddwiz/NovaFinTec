#!/usr/bin/env python3
# Risk-Parity Sleeve (equal risk per asset)
# Reads:
#   runs_plus/asset_returns.csv  (T x N)  OR builds from data/*.csv (Close/Adj Close)
# Writes:
#   runs_plus/risk_parity_weights.csv
# Appends a card.

import csv, numpy as np
from pathlib import Path
from qmods.risk_parity_sleeve import risk_parity_weights

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"data"
RUNS = ROOT/"runs_plus"; RUNS.mkdir(exist_ok=True)

def maybe_load_asset_returns():
    p = RUNS/"asset_returns.csv"
    if p.exists():
        try:    A = np.loadtxt(p, delimiter=",")
        except: A = np.loadtxt(p, delimiter=",", skiprows=1)
        if A.ndim == 1: A = A.reshape(-1,1)
        return A
    # build from data/*.csv
    series = []
    files = sorted(DATA.glob("*.csv"))
    for fp in files:
        try:
            with open(fp) as f:
                r = csv.DictReader(f)
                closes = []
                for row in r:
                    c = row.get("Adj Close") or row.get("Close") or row.get("close")
                    closes.append(float(c))
            c = np.array(closes, float)
            if len(c) > 5:
                series.append(np.diff(c)/ (c[:-1]+1e-12))
        except: pass
    if not series: return None
    T = min(len(s) for s in series)
    series = [s[-T:] for s in series]
    return np.stack(series, axis=1)

def append_card(title, html):
    for name in ["report_all.html","report_best_plus.html","report_plus.html"]:
        rp = ROOT/name
        if not rp.exists(): continue
        txt = rp.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        rp.write_text(txt.replace("</body>", card+"</body>") if "</body>" in txt else txt+card, encoding="utf-8")

if __name__ == "__main__":
    A = maybe_load_asset_returns()
    if A is None:
        print("(!) No asset returns found; skipping."); raise SystemExit(0)
    W = risk_parity_weights(A, window=63)
    np.savetxt(RUNS/"risk_parity_weights.csv", W, delimiter=",")
    append_card("Risk-Parity Sleeve ✔", f"<p>Wrote risk_parity_weights.csv  (T={W.shape[0]}, N={W.shape[1]})</p>")
    print(f"✅ Wrote {RUNS/'risk_parity_weights.csv'}")
