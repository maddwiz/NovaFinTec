#!/usr/bin/env python3
# Builds runs_plus/cluster_map.csv with guessed asset names.
# Sources (in this order):
#   1) runs_plus/asset_names.csv           (one name per line OR CSV header row)
#   2) data/*.csv and data_new/*.csv       (file stems as names)
#   3) runs_plus/portfolio_weights.csv     (fallback to A1..AN)
#
# Output:
#   runs_plus/cluster_map.csv with two columns:
#       asset,cluster
#   default cluster is "all" — edit this file to set real groups (e.g., Rates, Equities, FX, Metals, Crypto)
#
# Also appends a small card to report_* with how many assets were written.

import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"; RUNS.mkdir(exist_ok=True)
DATA_DIRS = [ROOT/"data", ROOT/"data_new"]

def read_asset_names_csv(p: Path):
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore").strip()
        if not txt:
            return None
        # If it's a header row: "A,B,C,..."
        if "," in txt and "\n" not in txt:
            cols = [c.strip() for c in txt.split(",") if c.strip()]
            return cols if cols else None
        # Else assume one name per line
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        # If a single line with commas, split it
        if len(lines) == 1 and "," in lines[0]:
            cols = [c.strip() for c in lines[0].split(",") if c.strip()]
            return cols if cols else None
        return lines if lines else None
    except Exception:
        return None

def read_from_data_dirs():
    names = []
    for d in DATA_DIRS:
        if not d.exists(): 
            continue
        for fp in sorted(d.glob("*.csv")):
            names.append(fp.stem)
    return list(dict.fromkeys(names))  # unique, keep order

def read_N_from_weights():
    for rel in ["runs_plus/portfolio_weights.csv", "portfolio_weights.csv"]:
        p = ROOT / rel
        if p.exists():
            try:
                a = np.loadtxt(p, delimiter=",")
            except Exception:
                try:
                    a = np.loadtxt(p, delimiter=",", skiprows=1)
                except Exception:
                    continue
            if a.ndim == 1:
                a = a.reshape(-1, 1)
            N = int(a.shape[1])
            return [f"A{i+1}" for i in range(N)]
    return None

def append_card(title, html):
    for name in ["report_all.html","report_best_plus.html","report_plus.html","report.html"]:
        page = ROOT / name
        if not page.exists(): 
            continue
        txt = page.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        page.write_text(txt.replace("</body>", card+"</body>") if "</body>" in txt else txt+card, encoding="utf-8")

if __name__ == "__main__":
    # 1) Try runs_plus/asset_names.csv
    names = read_asset_names_csv(RUNS/"asset_names.csv")

    # 2) Else try data dirs
    if not names:
        names = read_from_data_dirs()

    # 3) Else fall back to A1..AN from weights shape
    if not names:
        names = read_N_from_weights()

    if not names:
        print("(!) Could not infer asset names. Make sure data/*.csv or weights exist.")
        raise SystemExit(0)

    outp = RUNS/"cluster_map.csv"
    with outp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["asset","cluster"])
        for nm in names:
            w.writerow([nm, "all"])

    append_card("Cluster Map Template ✔", f"<p>Wrote cluster_map.csv with {len(names)} assets. Edit the 'cluster' column to set groups.</p>")
    print(f"✅ Wrote {outp}  (rows={len(names)})")
