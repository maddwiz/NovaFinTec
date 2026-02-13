#!/usr/bin/env python3
# Align cluster_map.csv to the number of columns in your current weights.

import csv
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"; RUNS.mkdir(exist_ok=True)

def load_weights_first():
    for rel in ["runs_plus/portfolio_weights.csv",
                "runs_plus/weights_regime.csv",
                "runs_plus/weights_cluster_capped.csv",
                "runs_plus/weights_capped.csv",
                "portfolio_weights.csv"]:
        p = ROOT/rel
        if p.exists():
            try:
                a = np.loadtxt(p, delimiter=",")
            except:
                a = np.loadtxt(p, delimiter=",", skiprows=1)
            if a.ndim == 1: a = a.reshape(-1,1)
            return a, rel
    return None, None

if __name__ == "__main__":
    W, src = load_weights_first()
    if W is None:
        print("(!) No weights found; run your pipeline first."); raise SystemExit(0)
    N = int(W.shape[1])

    cmap = RUNS/"cluster_map.csv"
    if not cmap.exists():
        print("(!) runs_plus/cluster_map.csv not found."); raise SystemExit(0)

    rows = []
    with cmap.open() as f:
        rd = csv.reader(f)
        data = list(rd)
    if not data:
        print("(!) cluster_map.csv empty."); raise SystemExit(0)

    header = data[0]
    body = data[1:] if len(data)>1 and "cluster" in ",".join(header).lower() else data
    # normalize to [asset,cluster]
    norm = []
    for r in body:
        if len(r)>=2:
            norm.append([r[0].strip(), r[1].strip()])
        elif len(r)==1:
            norm.append([r[0].strip(), "all"])

    # Trim or pad to N
    if len(norm) > N:
        norm = norm[:N]
        note = f"trimmed to N={N} from {len(body)}"
    elif len(norm) < N:
        deficit = N - len(norm)
        for i in range(deficit):
            norm.append([f"A{len(norm)+1}", "all"])
        note = f"padded to N={N} from {len(body)}"
    else:
        note = f"matched N={N}"

    outp = RUNS/"cluster_map.csv"
    with outp.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["asset","cluster"])
        wr.writerows(norm)

    print(f"✅ Aligned cluster_map.csv ({note}) using weights from {src}")
