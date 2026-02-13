#!/usr/bin/env python3
# Cluster Caps: cap total absolute weight per cluster.
# Reads:
#   runs_plus/portfolio_weights.csv OR portfolio_weights.csv    [T x N]
#   runs_plus/cluster_map.csv   [N] optional: integers/labels per asset
# Writes:
#   runs_plus/weights_cluster_capped.csv  [T x N]
#   runs_plus/cluster_caps_info.json
# Appends a card.

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"; RUNS.mkdir(exist_ok=True)

def load_matrix(rel):
    p = ROOT/rel
    if not p.exists(): return None
    try:
        a = np.loadtxt(p, delimiter=",")
    except:
        a = np.loadtxt(p, delimiter=",", skiprows=1)
    if a.ndim == 1: a = a.reshape(-1,1)
    return a

def first_matrix(paths):
    for rel in paths:
        a = load_matrix(rel)
        if a is not None: return a
    return None

def load_cluster_map():
    p = RUNS/"cluster_map.csv"
    if not p.exists(): return None
    try:
        raw = np.loadtxt(p, delimiter=",", dtype=str)
    except:
        raw = np.loadtxt(p, delimiter=",", dtype=str, skiprows=1)
    if raw.ndim == 0: raw = np.array([str(raw)])
    return raw

def append_card(title, html):
    for name in ["report_all.html","report_best_plus.html","report_plus.html","report.html"]:
        f = ROOT/name
        if not f.exists(): continue
        txt = f.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        f.write_text(txt.replace("</body>", card+"</body>") if "</body>" in txt else txt+card, encoding="utf-8")

def cap_by_cluster(W, clusters, cap=0.20):
    # W: [T,N], clusters: [N] labels
    Wc = W.copy()
    labels = np.array(clusters, dtype=str)
    uniq = sorted(set(labels.tolist()))
    for t in range(Wc.shape[0]):
        for g in uniq:
            idx = np.where(labels == g)[0]
            if idx.size == 0: continue
            tot = float(np.sum(np.abs(Wc[t, idx])))
            if tot > cap + 1e-12:
                scale = cap / (tot + 1e-12)
                Wc[t, idx] *= scale
    return Wc

if __name__ == "__main__":
    # SAFE: pick the first that exists (no 'or' on arrays)
    W = first_matrix(["runs_plus/portfolio_weights.csv","portfolio_weights.csv"])
    if W is None:
        print("(!) No portfolio_weights found; skipping."); raise SystemExit(0)

    N = W.shape[1]
    clusters = load_cluster_map()
    if clusters is None or len(clusters) != N:
        clusters = np.array(["all"]*N, dtype=str)
        msg = "cluster_map.csv missing → using single cluster"
    else:
        msg = "cluster_map.csv loaded"

    CAP = 0.20  # 20% per cluster
    Wc = cap_by_cluster(W, clusters, cap=CAP)
    np.savetxt(RUNS/"weights_cluster_capped.csv", Wc, delimiter=",")

    info = {"cap": CAP, "clusters_note": msg, "T": int(W.shape[0]), "N": int(N)}
    (RUNS/"cluster_caps_info.json").write_text(json.dumps(info, indent=2))

    counts = defaultdict(int)
    for c in clusters: counts[str(c)] += 1
    top = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:8]
    bullets = "".join([f"<li>{k}: {v} assets</li>" for k,v in top])

    html = f"<p>{msg}. Cap={CAP:.2f}. Saved weights_cluster_capped.csv (T={W.shape[0]}, N={N}).</p><ul>{bullets}</ul>"
    append_card("Cluster Caps ✔", html)
    print(f"✅ Wrote runs_plus/weights_cluster_capped.csv ({msg})")
