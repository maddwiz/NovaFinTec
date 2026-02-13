import json, pathlib
import numpy as np, pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = pathlib.Path("runs_plus")
FG   = RUNS/"family_graph.json"
OUT  = RUNS/"family_graph.png"

g = json.loads(FG.read_text()) if FG.exists() else {"nodes": [], "edges": []}
nodes = g.get("nodes", [])
edges = g.get("edges", [])

if not nodes:
    print("No nodes; did you build family_graph.json?"); raise SystemExit

# build similarity matrix
idx = {a:i for i,a in enumerate(nodes)}
M = np.eye(len(nodes))
for e in edges:
    i, j = idx.get(e["a"]), idx.get(e["b"])
    if i is None or j is None: continue
    sim = float(e.get("avg_similarity", 0.0))
    M[i, j] = sim
    M[j, i] = sim

fig = plt.figure(figsize=(5,4.5))
plt.imshow(M, vmin=0, vmax=1, aspect="equal")
plt.xticks(range(len(nodes)), nodes, rotation=45, ha="right")
plt.yticks(range(len(nodes)), nodes)
plt.title("DNA similarity (1 − distance)")
cbar = plt.colorbar(shrink=0.8)
cbar.set_label("similarity")
plt.tight_layout()
fig.savefig(OUT, dpi=140)
plt.close(fig)

print(f"Wrote {OUT}")
