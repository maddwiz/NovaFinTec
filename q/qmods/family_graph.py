# qmods/family_graph.py — safer family similarity across assets
import json, pathlib
import numpy as np
from .dna import fft_topk_dna, dna_distance
from .io import load_close

def build_family_graph(data_dir: pathlib.Path, assets,
                       step_days: int = 63,
                       min_window: int = 256,
                       out_path: pathlib.Path = pathlib.Path("runs_plus/family_graph.json")):
    # load closes with robust loader
    closes = {}
    for a in assets:
        try:
            s = load_close(data_dir / f"{a}.csv")
            closes[a] = s
        except Exception:
            continue

    if not closes:
        out_path.write_text(json.dumps({"nodes": [], "edges": []}, indent=2))
        return

    # shared calendar of available dates
    calendars = [set(s.index.date) for s in closes.values() if not s.empty]
    if not calendars:
        out_path.write_text(json.dumps({"nodes": list(closes.keys()), "edges": []}, indent=2))
        return
    shared = sorted(set.intersection(*calendars))
    if not shared:
        out_path.write_text(json.dumps({"nodes": list(closes.keys()), "edges": []}, indent=2))
        return

    dates = shared[::max(1, step_days)]
    edges = []
    aset = list(closes.keys())

    for i in range(len(aset)):
        for j in range(i+1, len(aset)):
            a, b = aset[i], aset[j]
            Sa, Sb = closes[a], closes[b]
            sims = []
            for d in dates:
                A = Sa.loc[:str(d)].values
                B = Sb.loc[:str(d)].values
                if A.size < min_window or B.size < min_window:
                    continue
                dna_a = fft_topk_dna(A[-min_window:])
                dna_b = fft_topk_dna(B[-min_window:])
                dist = dna_distance(dna_a, dna_b)
                if dist is not None and np.isfinite(dist):
                    sims.append(1.0 - dist)
            if sims:
                edges.append({
                    "a": a, "b": b,
                    "avg_similarity": float(np.mean(sims)),
                    "samples": len(sims)
                })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"nodes": aset, "edges": edges}, indent=2))
