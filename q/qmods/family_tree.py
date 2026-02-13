import numpy as np, pandas as pd
from .dna import fft_topk_dna, dna_distance

def family_edges(close_a: pd.Series, close_b: pd.Series, step=63):
    """
    Build coarse similarity edges between two assets:
      - take every 'step' day snapshot DNA for A and B
      - compute 1 - distance as similarity
      - return list of edges with date and similarity
    """
    idx = sorted(set(close_a.index.date) & set(close_b.index.date))
    if not idx: return []
    edges = []
    # sample every 'step' days on shared calendar
    idx = idx[::step]
    for d in idx:
        # nearest data on/before date
        a = close_a.loc[:str(d)]
        b = close_b.loc[:str(d)]
        if a.empty or b.empty: continue
        dna_a = fft_topk_dna(a.values)
        dna_b = fft_topk_dna(b.values)
        sim = 1.0 - (dna_distance(dna_a, dna_b) or 0.0)
        edges.append({"date": str(d), "a": "A", "b": "B", "similarity": float(sim)})
    return edges
