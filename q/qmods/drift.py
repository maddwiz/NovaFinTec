# qmods/drift.py — rolling DNA drift
import numpy as np
import pandas as pd
from .dna import fft_topk_dna, dna_distance

def rolling_dna_drift(close: pd.Series, window: int = 126):
    dnas = []
    vals = close.values
    for i in range(len(vals)):
        lo = max(0, i - window + 1)
        window_slice = vals[lo:i+1]
        # robust DNA (no k arg)
        dnas.append(fft_topk_dna(window_slice))
    # drift = distance to 1-step-lag DNA
    dist = [np.nan]
    for i in range(1, len(dnas)):
        dist.append(dna_distance(dnas[i-1], dnas[i]))
    return pd.Series(dist, index=close.index).astype(float)
