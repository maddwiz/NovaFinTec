#!/usr/bin/env python3
import numpy as np

def arb_weights(hive_scores: dict, alpha=2.0):
    """
    Softmax allocation over hives based on standardized health scores.
    hive_scores: {name: [T]}
    Returns: (names, W) where W=[T,H] per-hive weights that sum to 1.
    """
    names = sorted(hive_scores.keys())
    S = np.stack([np.asarray(hive_scores[n], float) for n in names], axis=1)
    mu = S.mean(0, keepdims=True); sd = S.std(0, keepdims=True) + 1e-9
    Z = (S - mu) / sd
    W = np.exp(alpha * Z); W = W / (W.sum(1, keepdims=True) + 1e-9)
    return names, W
