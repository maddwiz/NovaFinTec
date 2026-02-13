#!/usr/bin/env python3
import numpy as np

def arb_weights(hive_scores: dict, alpha=2.0, drawdown_penalty: dict | None = None, disagreement_penalty: dict | None = None):
    """
    Softmax allocation over hives based on standardized health scores with penalties.
    hive_scores: {name: [T]} base score (higher=better)
    drawdown_penalty: {name: [T]} penalty in [0,1], where 1 is worst
    disagreement_penalty: {name: [T]} penalty in [0,1], where 1 is worst
    Returns: (names, W) where W=[T,H] per-hive weights that sum to 1.
    """
    names = sorted(hive_scores.keys())
    S = np.stack([np.asarray(hive_scores[n], float) for n in names], axis=1)
    mu = S.mean(0, keepdims=True); sd = S.std(0, keepdims=True) + 1e-9
    Z = (S - mu) / sd

    if drawdown_penalty:
        D = np.stack([np.asarray(drawdown_penalty.get(n, np.zeros(S.shape[0])), float) for n in names], axis=1)
        Z = Z - 1.4 * np.clip(D, 0.0, 1.0)
    if disagreement_penalty:
        G = np.stack([np.asarray(disagreement_penalty.get(n, np.zeros(S.shape[0])), float) for n in names], axis=1)
        Z = Z - 1.0 * np.clip(G, 0.0, 1.0)

    W = np.exp(alpha * Z)
    W = W / (W.sum(1, keepdims=True) + 1e-9)
    return names, W
