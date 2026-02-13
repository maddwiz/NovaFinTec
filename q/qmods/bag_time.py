#!/usr/bin/env python3
import numpy as np

def time_bags(T, bag_size=0.7, bags=20, seed=42):
    rng = np.random.default_rng(seed)
    L = max(1, int(T * bag_size))
    idxs = []
    for _ in range(bags):
        s = int(rng.integers(0, max(1, T - L + 1)))
        idxs.append(np.arange(s, s+L))
    return idxs

def bag_mean(pred_matrix: np.ndarray):
    P = np.asarray(pred_matrix, float)
    return np.nanmean(P, axis=0)
