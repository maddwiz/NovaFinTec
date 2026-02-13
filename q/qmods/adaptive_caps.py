#!/usr/bin/env python3
import numpy as np

def adaptive_cap(vol_series: np.ndarray, cap_min=0.05, cap_max=0.15):
    v = np.asarray(vol_series, float)
    z = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
    u = 1.0 / (1.0 + np.exp(-z))              # 0..1
    return cap_min + (cap_max - cap_min) * (1.0 - u)  # higher vol -> smaller cap

def apply_caps(weights_t: np.ndarray, caps_t: np.ndarray):
    W = np.asarray(weights_t, float).copy()
    C = np.asarray(caps_t, float).ravel()
    T = min(W.shape[0], len(C))
    for t in range(T):
        cap = float(C[t])
        W[t] = np.clip(W[t], -cap, cap)
    return W
