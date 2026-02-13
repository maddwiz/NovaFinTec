#!/usr/bin/env python3
import numpy as np

def crisis_flag(vol_series: np.ndarray, z=2.0, min_len=3):
    v = np.asarray(vol_series, float)
    zed = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
    raw = (zed > z).astype(float)
    out = np.zeros_like(raw)
    streak = 0
    for i, f in enumerate(raw):
        streak = streak + 1 if f > 0 else 0
        out[i] = 1.0 if streak >= min_len else 0.0
    return out

def crisis_overlay(base_w: np.ndarray, hedge_w: np.ndarray, flag: np.ndarray, alpha=0.5):
    out = base_w.copy()
    T = min(len(flag), base_w.shape[0], hedge_w.shape[0])
    for t in range(T):
        if flag[t] > 0:
            out[t] = (1 - alpha) * base_w[t] + alpha * hedge_w[t]
    return out
