#!/usr/bin/env python3
import numpy as np

def shock_mask(vol_series: np.ndarray, z=2.5, min_len=2):
    v = np.asarray(vol_series, float)
    zed = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
    raw = (np.abs(zed) > z).astype(int)
    out = np.zeros_like(raw)
    streak = 0
    for i, r in enumerate(raw):
        streak = streak + 1 if r else 0
        out[i] = 1 if streak >= min_len else 0
    return out

def apply_shock_guard(signal: np.ndarray, mask: np.ndarray, alpha=0.5):
    s = np.asarray(signal, float)
    m = np.asarray(mask, int)
    L = min(len(s), len(m))
    out = s.copy()
    out[:L] = s[:L] * (1.0 - alpha * m[:L])
    return out
