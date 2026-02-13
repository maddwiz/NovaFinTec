#!/usr/bin/env python3
import numpy as np

def reflex_health(latent_returns: np.ndarray, lookback=126):
    r = np.asarray(latent_returns, float)
    out = np.zeros_like(r, float)
    for i in range(len(r)):
        j = max(0, i - lookback + 1)
        w = r[j:i+1]
        mu = np.nanmean(w)
        sd = np.nanstd(w) + 1e-12
        out[i] = max(0.0, (mu / sd) * np.sqrt(252.0))
    return out

def gate_reflex(reflex_signal: np.ndarray, health: np.ndarray, min_h=0.5):
    s = np.asarray(reflex_signal, float)
    h = np.asarray(health, float)
    scale = np.clip(h / (min_h + 1e-12), 0.0, 1.0)
    L = min(len(s), len(scale))
    return s[:L] * scale[:L]
