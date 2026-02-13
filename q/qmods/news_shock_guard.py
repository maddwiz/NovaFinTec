#!/usr/bin/env python3
from __future__ import annotations

import numpy as np


def shock_mask(
    vol_series: np.ndarray,
    z: float = 2.5,
    min_len: int = 2,
    lookback: int = 63,
    cooldown: int = 3,
    quantile: float | None = 0.985,
):
    """
    Build a robust shock mask from volatility/absolute-return series.
    - rolling robust z-score (median/MAD)
    - optional quantile trigger
    - streak confirmation + cooldown persistence
    """
    v = np.asarray(vol_series, float).ravel()
    n = len(v)
    if n == 0:
        return np.zeros(0, dtype=int)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    lb = int(max(10, lookback))
    zz = np.zeros(n, dtype=float)
    qq = np.zeros(n, dtype=float)
    use_q = (quantile is not None)
    qv = float(np.clip(float(quantile), 0.80, 0.999)) if use_q else None

    for t in range(n):
        j = max(0, t - lb + 1)
        w = v[j : t + 1]
        med = float(np.median(w))
        mad = float(np.median(np.abs(w - med))) + 1e-9
        zz[t] = (v[t] - med) / (1.4826 * mad)
        if use_q:
            qq[t] = 1.0 if v[t] >= float(np.quantile(w, qv)) else 0.0

    raw = (zz >= float(z)).astype(int)
    if use_q:
        raw = np.maximum(raw, qq.astype(int))

    out = np.zeros(n, dtype=int)
    streak = 0
    cool = 0
    need = int(max(1, min_len))
    cd = int(max(0, cooldown))
    for i in range(n):
        if raw[i] > 0:
            streak += 1
        else:
            streak = 0

        if streak >= need:
            out[i] = 1
            cool = cd
            continue

        if cool > 0:
            out[i] = 1
            cool -= 1

    return out


def apply_shock_guard(signal: np.ndarray, mask: np.ndarray, alpha=0.5):
    s = np.asarray(signal, float)
    m = np.asarray(mask, int)
    L = min(len(s), len(m))
    a = float(np.clip(float(alpha), 0.0, 1.0))
    out = s.copy()
    out[:L] = s[:L] * (1.0 - a * np.clip(m[:L], 0, 1))
    return out
