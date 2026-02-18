"""Adaptive signal decomposition using simplified empirical mode decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _find_extrema(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    if n < 3:
        return np.array([], dtype=int), np.array([], dtype=int)
    dx = np.diff(np.asarray(x, float))
    sign_dx = np.sign(dx)
    for i in range(1, len(sign_dx)):
        if sign_dx[i] == 0:
            sign_dx[i] = sign_dx[i - 1]
    sign_changes = np.diff(sign_dx)
    maxima = np.where(sign_changes < 0)[0] + 1
    minima = np.where(sign_changes > 0)[0] + 1
    return maxima, minima


def _envelope_mean(x: np.ndarray) -> np.ndarray:
    maxima, minima = _find_extrema(x)
    n = len(x)
    t = np.arange(n)

    if len(maxima) < 2 or len(minima) < 2:
        return np.zeros(n, dtype=float)

    upper = np.interp(t, maxima, x[maxima])
    lower = np.interp(t, minima, x[minima])
    return (upper + lower) / 2.0


def empirical_mode_decomposition(
    x: np.ndarray,
    max_imfs: int = 5,
    max_sift_iterations: int = 10,
    sift_threshold: float = 0.2,
) -> list[np.ndarray]:
    """Extract IMFs using a dependency-free EMD approximation."""
    arr = np.asarray(x, dtype=float).copy()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(arr)
    if n < 10:
        return [arr.copy()]

    imfs: list[np.ndarray] = []
    residual = arr.copy()

    for _ in range(max_imfs):
        h = residual.copy()
        for _ in range(max_sift_iterations):
            m = _envelope_mean(h)
            h_new = h - m
            diff = np.sum((h_new - h) ** 2) / (np.sum(h ** 2) + 1e-12)
            h = h_new
            if diff < float(sift_threshold):
                break

        zero_crossings = np.sum(np.abs(np.diff(np.sign(h))) > 0)
        maxima, minima = _find_extrema(h)
        extrema_count = len(maxima) + len(minima)

        if zero_crossings < 2 or extrema_count < 2:
            break

        imfs.append(h)
        residual = residual - h

        if np.std(residual) < 1e-8 * max(np.std(arr), 1e-12):
            break

    imfs.append(residual)
    return imfs


def estimate_dominant_period(imf: np.ndarray, min_period: int = 3) -> int:
    sign = np.sign(np.asarray(imf, float))
    crossings = np.where(np.abs(np.diff(sign)) > 0)[0]
    if len(crossings) < 2:
        return max(int(min_period), int(len(imf)))
    avg_half_period = float(len(imf)) / float(len(crossings))
    period = int(max(min_period, round(2.0 * avg_half_period)))
    return max(1, period)


def adaptive_momentum(imf: np.ndarray, period: int) -> np.ndarray:
    lookback = max(5, int(period) // 2)
    s = pd.Series(np.asarray(imf, float))
    slope = s.diff(lookback) / (lookback + 1e-12)
    mu = slope.rolling(int(max(5, period)), min_periods=max(5, int(period) // 4)).mean()
    sd = slope.rolling(int(max(5, period)), min_periods=max(5, int(period) // 4)).std(ddof=1).replace(0.0, np.nan)
    z = ((slope - mu) / (sd + 1e-12)).clip(-4.0, 4.0).fillna(0.0)
    return z.values.astype(float)


def adaptive_meanrev(imf: np.ndarray, period: int) -> np.ndarray:
    p = int(max(5, period))
    s = pd.Series(np.asarray(imf, float))
    deviation = s - s.rolling(p, min_periods=max(5, p // 4)).mean()
    sd = s.rolling(p, min_periods=max(5, p // 4)).std(ddof=1).replace(0.0, np.nan)
    z = (deviation / (sd + 1e-12)).clip(-4.0, 4.0).fillna(0.0)
    return (-z).values.astype(float)


def decompose_and_signal(
    close: np.ndarray,
    max_imfs: int = 5,
    trend_period_threshold: int = 126,
) -> dict:
    """Decompose price into IMFs and emit adaptive trend/cycle/composite signals."""
    px = np.maximum(np.asarray(close, float).ravel(), 1e-12)
    if px.size == 0:
        z = np.zeros(0, dtype=float)
        return {
            "trend_signal": z,
            "cycle_signal": z,
            "composite": z,
            "imf_periods": [],
            "imf_count": 0,
            "trend_weight_mean": 0.5,
            "cycle_weight_mean": 0.5,
        }

    logp = np.log(px)
    imfs = empirical_mode_decomposition(logp, max_imfs=max_imfs)

    trend_signals: list[tuple[np.ndarray, float]] = []
    cycle_signals: list[tuple[np.ndarray, float]] = []
    periods: list[int] = []

    for i, imf in enumerate(imfs):
        period = estimate_dominant_period(imf)
        periods.append(period)

        if i == len(imfs) - 1:
            sig = adaptive_momentum(imf, max(period, 63))
            trend_signals.append((sig, 1.0))
        elif period >= int(trend_period_threshold):
            sig = adaptive_momentum(imf, period)
            weight = min(2.0, period / 63.0)
            trend_signals.append((sig, float(weight)))
        else:
            sig = adaptive_meanrev(imf, period)
            weight = min(2.0, 63.0 / max(period, 5))
            cycle_signals.append((sig, float(weight)))

    n = len(logp)
    trend = np.zeros(n, dtype=float)
    cycle = np.zeros(n, dtype=float)
    trend_total_w = 0.0
    cycle_total_w = 0.0

    for sig, w in trend_signals:
        s = np.nan_to_num(sig[-n:], nan=0.0)
        if len(s) == n:
            trend += float(w) * s
            trend_total_w += float(w)

    for sig, w in cycle_signals:
        s = np.nan_to_num(sig[-n:], nan=0.0)
        if len(s) == n:
            cycle += float(w) * s
            cycle_total_w += float(w)

    if trend_total_w > 0:
        trend /= trend_total_w
    if cycle_total_w > 0:
        cycle /= cycle_total_w

    trend_strength = pd.Series(np.abs(trend)).rolling(21, min_periods=5).mean().fillna(0.5).values
    cycle_strength = pd.Series(np.abs(cycle)).rolling(21, min_periods=5).mean().fillna(0.5).values
    total = trend_strength + cycle_strength + 1e-12
    trend_weight = trend_strength / total
    cycle_weight = cycle_strength / total

    composite = trend_weight * np.tanh(trend) + cycle_weight * np.tanh(cycle)

    return {
        "trend_signal": np.tanh(trend),
        "cycle_signal": np.tanh(cycle),
        "composite": np.clip(composite, -1.0, 1.0),
        "imf_periods": periods,
        "imf_count": len(imfs),
        "trend_weight_mean": float(np.mean(trend_weight)),
        "cycle_weight_mean": float(np.mean(cycle_weight)),
    }
