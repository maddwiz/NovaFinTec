from __future__ import annotations

import numpy as np


def _safe_1d(x) -> np.ndarray:
    a = np.asarray(x, float).ravel()
    if a.size == 0:
        return np.zeros(0, dtype=float)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def fit_empirical_calibrator(
    confidence: np.ndarray,
    outcome: np.ndarray,
    n_bins: int = 10,
    min_count: int = 24,
) -> dict:
    """
    Fit a monotone empirical confidence->hit-rate mapping.
    confidence in [0,1], outcome in {0,1}.
    """
    c = np.clip(_safe_1d(confidence), 0.0, 1.0)
    y = np.clip(_safe_1d(outcome), 0.0, 1.0)
    L = min(len(c), len(y))
    c = c[:L]
    y = y[:L]
    if L <= 0:
        return {
            "bin_centers": [0.0, 1.0],
            "bin_hit_rate": [0.5, 0.5],
            "counts": [0, 0],
            "samples": 0,
            "mean_hit_rate": 0.5,
        }

    ord_idx = np.argsort(c)
    cs = c[ord_idx]
    ys = y[ord_idx]

    bins = int(max(3, min(30, n_bins)))
    target = max(int(min_count), int(np.ceil(L / bins)))

    centers = []
    hit = []
    cnt = []
    i = 0
    while i < L:
        j = min(L, i + target)
        cbin = cs[i:j]
        ybin = ys[i:j]
        if len(cbin) == 0:
            break
        centers.append(float(np.mean(cbin)))
        hit.append(float(np.mean(ybin)))
        cnt.append(int(len(cbin)))
        i = j

    if len(centers) == 1:
        centers = [0.0, centers[0], 1.0]
        hit = [hit[0], hit[0], hit[0]]
        cnt = [0, cnt[0], 0]

    x = np.asarray(centers, float)
    h = np.asarray(hit, float)
    if len(h) >= 2:
        # Enforce monotone non-decreasing hit with confidence.
        h = np.maximum.accumulate(h)
        h = np.clip(h, 0.0, 1.0)

    return {
        "bin_centers": x.tolist(),
        "bin_hit_rate": h.tolist(),
        "counts": cnt,
        "samples": int(L),
        "mean_hit_rate": float(np.mean(y)) if L else 0.5,
    }


def apply_empirical_calibrator(confidence: np.ndarray, calibrator: dict) -> np.ndarray:
    c = np.clip(_safe_1d(confidence), 0.0, 1.0)
    if len(c) == 0:
        return c
    x = np.asarray((calibrator or {}).get("bin_centers", [0.0, 1.0]), float)
    y = np.asarray((calibrator or {}).get("bin_hit_rate", [0.5, 0.5]), float)
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return np.full_like(c, 0.5, dtype=float)
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    ord_idx = np.argsort(x)
    x = x[ord_idx]
    y = y[ord_idx]
    return np.clip(np.interp(c, x, y), 0.0, 1.0)


def reliability_governor_from_calibrated(
    calibrated_confidence: np.ndarray,
    lo: float = 0.70,
    hi: float = 1.18,
    smooth: float = 0.85,
) -> np.ndarray:
    c = np.clip(_safe_1d(calibrated_confidence), 0.0, 1.0)
    if len(c) == 0:
        return c
    # Baseline 0.5 hit rate => neutral governor around the midpoint.
    edge = np.clip((c - 0.5) / 0.30, -1.0, 1.0)
    g = ((lo + hi) * 0.5) + 0.5 * (hi - lo) * edge
    g = np.clip(g, min(lo, hi), max(lo, hi))

    a = float(np.clip(smooth, 0.0, 0.99))
    if a > 0.0 and len(g) > 1:
        out = g.copy()
        for t in range(1, len(out)):
            out[t] = a * out[t - 1] + (1.0 - a) * out[t]
        g = out
    return np.clip(g, min(lo, hi), max(lo, hi))
