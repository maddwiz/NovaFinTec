#!/usr/bin/env python3
import numpy as np


def _as_time_array(x, T: int, lo: float, hi: float, default: float) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return np.full(T, float(np.clip(float(arr), lo, hi)), dtype=float)
    arr = arr.ravel()
    if arr.size == 0:
        return np.full(T, float(default), dtype=float)
    if arr.size < T:
        pad = np.full(T - arr.size, float(arr[-1]), dtype=float)
        arr = np.concatenate([arr, pad], axis=0)
    if arr.size > T:
        arr = arr[:T]
    arr = np.nan_to_num(arr, nan=default, posinf=default, neginf=default)
    return np.clip(arr, lo, hi).astype(float)


def _entropy_norm(w: np.ndarray) -> float:
    a = np.asarray(w, float).ravel()
    if len(a) <= 1:
        return 1.0
    a = np.clip(a, 0.0, None)
    s = float(np.sum(a))
    if s <= 0:
        return 1.0
    p = a / s
    h = -np.sum(np.where(p > 0.0, p * np.log(p), 0.0))
    return float(np.clip(h / np.log(len(p)), 0.0, 1.0))


def arb_weights(
    hive_scores: dict,
    alpha=2.0,
    drawdown_penalty: dict | None = None,
    disagreement_penalty: dict | None = None,
    downside_penalty: dict | None = None,
    crowding_penalty: dict | None = None,
    inertia: float = 0.80,
    max_weight: float = 0.70,
    min_weight: float = 0.00,
    entropy_target: float | None = None,
    entropy_strength: float = 0.0,
):
    """
    Softmax allocation over hives based on standardized health scores with penalties,
    plus practical execution constraints:
      - inertia smoothing over time
      - optional min/max per-hive weight clamps
      - optional time-varying alpha/inertia schedules
    hive_scores: {name: [T]} base score (higher=better)
    drawdown_penalty: {name: [T]} penalty in [0,1], where 1 is worst
    disagreement_penalty: {name: [T]} penalty in [0,1], where 1 is worst
    downside_penalty: {name: [T]} penalty in [0,1], where 1 is worst downside profile
    crowding_penalty: {name: [T]} penalty in [0,1], where 1 is highest cross-hive crowding/correlation
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
    if downside_penalty:
        U = np.stack([np.asarray(downside_penalty.get(n, np.zeros(S.shape[0])), float) for n in names], axis=1)
        Z = Z - 1.1 * np.clip(U, 0.0, 1.0)
    if crowding_penalty:
        C = np.stack([np.asarray(crowding_penalty.get(n, np.zeros(S.shape[0])), float) for n in names], axis=1)
        Z = Z - 0.9 * np.clip(C, 0.0, 1.0)

    T = Z.shape[0]
    alpha_t = _as_time_array(alpha, T, lo=0.2, hi=10.0, default=2.0)
    inertia_t = _as_time_array(inertia, T, lo=0.0, hi=0.98, default=0.80)

    W = np.zeros_like(Z, dtype=float)
    for t in range(T):
        z = Z[t]
        z = z - np.max(z)
        e = np.exp(alpha_t[t] * z)
        s = float(np.sum(e))
        if s <= 0:
            wrow = np.full(e.shape[0], 1.0 / max(1, e.shape[0]), dtype=float)
        else:
            wrow = e / s

        # Clamp single-hive concentration and ensure small floor for exploration/recovery.
        mn = float(np.clip(min_weight, 0.0, 0.95))
        mx = float(np.clip(max_weight, mn + 1e-6, 1.0))
        if mn > 0.0 or mx < 0.999:
            wrow = np.clip(wrow, mn, mx)
            wrow = wrow / (wrow.sum() + 1e-9)

        if t > 0 and inertia_t[t] > 0.0:
            wrow = inertia_t[t] * W[t - 1] + (1.0 - inertia_t[t]) * wrow
            wrow = np.clip(wrow, 0.0, None)
            wrow = wrow / (wrow.sum() + 1e-9)

        # Entropy-aware diversification: if concentration gets too high,
        # blend toward uniform to keep cross-hive exploration alive.
        if entropy_target is not None:
            et = float(np.clip(entropy_target, 0.0, 1.0))
            es = float(np.clip(entropy_strength, 0.0, 1.0))
            if et > 0.0 and es > 0.0 and len(wrow) > 1:
                hcur = _entropy_norm(wrow)
                if hcur < et:
                    lam = es * (et - hcur) / max(et, 1e-9)
                    lam = float(np.clip(lam, 0.0, es))
                    uni = np.full_like(wrow, 1.0 / len(wrow))
                    wrow = (1.0 - lam) * wrow + lam * uni
                    wrow = np.clip(wrow, mn, mx)
                    wrow = wrow / (wrow.sum() + 1e-9)

        W[t] = wrow
    return names, W
