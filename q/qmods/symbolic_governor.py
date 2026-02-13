from __future__ import annotations

import numpy as np


def _safe_1d(x):
    a = np.asarray(x, float).ravel()
    if a.size == 0:
        return np.zeros(0, dtype=float)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def _smooth(x: np.ndarray, alpha: float = 0.88) -> np.ndarray:
    a = _safe_1d(x)
    if len(a) <= 1:
        return a
    k = float(np.clip(alpha, 0.0, 0.99))
    out = a.copy()
    for t in range(1, len(out)):
        out[t] = k * out[t - 1] + (1.0 - k) * out[t]
    return out


def build_symbolic_governor(
    sym_signal: np.ndarray,
    sym_affect: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    events_n: np.ndarray | None = None,
    lo: float = 0.72,
    hi: float = 1.12,
    smooth: float = 0.88,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Convert symbolic/affective state into:
      - symbolic stress [0,1]
      - symbolic governor [lo,hi]
    """
    s = _safe_1d(sym_signal)
    T = len(s)
    if T == 0:
        return np.zeros(0, float), np.zeros(0, float), {"status": "empty"}

    a = _safe_1d(sym_affect) if sym_affect is not None else np.zeros(T, float)
    c = _safe_1d(confidence) if confidence is not None else np.ones(T, float) * 0.5
    n = _safe_1d(events_n) if events_n is not None else np.zeros(T, float)
    if len(a) < T:
        x = np.zeros(T, float)
        x[: len(a)] = a
        a = x
    else:
        a = a[:T]
    if len(c) < T:
        x = np.zeros(T, float)
        x[: len(c)] = c
        c = x
    else:
        c = c[:T]
    if len(n) < T:
        x = np.zeros(T, float)
        x[: len(n)] = n
        n = x
    else:
        n = n[:T]

    c = np.clip(c, 0.0, 1.0)
    neg_bias = np.clip(-s, 0.0, 1.0)
    affect = np.clip(a, 0.0, 1.0)
    # Event intensity anomaly proxy.
    den = float(np.percentile(n, 90)) + 1e-9 if len(n) else 1.0
    inten = np.clip(n / den, 0.0, 2.0)
    inten = np.clip(inten / 1.25, 0.0, 1.0)

    stress = np.clip((0.45 * neg_bias + 0.35 * affect + 0.20 * inten) * (0.55 + 0.45 * c), 0.0, 1.0)
    stress = _smooth(stress, alpha=smooth)

    lo_f = float(min(lo, hi))
    hi_f = float(max(lo, hi))
    gov = hi_f - (hi_f - lo_f) * stress
    gov = np.clip(_smooth(gov, alpha=smooth), lo_f, hi_f)

    info = {
        "status": "ok",
        "length": int(T),
        "mean_stress": float(np.mean(stress)),
        "max_stress": float(np.max(stress)),
        "mean_governor": float(np.mean(gov)),
        "min_governor": float(np.min(gov)),
        "max_governor": float(np.max(gov)),
    }
    return stress, gov, info
