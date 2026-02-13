from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_1d(x) -> np.ndarray:
    a = np.asarray(x, float).ravel()
    if a.size == 0:
        return np.zeros(0, dtype=float)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def _tanh_zscore(x: np.ndarray, win: int = 63) -> np.ndarray:
    a = _safe_1d(x)
    if a.size == 0:
        return a
    s = pd.Series(a)
    mu = s.rolling(int(max(3, win)), min_periods=max(6, int(win // 3))).mean()
    sd = s.rolling(int(max(3, win)), min_periods=max(6, int(win // 3))).std(ddof=1).replace(0.0, np.nan)
    z = (s - mu) / sd
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return np.tanh(z.values.astype(float))


def _rolling_signal_quality(signal: np.ndarray, returns: np.ndarray, win: int = 63) -> np.ndarray:
    """
    Signal efficacy proxy:
    - use lagged signal as position
    - compute rolling sharpe-like ratio
    - map into [0, 1] quality
    """
    s = _safe_1d(signal)
    r = _safe_1d(returns)
    L = min(len(s), len(r))
    if L <= 1:
        return np.zeros(L, dtype=float)
    s = s[-L:]
    r = r[-L:]
    pos = np.roll(s, 1)
    pos[0] = 0.0
    pnl = pos * r
    ser = pd.Series(pnl)
    mu = ser.rolling(int(max(5, win)), min_periods=max(8, int(win // 3))).mean()
    sd = ser.rolling(int(max(5, win)), min_periods=max(8, int(win // 3))).std(ddof=1).replace(0.0, np.nan)
    sh = (mu / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return np.clip(0.5 + 0.5 * np.tanh(sh.values.astype(float) / 1.5), 0.0, 1.0)


def _smooth(x: np.ndarray, alpha: float = 0.88) -> np.ndarray:
    a = _safe_1d(x)
    if len(a) <= 1:
        return a
    out = a.copy()
    k = float(np.clip(alpha, 0.0, 0.99))
    for t in range(1, len(out)):
        out[t] = k * out[t - 1] + (1.0 - k) * out[t]
    return out


def build_dream_coherence_governor(
    signals: dict[str, np.ndarray],
    returns: np.ndarray,
    lo: float = 0.70,
    hi: float = 1.15,
    smooth: float = 0.88,
) -> tuple[np.ndarray, dict]:
    """
    Blend dream/reflex/symbolic/council streams into a coherence governor.
    Returns:
      - governor series in [lo, hi]
      - diagnostics dict with component quality metrics
    """
    clean = {}
    for name, vec in (signals or {}).items():
        a = _safe_1d(vec)
        if len(a) > 0:
            clean[str(name)] = a

    ret = _safe_1d(returns)
    if len(ret) == 0:
        return np.zeros(0, dtype=float), {"status": "missing_returns", "signals": []}

    if not clean:
        gov = np.full(len(ret), 0.90, dtype=float)
        return gov, {
            "status": "no_signals",
            "signals": [],
            "length": int(len(ret)),
            "mean_coherence": 0.45,
            "mean_governor": float(np.mean(gov)),
        }

    L = min([len(ret)] + [len(v) for v in clean.values()])
    if L <= 0:
        return np.zeros(0, dtype=float), {"status": "empty_alignment", "signals": list(clean.keys())}

    ret = ret[-L:]
    names = sorted(clean.keys())
    mats = []
    for name in names:
        mats.append(_tanh_zscore(clean[name][-L:], win=63))
    M = np.column_stack(mats) if mats else np.zeros((L, 1), dtype=float)

    consensus = np.tanh(np.mean(M, axis=1))
    dispersion = np.std(M, axis=1)
    agreement = np.clip(1.0 - dispersion / 0.90, 0.0, 1.0)
    efficacy = _rolling_signal_quality(consensus, ret, win=63)

    ema = pd.Series(consensus).ewm(span=21, adjust=False).mean().values.astype(float)
    drift = np.abs(consensus - ema)
    stability = np.clip(1.0 - drift / 0.80, 0.0, 1.0)

    coherence = np.clip(0.45 * agreement + 0.40 * efficacy + 0.15 * stability, 0.0, 1.0)
    if M.shape[1] == 1:
        coherence = np.clip(0.85 * coherence, 0.0, 1.0)

    lo_f = float(min(lo, hi))
    hi_f = float(max(lo, hi))
    gov = lo_f + (hi_f - lo_f) * coherence
    gov = _smooth(np.clip(gov, lo_f, hi_f), alpha=smooth)
    gov = np.clip(gov, lo_f, hi_f)

    per_signal_corr = {}
    if len(consensus) >= 12:
        c = pd.Series(consensus)
        for j, n in enumerate(names):
            s = pd.Series(M[:, j])
            corr = c.corr(s)
            per_signal_corr[n] = float(corr) if np.isfinite(corr) else 0.0

    info = {
        "status": "ok",
        "signals": names,
        "length": int(L),
        "mean_agreement": float(np.mean(agreement)),
        "mean_efficacy": float(np.mean(efficacy)),
        "mean_stability": float(np.mean(stability)),
        "mean_coherence": float(np.mean(coherence)),
        "mean_governor": float(np.mean(gov)),
        "min_governor": float(np.min(gov)),
        "max_governor": float(np.max(gov)),
        "per_signal_consensus_corr": per_signal_corr,
    }
    return gov, info
