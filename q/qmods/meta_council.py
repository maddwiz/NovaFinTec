# qmods/meta_council.py — robust base signals + Meta v2 channels
import numpy as np
import pandas as pd

EPS = 1e-9

def _zscore(x: pd.Series, win: int) -> pd.Series:
    m = x.rolling(win, min_periods=max(10, win//5)).mean()
    s = x.rolling(win, min_periods=max(10, win//5)).std(ddof=1)
    z = (x - m) / (s + EPS)
    return z.replace([np.inf, -np.inf], 0.0).fillna(0.0)

def _slope_z(logp: pd.Series, win: int) -> pd.Series:
    n = len(logp)
    if n == 0:
        return pd.Series([], dtype=float, index=logp.index)
    t = pd.Series(np.arange(n, dtype=float), index=logp.index)
    cov = (logp * t).rolling(win, min_periods=max(10, win//5)).mean() - \
          logp.rolling(win, min_periods=max(10, win//5)).mean() * \
          t.rolling(win, min_periods=max(10, win//5)).mean()
    var = t.rolling(win, min_periods=max(10, win//5)).var(ddof=1) + EPS
    slope = cov / var
    return _zscore(slope, win=max(20, win//2))

# ==== v1 core channels ====

def momentum_signal(close: pd.Series) -> np.ndarray:
    """Trend/momentum: z-scored 63d OLS slope on log price."""
    logp = np.log(close.clip(lower=EPS))
    sig = _slope_z(logp, win=63).clip(-5, 5)
    return sig.to_numpy()

def meanrev_signal(close: pd.Series) -> np.ndarray:
    """Mean-reversion: negative z-scored 5d return vs 63d baseline."""
    logp = np.log(close.clip(lower=EPS))
    r5  = logp.diff(5)
    z   = _zscore(r5, win=63).clip(-5, 5)
    return (-z).to_numpy()

def carry_signal(close: pd.Series) -> np.ndarray:
    """Carry-style: normalized MA gap (21 vs 252) on level."""
    s = close.astype(float)
    ma_s = s.rolling(21, min_periods=10).mean()
    ma_l = s.rolling(252, min_periods=40).mean()
    gap  = (ma_s - ma_l) / (ma_l.abs() + EPS)
    z    = _zscore(gap, win=126).clip(-5, 5)
    return z.to_numpy()

# ==== v2 extra channels (not yet weighted by default) ====

def volatility_breakout(close: pd.Series) -> np.ndarray:
    """Return / rolling-vol breakout (z-scored)."""
    r = np.r_[0.0, np.diff(np.log(np.maximum(close.values, 1e-12)))]
    vol = pd.Series(r, index=close.index).rolling(63, min_periods=21).std(ddof=1)
    z = (pd.Series(r, index=close.index) / (vol + EPS)).clip(-5, 5).fillna(0.0)
    return z.to_numpy()

def trend_persistence(close: pd.Series) -> np.ndarray:
    """Sign-persistence of daily returns over ~1M window."""
    r1 = np.sign(np.r_[0.0, np.diff(np.log(np.maximum(close.values, 1e-12)))])
    score = pd.Series(r1, index=close.index).rolling(21, min_periods=7).mean().clip(-1, 1).fillna(0.0)
    return score.to_numpy()

def meta_v2_bundle(close: pd.Series) -> dict:
    """Convenience bundle of all channels."""
    return {
        "mom":   momentum_signal(close),
        "mr":    meanrev_signal(close),
        "carry": carry_signal(close),
        "vbo":   volatility_breakout(close),
        "tp":    trend_persistence(close),
    }
