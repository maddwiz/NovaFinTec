
import numpy as np, pandas as pd
from .dna import rolling_drift, drift_velocity

def ema(a: pd.Series, span: int):
    return a.ewm(span=span, adjust=False).mean()

def dna_signal(close: pd.Series, window=64, topk=32):
    drift = pd.Series(rolling_drift(close.values, window=window, topk=topk), index=close.index)
    vel = pd.Series(drift_velocity(drift.values), index=close.index)
    sig = np.sign(-vel)  # falling drift -> +1, rising -> -1
    return sig.fillna(0.0).astype(float), drift, vel

def trend_signal(close: pd.Series, fast=10, slow=40):
    f = ema(close, fast); s = ema(close, slow)
    sig = np.sign(f - s)
    return sig.fillna(0.0).astype(float)

def momentum_signal(close: pd.Series, look=20):
    r = np.log(close).diff(look)
    sig = np.sign(r).fillna(0.0).astype(float)
    return sig

def simple_ensemble(signals: dict, weights: dict=None):
    keys = list(signals.keys())
    n = len(keys)
    if weights is None:
        weights = {k: 1.0/n for k in keys}
    s = None
    for k in keys:
        s = signals[k]*weights.get(k,0) if s is None else s + signals[k]*weights.get(k,0)
    return np.sign(s).astype(float)
