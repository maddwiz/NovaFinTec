
import numpy as np, pandas as pd

def crisis_anchor_from_vix(vix_close: pd.Series, high_thresh=30.0, super_thresh=50.0):
    v = vix_close.ffill().fillna(0.0)
    mult = pd.Series(1.0, index=v.index)
    mult[v >= high_thresh] = 0.5
    mult[v >= super_thresh] = 0.0
    return mult

def crisis_anchor_from_internal(drift: pd.Series, thresh=0.20):
    d = drift.fillna(0.0).abs()
    mult = pd.Series(1.0, index=d.index)
    mult[d > thresh] = 0.5
    return mult
