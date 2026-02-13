import numpy as np, pandas as pd

def find_drift_alarms(drift_series: pd.Series, z=2.0, min_segs=100):
    """
    Simple spike detector on dna_drift_pct:
      - compute rolling mean/std
      - flag points where (x - mean)/std > z
    Returns list of dicts with date, value, zscore.
    """
    s = drift_series.copy()
    s = s.astype(float)
    s = s.where(np.isfinite(s))
    if s.dropna().shape[0] < min_segs:
        return []
    mu = s.rolling(63, min_periods=20).mean()
    sd = s.rolling(63, min_periods=20).std()
    zsc = (s - mu) / (sd.replace(0, np.nan))
    out = []
    for idx, val in s.items():
        zv = zsc.loc[idx]
        if np.isfinite(val) and np.isfinite(zv) and (zv >= z):
            out.append({"date": str(idx.date()), "drift": float(val), "z": float(zv)})
    return out
