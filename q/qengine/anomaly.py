import numpy as np
import pandas as pd

def _zscore(s: pd.Series, eps=1e-9):
    s = s.astype(float)
    m = s.rolling(60, min_periods=1).mean()
    sd = s.rolling(60, min_periods=1).std().fillna(0.0)
    return (s - m) / (sd + eps)

def anomaly_triage(close: pd.Series, drift: pd.Series, z_th: float = 4.0) -> pd.Series:
    r = np.log(close).diff().fillna(0.0)
    rz = _zscore(r).fillna(0.0)
    dz = _zscore(drift.fillna(0.0)).fillna(0.0)
    rev = (np.sign(r.shift(-1).fillna(0.0)) == -np.sign(r))
    glitch = (rz.abs() > z_th) & rev
    unconfirmed_drift = (dz.abs() > z_th) & (rz.abs() < 1.0)
    quarantine = (glitch | unconfirmed_drift).astype(bool)
    quarantine.name = "quarantine"
    return quarantine

def apply_quarantine(position: pd.Series, quarantine_mask: pd.Series, scale: float = 0.0) -> pd.Series:
    q = quarantine_mask.reindex(position.index).fillna(False)
    return position * (1.0 - (1.0 - scale) * q.astype(float))
