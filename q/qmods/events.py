import numpy as np, pandas as pd

def make_event_notes(drift: pd.Series, bpm: pd.Series,
                     drift_z: float = 2.0,
                     bpm_hot: float = 120.0,
                     bpm_calm: float = 70.0):
    """
    Simple rule-based event notes:
      - High DNA drift z-spikes -> 'regime shift' notes
      - Heartbeat high/low -> 'volatility spike' / 'calm regime'
    Returns a list of dicts with date + text.
    """
    notes = []
    # prep
    d = pd.Series(drift).astype(float)
    b = pd.Series(bpm).astype(float)
    mu = d.rolling(63, min_periods=20).mean()
    sd = d.rolling(63, min_periods=20).std()
    z  = (d - mu) / (sd.replace(0, np.nan))

    for idx in d.index:
        row = []
        # drift spike
        if np.isfinite(d.loc[idx]) and np.isfinite(z.loc[idx]) and z.loc[idx] >= drift_z:
            row.append(f"DNA drift spike (z≈{z.loc[idx]:.1f})")
        # heartbeat zones
        if np.isfinite(b.loc[idx]):
            if b.loc[idx] >= bpm_hot:
                row.append(f"Heartbeat {b.loc[idx]:.0f} bpm (volatility spike)")
            elif b.loc[idx] <= bpm_calm:
                row.append(f"Heartbeat {b.loc[idx]:.0f} bpm (calm regime)")
        if row:
            notes.append({"date": str(getattr(idx, "date", lambda: idx)()), "text": "; ".join(row)})
    return notes
