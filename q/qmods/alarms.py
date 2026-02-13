# qmods/alarms.py — simple DNA-drift alarms
from pathlib import Path
import json, numpy as np, pandas as pd
from .io import load_close
from .drift import rolling_dna_drift

def make_alarms(csv_path: Path, lookback=126, z_thresh=2.5):
    try:
        s = load_close(csv_path)
    except Exception:
        return []
    d = rolling_dna_drift(s, lookback).dropna()
    if d.empty:
        return []
    z = (d - d.mean())/(d.std(ddof=1)+1e-9)
    spikes = z[z > z_thresh]
    if spikes.empty:
        return []
    alarms = []
    for ts, zval in spikes.iloc[-5:].items():
        alarms.append({
            "type": "dna_drift_spike",
            "date": ts.strftime("%Y-%m-%d"),
            "z": float(round(zval, 2)),
            "msg": f"DNA drift spike z={round(zval,2)}"
        })
    return alarms
