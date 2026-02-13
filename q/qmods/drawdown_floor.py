#!/usr/bin/env python3
import numpy as np

def drawdown_floor_series(cum_pnl: np.ndarray, floor: float = -0.12, cut: float = 0.5):
    # cum_pnl: cumulative pnl path; scale=1 until DD < floor, then 'cut' until new high
    peak = -1e9
    scale = np.ones_like(cum_pnl, float)
    in_dd = False
    for i, x in enumerate(np.asarray(cum_pnl, float)):
        if x > peak:
            peak = x
            in_dd = False
        if x - peak < floor:
            in_dd = True
        scale[i] = (cut if in_dd else 1.0)
    return scale
