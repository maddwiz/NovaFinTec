#!/usr/bin/env python3
import numpy as np

def tail_blend(base: np.ndarray, hedges: list, weights: list[float]):
    """
    Blend multiple hedge overlays into 'base' weight path.
    base: [T,N], hedges: list of [T,N], weights sum <= 1.0 (hedge budget).
    """
    base = np.asarray(base, float)
    if not hedges or not weights:
        return base
    H = np.zeros_like(base)
    for w, h in zip(weights, hedges):
        H += float(w) * np.asarray(h, float)
    total = min(1.0, max(0.0, sum(weights)))
    return (1.0 - total) * base + H
