#!/usr/bin/env python3
import numpy as np

def neutralize(target: np.ndarray, to_neutralize: np.ndarray, strength=1.0):
    t = np.asarray(target, float)
    X = np.asarray(to_neutralize, float)
    if t.ndim == 1: 
        t = t[:, None]
    if X.ndim == 1: 
        X = X[:, None]
    X = X - np.nanmean(X, axis=0, keepdims=True)
    XtX = X.T @ X + 1e-6 * np.eye(X.shape[1])
    beta = np.linalg.pinv(XtX) @ (X.T @ t)
    resid = t - strength * (X @ beta)
    return resid.squeeze()
