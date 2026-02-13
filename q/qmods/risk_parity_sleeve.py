#!/usr/bin/env python3
import numpy as np

def risk_parity_weights(returns: np.ndarray, window=63, eps=1e-8):
    R = np.asarray(returns, float)
    T, N = R.shape
    W = np.zeros_like(R)
    for i in range(T):
        j = max(0, i - window + 1)
        if i == 0:
            cov = np.eye(N)
        else:
            cov = np.cov(R[j:i+1].T)
        iv = 1.0 / (np.sqrt(np.clip(np.diag(cov), eps, None)))
        w = iv / (np.sum(np.abs(iv)) + eps)
        W[i] = w
    return W
