#!/usr/bin/env python3
import numpy as np

class MetaStackV1:
    """
    Leakage-safe linear meta-learner over council outputs.
    fit(X,y): X=[T,K] (lagged), y=[T] realized returns.
    predict(X): returns [T].
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        self.mean_ = None

    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y).ravel()
        if X.shape[0] < 3 or X.ndim != 2:
            self.coef_ = np.ones(X.shape[1]) / max(1, X.shape[1])
            self.mean_ = np.zeros(X.shape[1])
            return self
        # 1-step lag to reduce leakage
        X = X[:-1]; y = y[1:]
        self.mean_ = X.mean(0, keepdims=True)
        Xc = X - self.mean_
        XtX = Xc.T @ Xc
        aI = self.alpha * np.eye(XtX.shape[0])
        self.coef_ = np.linalg.pinv(XtX + aI) @ (Xc.T @ y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        if self.coef_ is None:
            return np.mean(X, axis=1)
        Xc = X - (self.mean_ if self.mean_ is not None else X.mean(0, keepdims=True))
        return (Xc @ self.coef_).ravel()
