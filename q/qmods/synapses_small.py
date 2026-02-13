#!/usr/bin/env python3
import numpy as np

class SynapseSmall:
    """
    Tiny 1-hidden-layer MLP to fuse K council signals nonlinearly.
    Uses tanh + L2 reg; 1-step lag fit to reduce leakage.
    """
    def __init__(self, hidden=8, lr=0.01, reg=1e-3, epochs=200, seed=13):
        self.hidden = hidden; self.lr = lr; self.reg = reg; self.epochs = epochs
        self.rng = np.random.default_rng(seed)
        self.W1 = None; self.b1 = None; self.W2 = None; self.b2 = None

    def _init(self, K):
        self.W1 = self.rng.standard_normal((K, self.hidden)) / np.sqrt(K)
        self.b1 = np.zeros(self.hidden)
        self.W2 = self.rng.standard_normal((self.hidden, 1)) / np.sqrt(self.hidden)
        self.b2 = np.zeros(1)

    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y).reshape(-1,1)
        if X.ndim != 2 or X.shape[0] < 3:
            self._init(X.shape[1]); return self
        # lag
        X = X[:-1]; y = y[1:]
        T, K = X.shape
        self._init(K)
        for _ in range(self.epochs):
            Z1 = np.tanh(X @ self.W1 + self.b1)
            yhat = Z1 @ self.W2 + self.b2
            err = yhat - y
            dW2 = Z1.T @ err / T + self.reg * self.W2
            db2 = err.mean(0)
            dZ1 = err @ self.W2.T * (1 - Z1**2)
            dW1 = X.T @ dZ1 / T + self.reg * self.W1
            db1 = dZ1.mean(0)
            self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1
        return self

    def predict(self, X):
        X = np.asarray(X)
        Z1 = np.tanh(X @ self.W1 + self.b1)
        yhat = Z1 @ self.W2 + self.b2
        return yhat.ravel()
