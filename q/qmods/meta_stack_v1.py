#!/usr/bin/env python3
from __future__ import annotations

import numpy as np


def _safe_std(x, axis=0, eps=1e-8):
    s = np.nanstd(x, axis=axis)
    s = np.where(np.isfinite(s), s, 1.0)
    return np.maximum(s, eps)


def _annualized_sharpe(r):
    r = np.asarray(r, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size < 4:
        return -1e9
    mu = float(np.mean(r))
    sd = float(np.std(r) + 1e-12)
    return (mu / sd) * np.sqrt(252.0)


class MetaStackV1:
    """
    Leakage-safe ridge meta-learner over council outputs.
    - Uses 1-step lag (X[t-1] -> y[t]) to reduce leakage.
    - Chooses ridge alpha with rolling time-split CV.
    - Exposes confidence from margin vs residual noise.
    """

    def __init__(
        self,
        alpha=1.0,
        alphas=None,
        min_train=84,
        val_size=63,
        step=21,
        winsor=5.0,
    ):
        self.alpha = float(alpha)
        self.alphas = [float(a) for a in (alphas if alphas is not None else [0.03, 0.1, 0.3, 1.0, 3.0, 10.0])]
        self.min_train = int(max(32, min_train))
        self.val_size = int(max(10, val_size))
        self.step = int(max(5, step))
        self.winsor = float(max(1.0, winsor))

        self.coef_ = None
        self.intercept_ = 0.0
        self.x_mu_ = None
        self.x_sd_ = None
        self.alpha_ = self.alpha
        self.cv_score_ = None
        self.resid_std_ = 1.0

    def _winsorize(self, y):
        y = np.asarray(y, dtype=float).ravel()
        mu = float(np.nanmean(y))
        sd = float(np.nanstd(y) + 1e-12)
        lo = mu - self.winsor * sd
        hi = mu + self.winsor * sd
        return np.clip(y, lo, hi)

    @staticmethod
    def _ridge_fit(X, y, alpha):
        # X is assumed standardized and finite.
        y = np.asarray(y, dtype=float).ravel()
        yc = y - float(np.mean(y))
        XtX = X.T @ X
        K = XtX.shape[0]
        reg = float(alpha) * np.eye(K, dtype=float)
        coef = np.linalg.pinv(XtX + reg) @ (X.T @ yc)
        intercept = float(np.mean(y))
        return coef, intercept

    @staticmethod
    def _ridge_predict(X, coef, intercept):
        return (X @ coef + intercept).ravel()

    def _select_alpha(self, X, y):
        T = X.shape[0]
        if T < (self.min_train + self.val_size + 5):
            return self.alpha, np.nan

        best_a = self.alpha
        best_score = -1e9
        for a in self.alphas:
            fold_scores = []
            for end in range(self.min_train, T - self.val_size + 1, self.step):
                Xtr = X[:end]
                ytr = y[:end]
                Xva = X[end : end + self.val_size]
                yva = y[end : end + self.val_size]
                if len(Xva) < 5:
                    continue

                coef, b0 = self._ridge_fit(Xtr, ytr, a)
                pred = self._ridge_predict(Xva, coef, b0)
                # Score as tradable signal quality, not raw regression loss.
                fold_scores.append(_annualized_sharpe(pred * yva))

            if fold_scores:
                score = float(np.mean(fold_scores))
                if score > best_score:
                    best_score = score
                    best_a = float(a)
        return best_a, best_score

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2 or X.shape[0] < 6:
            k = X.shape[1] if X.ndim == 2 else 1
            self.coef_ = np.ones(k, dtype=float) / max(1, k)
            self.intercept_ = 0.0
            self.x_mu_ = np.zeros((1, k), dtype=float)
            self.x_sd_ = np.ones((1, k), dtype=float)
            self.alpha_ = self.alpha
            self.cv_score_ = np.nan
            self.resid_std_ = 1.0
            return self

        T = min(X.shape[0], y.shape[0])
        X = X[:T]
        y = y[:T]

        # 1-step lag: today's prediction uses yesterday's council state.
        Xl = X[:-1]
        yl = y[1:]
        if Xl.shape[0] < 6:
            k = X.shape[1]
            self.coef_ = np.ones(k, dtype=float) / max(1, k)
            self.intercept_ = 0.0
            self.x_mu_ = np.zeros((1, k), dtype=float)
            self.x_sd_ = np.ones((1, k), dtype=float)
            self.alpha_ = self.alpha
            self.cv_score_ = np.nan
            self.resid_std_ = 1.0
            return self

        self.x_mu_ = np.nanmean(Xl, axis=0, keepdims=True)
        self.x_sd_ = _safe_std(Xl, axis=0).reshape(1, -1)
        Xs = (Xl - self.x_mu_) / self.x_sd_
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        yw = np.nan_to_num(self._winsorize(yl), nan=0.0, posinf=0.0, neginf=0.0)

        self.alpha_, self.cv_score_ = self._select_alpha(Xs, yw)
        self.coef_, self.intercept_ = self._ridge_fit(Xs, yw, self.alpha_)

        yhat = self._ridge_predict(Xs, self.coef_, self.intercept_)
        resid = yhat - yw
        rs = float(np.nanstd(resid, ddof=1))
        self.resid_std_ = rs if np.isfinite(rs) and rs > 1e-8 else 1.0
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            return np.asarray([], dtype=float)
        if self.coef_ is None or self.x_mu_ is None or self.x_sd_ is None:
            return np.nanmean(X, axis=1)
        Xs = (X - self.x_mu_) / self.x_sd_
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        return self._ridge_predict(Xs, self.coef_, self.intercept_)

    def predict_confidence(self, X):
        p = self.predict(X)
        z = np.abs(p) / (self.resid_std_ + 1e-8)
        return np.tanh(z)
