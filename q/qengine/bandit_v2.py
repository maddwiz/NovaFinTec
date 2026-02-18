"""Thompson-sampling council bandit with decay and magnitude-aware rewards."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class ThompsonBandit:
    """Beta-Bernoulli Thompson sampling with exponential decay."""

    def __init__(
        self,
        n_arms: int | None = None,
        prior_alpha: float = 2.0,
        prior_beta: float = 2.0,
        decay: float = 0.995,
        magnitude_scaling: bool = True,
        prior_file: str | None = None,
    ):
        self.n = int(max(0, n_arms or 0))
        self.prior_alpha = float(max(0.5, prior_alpha))
        self.prior_beta = float(max(0.5, prior_beta))
        self.decay = float(np.clip(decay, 0.90, 1.0))
        self.magnitude_scaling = bool(magnitude_scaling)
        self.prior_file = str(prior_file or "")

        self.alpha = np.full(max(1, self.n), self.prior_alpha, dtype=float)
        self.beta_ = np.full(max(1, self.n), self.prior_beta, dtype=float)
        self.keys_: list[str] = []
        self.weights_: dict[str, float] = {}
        self._total_updates = 0

    def _ensure_n(self, n_arms: int) -> None:
        n = int(max(1, n_arms))
        if self.n == n and self.alpha.size == n and self.beta_.size == n:
            return
        self.n = n
        self.alpha = np.full(n, self.prior_alpha, dtype=float)
        self.beta_ = np.full(n, self.prior_beta, dtype=float)

    def _decay_all(self) -> None:
        self.alpha *= self.decay
        self.beta_ *= self.decay
        self.alpha = np.maximum(self.alpha, 0.5)
        self.beta_ = np.maximum(self.beta_, 0.5)

    def _scale(self, magnitude: float) -> float:
        if not self.magnitude_scaling:
            return 1.0
        return float(np.clip(float(abs(magnitude)) / 0.01, 0.1, 10.0))

    def update(self, arm: int, reward: float, magnitude: float = 1.0) -> None:
        if self.n <= 0:
            return
        i = int(arm)
        if i < 0 or i >= self.n:
            return
        self._decay_all()
        s = self._scale(magnitude)
        if float(reward) > 0.5:
            self.alpha[i] += s
        else:
            self.beta_[i] += s
        self._total_updates += 1

    def update_all(self, rewards: np.ndarray, magnitudes: np.ndarray | None = None) -> None:
        if self.n <= 0:
            return
        r = np.asarray(rewards, float).ravel()
        if magnitudes is None:
            m = np.ones_like(r, dtype=float)
        else:
            m = np.asarray(magnitudes, float).ravel()
            if m.size < r.size:
                pad = np.ones(r.size - m.size, dtype=float)
                m = np.concatenate([m, pad])

        self._decay_all()
        L = min(self.n, r.size)
        for i in range(L):
            s = self._scale(m[i])
            if r[i] > 0.5:
                self.alpha[i] += s
            else:
                self.beta_[i] += s
        self._total_updates += 1

    def sample_weights(self, temperature: float = 1.0) -> np.ndarray:
        if self.n <= 0:
            return np.asarray([], float)
        t = float(max(1e-6, temperature))
        # Keep deterministic-ish behavior via means when temperature extremely small.
        if t < 1e-3:
            return self.mean_weights()
        samples = np.array(
            [
                np.random.beta(
                    max(0.5, self.alpha[i] ** (1.0 / t)),
                    max(0.5, self.beta_[i] ** (1.0 / t)),
                )
                for i in range(self.n)
            ],
            dtype=float,
        )
        s = float(samples.sum())
        if s <= 1e-12:
            return np.full(self.n, 1.0 / self.n, dtype=float)
        return samples / s

    def mean_weights(self) -> np.ndarray:
        if self.n <= 0:
            return np.asarray([], float)
        means = self.alpha / (self.alpha + self.beta_ + 1e-12)
        s = float(means.sum())
        if s <= 1e-12:
            return np.full(self.n, 1.0 / self.n, dtype=float)
        return means / s

    def confidence_intervals(self, percentile: float = 0.90) -> np.ndarray:
        if self.n <= 0:
            return np.asarray([], float)
        lo_p = float((1.0 - percentile) / 2.0)
        hi_p = float(1.0 - lo_p)
        try:
            from scipy.stats import beta as beta_dist  # type: ignore

            widths = np.array(
                [
                    beta_dist.ppf(hi_p, self.alpha[i], self.beta_[i])
                    - beta_dist.ppf(lo_p, self.alpha[i], self.beta_[i])
                    for i in range(self.n)
                ],
                dtype=float,
            )
            return np.nan_to_num(widths, nan=1.0, posinf=1.0, neginf=1.0)
        except Exception:
            # Fallback approximation using normal approx on Beta variance.
            mu = self.alpha / (self.alpha + self.beta_ + 1e-12)
            var = (self.alpha * self.beta_) / (
                (self.alpha + self.beta_) ** 2 * (self.alpha + self.beta_ + 1.0) + 1e-12
            )
            z = 1.645 if percentile >= 0.90 else 1.282
            w = 2.0 * z * np.sqrt(np.maximum(var, 0.0))
            return np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)

    def effective_sample_size(self) -> np.ndarray:
        if self.n <= 0:
            return np.asarray([], float)
        return self.alpha + self.beta_ - 1.0

    def _apply_signal_priors(self, keys: list[str]) -> None:
        if not self.prior_file:
            return
        p = Path(self.prior_file)
        if not p.exists():
            return
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return
        pri = raw.get("signal_priors") if isinstance(raw, dict) else None
        if not isinstance(pri, dict):
            pri = raw if isinstance(raw, dict) else {}

        for i, k in enumerate(keys):
            row = pri.get(str(k), {}) if isinstance(pri, dict) else {}
            if not isinstance(row, dict):
                continue
            a = float(row.get("alpha", row.get("wins", self.prior_alpha)))
            b = float(row.get("beta", row.get("losses", self.prior_beta)))
            self.alpha[i] = max(0.5, a)
            self.beta_[i] = max(0.5, b)

    def fit(self, signals: dict, returns) -> "ThompsonBandit":
        keys = list(signals.keys())
        if not keys:
            self.keys_ = []
            self.weights_ = {}
            self._ensure_n(1)
            return self

        S = pd.concat({k: pd.Series(v) for k, v in signals.items()}, axis=1).dropna()
        R = pd.Series(returns).reindex(S.index).fillna(0.0)

        self._ensure_n(len(keys))
        self.keys_ = list(keys)
        self.alpha[:] = self.prior_alpha
        self.beta_[:] = self.prior_beta
        self._apply_signal_priors(self.keys_)

        if len(S) <= 1:
            w = self.mean_weights()
            self.weights_ = {k: float(v) for k, v in zip(keys, w)}
            return self

        for i in range(1, len(S)):
            sign_pred = np.sign(S.iloc[i - 1].values)
            r = float(R.iloc[i])
            sign_real = 1.0 if r > 0 else (-1.0 if r < 0 else 0.0)
            rewards = (sign_pred == sign_real).astype(float)
            magnitudes = np.full(self.n, abs(r), dtype=float)
            self.update_all(rewards, magnitudes)

        w = self.mean_weights()
        self.weights_ = {k: float(v) for k, v in zip(keys, w)}
        return self

    def get_weights(self) -> dict:
        return dict(self.weights_ or {})
