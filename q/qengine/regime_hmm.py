"""Online HMM for regime-transition risk estimation."""

from __future__ import annotations

import numpy as np


class RegimeHMM:
    STATES = ["CALM_TREND", "CALM_CHOP", "HIGHVOL_TREND", "CRISIS"]
    N_STATES = 4

    def __init__(self, n_features: int = 6):
        self.n_features = int(max(1, n_features))
        self.A = np.array(
            [
                [0.92, 0.04, 0.03, 0.01],
                [0.05, 0.88, 0.05, 0.02],
                [0.03, 0.04, 0.88, 0.05],
                [0.02, 0.03, 0.10, 0.85],
            ],
            dtype=float,
        )
        self.mu = np.zeros((self.N_STATES, self.n_features), dtype=float)
        self.sigma = np.ones((self.N_STATES, self.n_features), dtype=float)
        self.belief = np.ones(self.N_STATES, dtype=float) / float(self.N_STATES)
        self._prev_state: int | None = None

    def _emission_prob(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, float).ravel()
        if x.size < self.n_features:
            x = np.pad(x, (0, self.n_features - x.size), constant_values=0.0)
        x = x[: self.n_features]

        probs = np.zeros(self.N_STATES, dtype=float)
        for s in range(self.N_STATES):
            diff = x - self.mu[s]
            var = self.sigma[s] + 1e-8
            log_p = -0.5 * np.sum((diff ** 2) / var + np.log(var))
            probs[s] = float(np.exp(np.clip(log_p, -500.0, 0.0)))

        total = float(probs.sum())
        if total > 0.0 and np.isfinite(total):
            return probs / total
        return np.ones(self.N_STATES, dtype=float) / float(self.N_STATES)

    def update(self, obs: np.ndarray, known_state: int | None = None) -> None:
        predicted = self.A.T @ self.belief
        emission = self._emission_prob(obs)

        posterior = predicted * emission
        total = float(posterior.sum())
        if total > 0:
            posterior /= total
        else:
            posterior = predicted

        if known_state is not None and 0 <= int(known_state) < self.N_STATES:
            anchor = np.zeros(self.N_STATES, dtype=float)
            anchor[int(known_state)] = 1.0
            posterior = 0.70 * posterior + 0.30 * anchor
            posterior /= float(posterior.sum())

        self.belief = np.nan_to_num(posterior, nan=1.0 / self.N_STATES)
        self.belief = np.clip(self.belief, 1e-9, 1.0)
        self.belief /= float(self.belief.sum())

        lr = 0.005
        st = int(np.argmax(self.belief))
        x = np.asarray(obs, float).ravel()
        if x.size < self.n_features:
            x = np.pad(x, (0, self.n_features - x.size), constant_values=0.0)
        x = x[: self.n_features]

        self.mu[st] += lr * (x - self.mu[st])
        diff_sq = (x - self.mu[st]) ** 2
        self.sigma[st] += lr * (diff_sq - self.sigma[st])
        self.sigma = np.maximum(self.sigma, 0.01)

        if self._prev_state is not None:
            self.A[self._prev_state, st] += lr
            row_sum = float(self.A[self._prev_state].sum())
            if row_sum > 0:
                self.A[self._prev_state] /= row_sum
        self._prev_state = st

    def transition_risk(self, horizon: int = 5) -> dict:
        hz = int(max(1, horizon))
        current = int(np.argmax(self.belief))

        A_h = np.linalg.matrix_power(self.A, hz)
        future_belief = A_h.T @ self.belief
        future_belief = np.clip(np.nan_to_num(future_belief, nan=0.0), 0.0, 1.0)
        if float(future_belief.sum()) <= 0:
            future_belief = np.ones(self.N_STATES, dtype=float) / self.N_STATES
        else:
            future_belief /= float(future_belief.sum())

        stay_prob = float(future_belief[current])
        transition_prob = float(np.clip(1.0 - stay_prob, 0.0, 1.0))

        destination = {self.STATES[s]: float(future_belief[s]) for s in range(self.N_STATES)}
        crisis_idx = self.STATES.index("CRISIS")

        return {
            "transition_prob": transition_prob,
            "current_regime": self.STATES[current],
            "current_confidence": float(self.belief[current]),
            "destination_probs": destination,
            "crisis_risk_5d": float(np.clip(future_belief[crisis_idx], 0.0, 1.0)),
        }
