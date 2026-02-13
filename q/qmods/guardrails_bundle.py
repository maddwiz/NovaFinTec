#!/usr/bin/env python3
# Parameter stability, turnover cost, council disagreement

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class StabilityResult:
    keep_mask: np.ndarray
    stability_score: float


@dataclass
class TurnoverGovernedResult:
    weights: np.ndarray
    turnover_before: np.ndarray
    turnover_after: np.ndarray
    scale_applied: np.ndarray


def parameter_stability_filter(params_history: np.ndarray, thresh: float = 0.6) -> StabilityResult:
    # params_history: [n_windows, n_params]
    if params_history.ndim != 2:
        raise ValueError("params_history must be 2D [n_windows, n_params]")
    stds = np.std(params_history, axis=0)
    med = np.median(params_history, axis=0)
    keep = (stds <= (np.abs(med) + 1e-8) * (1.0 - thresh)) | (stds < 1e-3)
    score = float(np.mean(keep))
    return StabilityResult(keep_mask=keep, stability_score=score)

def turnover_cost_penalty(weights_t: np.ndarray, fee_bps: float = 5.0) -> float:
    # weights_t: [T, N]
    if weights_t.ndim != 2 or weights_t.shape[0] < 2:
        return 0.0
    turns = np.sum(np.abs(np.diff(weights_t, axis=0)), axis=1)  # L1 turnover per step
    avg_turn = float(np.mean(turns)) if len(turns) else 0.0
    cost = - (fee_bps / 10000.0) * avg_turn * 252.0  # negative Sharpe adj
    return cost

def disagreement_gate(votes: np.ndarray, clamp=(0.5, 1.0)) -> float:
    # votes: [K] council outputs in [-1,1]
    v = votes.ravel()
    dispersion = float(np.std(v))
    return max(clamp[0], clamp[1] - dispersion)


def apply_turnover_governor(weights_t: np.ndarray, max_step_turnover: float = 0.35) -> TurnoverGovernedResult:
    """
    Enforce an L1 turnover budget per step:
      sum_i |w_t[i] - w_{t-1}[i]| <= max_step_turnover
    """
    w = np.asarray(weights_t, float)
    if w.ndim != 2 or w.shape[0] < 2:
        empty = np.zeros(max(0, w.shape[0] - 1), dtype=float)
        ones = np.ones_like(empty)
        return TurnoverGovernedResult(weights=w.copy(), turnover_before=empty, turnover_after=empty, scale_applied=ones)

    cap = float(max(0.0, max_step_turnover))
    out = w.copy()
    t_before = np.zeros(w.shape[0] - 1, dtype=float)
    t_after = np.zeros(w.shape[0] - 1, dtype=float)
    scales = np.ones(w.shape[0] - 1, dtype=float)

    for t in range(1, w.shape[0]):
        prev = out[t - 1]
        target = w[t]
        delta = target - prev
        turn = float(np.sum(np.abs(delta)))
        t_before[t - 1] = turn
        if cap > 0.0 and turn > cap:
            alpha = cap / (turn + 1e-12)
            scales[t - 1] = alpha
            out[t] = prev + alpha * delta
        else:
            out[t] = target
        t_after[t - 1] = float(np.sum(np.abs(out[t] - out[t - 1])))

    return TurnoverGovernedResult(
        weights=out,
        turnover_before=t_before,
        turnover_after=t_after,
        scale_applied=scales,
    )
