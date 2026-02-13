from __future__ import annotations

import numpy as np


def annualized_sharpe(r: np.ndarray) -> float:
    x = np.asarray(r, float).ravel()
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return 0.0
    mu = float(np.mean(x))
    sd = float(np.std(x) + 1e-12)
    return float((mu / sd) * np.sqrt(252.0))


def max_dd(r: np.ndarray) -> float:
    x = np.asarray(r, float).ravel()
    if len(x) == 0:
        return 0.0
    eq = np.cumprod(1.0 + np.clip(x, -0.95, 0.95))
    pk = np.maximum.accumulate(eq)
    dd = eq / (pk + 1e-12) - 1.0
    return float(np.min(dd))


def run_scenarios(W: np.ndarray, A: np.ndarray) -> dict:
    """
    W: [T,N] weights
    A: [T,N] asset returns
    """
    T = min(W.shape[0], A.shape[0])
    N = min(W.shape[1], A.shape[1])
    W = np.asarray(W[:T, :N], float)
    A = np.asarray(A[:T, :N], float)
    base = np.sum(W * A, axis=1)

    # Scenario transforms on returns matrix.
    market = np.mean(A, axis=1, keepdims=True)
    vol_spike = 1.5 * A

    trend_reversal = A.copy()
    if T > 20:
        cut = max(5, int(0.67 * T))
        trend_reversal[cut:] = -1.0 * trend_reversal[cut:]

    flash_crash = A.copy()
    if T > 0:
        shock_idx = np.arange(T) % 63 == 0
        flash_crash[shock_idx] = flash_crash[shock_idx] - 0.08 - 0.30 * np.abs(flash_crash[shock_idx])

    corr_shock = np.sign(market) * np.abs(A)

    scenarios = {
        "baseline": A,
        "vol_spike_1p5x": vol_spike,
        "trend_reversal_tail": trend_reversal,
        "flash_crash_monthly": flash_crash,
        "corr_shock": corr_shock,
    }

    out = {}
    for name, R in scenarios.items():
        pnl = np.sum(W * np.asarray(R, float), axis=1)
        out[name] = {
            "mean_daily": float(np.mean(pnl)) if len(pnl) else 0.0,
            "sharpe": float(annualized_sharpe(pnl)),
            "max_dd": float(max_dd(pnl)),
            "tail_p01": float(np.nanpercentile(pnl, 1)) if len(pnl) else 0.0,
        }

    worst_dd = min(v["max_dd"] for v in out.values()) if out else 0.0
    worst_tail = min(v["tail_p01"] for v in out.values()) if out else 0.0
    worst_sh = min(v["sharpe"] for v in out.values()) if out else 0.0
    # Pass thresholds tuned to be strict enough to catch fragility but not noisy.
    passed = bool((worst_dd > -0.70) and (worst_tail > -0.25) and (worst_sh > -8.0))
    summary = {
        "pass": passed,
        "worst_max_dd": float(worst_dd),
        "worst_tail_p01": float(worst_tail),
        "worst_sharpe": float(worst_sh),
    }
    return {"summary": summary, "scenarios": out}
