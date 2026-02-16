import numpy as np

from qmods.cross_hive_arb_v1 import arb_weights


def test_arb_weights_sum_and_bounds():
    T = 120
    scores = {
        "EQ": np.linspace(-0.2, 0.6, T),
        "FX": np.linspace(0.3, -0.1, T),
        "RATES": np.sin(np.linspace(0, 6, T)) * 0.2,
    }
    names, W = arb_weights(
        scores,
        alpha=2.0,
        inertia=0.85,
        min_weight=0.02,
        max_weight=0.70,
    )
    assert names == ["EQ", "FX", "RATES"]
    assert W.shape == (T, 3)
    assert np.isfinite(W).all()
    rs = W.sum(axis=1)
    assert np.allclose(rs, 1.0, atol=1e-6)
    assert float(np.min(W)) >= 0.0
    assert float(np.max(W)) <= 1.0 + 1e-9


def test_arb_weights_inertia_reduces_turnover():
    T = 200
    rng = np.random.default_rng(3)
    scores = {
        "A": rng.normal(0, 1, T),
        "B": rng.normal(0, 1, T),
        "C": rng.normal(0, 1, T),
    }
    _, w_fast = arb_weights(scores, inertia=0.0, alpha=2.0)
    _, w_slow = arb_weights(scores, inertia=0.9, alpha=2.0)
    t_fast = float(np.mean(np.sum(np.abs(np.diff(w_fast, axis=0)), axis=1)))
    t_slow = float(np.mean(np.sum(np.abs(np.diff(w_slow, axis=0)), axis=1)))
    assert t_slow < t_fast


def test_arb_weights_accepts_time_varying_alpha_and_inertia():
    T = 90
    scores = {
        "A": np.linspace(-0.4, 0.5, T),
        "B": np.linspace(0.2, -0.1, T),
        "C": np.sin(np.linspace(0, 5, T)) * 0.25,
    }
    alpha_t = np.linspace(1.0, 3.0, T)
    inertia_t = np.linspace(0.2, 0.9, T)
    names, W = arb_weights(scores, alpha=alpha_t, inertia=inertia_t, min_weight=0.01, max_weight=0.8)
    assert names == ["A", "B", "C"]
    assert W.shape == (T, 3)
    assert np.isfinite(W).all()
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-6)


def test_arb_weights_entropy_control_reduces_concentration():
    T = 140
    x = np.linspace(0.0, 12.0, T)
    scores = {
        "A": 1.8 * np.sin(x) + 0.3 * np.sin(2.3 * x),
        "B": 1.5 * np.cos(0.9 * x) - 0.2 * np.sin(1.7 * x),
        "C": -1.2 * np.sin(1.1 * x) + 0.4 * np.cos(1.9 * x),
    }
    _, w_base = arb_weights(scores, alpha=4.0, inertia=0.0, min_weight=0.0, max_weight=1.0)
    _, w_div = arb_weights(
        scores,
        alpha=4.0,
        inertia=0.0,
        min_weight=0.0,
        max_weight=1.0,
        entropy_target=0.70,
        entropy_strength=0.8,
    )
    hhi_base = float(np.mean(np.sum(w_base * w_base, axis=1)))
    hhi_div = float(np.mean(np.sum(w_div * w_div, axis=1)))
    assert hhi_div < hhi_base


def test_arb_weights_accepts_time_varying_entropy_controls():
    T = 120
    x = np.linspace(0.0, 9.0, T)
    scores = {
        "A": 1.4 * np.sin(x),
        "B": 1.2 * np.cos(1.1 * x),
        "C": -1.1 * np.sin(0.9 * x + 0.2),
    }
    et_lo = np.full(T, 0.50, dtype=float)
    es_lo = np.full(T, 0.10, dtype=float)
    et_hi = np.linspace(0.55, 0.85, T)
    es_hi = np.linspace(0.20, 0.95, T)

    _, w_lo = arb_weights(scores, alpha=3.8, inertia=0.0, entropy_target=et_lo, entropy_strength=es_lo)
    _, w_hi = arb_weights(scores, alpha=3.8, inertia=0.0, entropy_target=et_hi, entropy_strength=es_hi)

    hhi_lo = float(np.mean(np.sum(w_lo * w_lo, axis=1)))
    hhi_hi = float(np.mean(np.sum(w_hi * w_hi, axis=1)))
    assert hhi_hi < hhi_lo


def test_arb_weights_downside_penalty_reduces_weight():
    T = 160
    scores = {
        "A": np.full(T, 0.25, dtype=float),
        "B": np.full(T, 0.25, dtype=float),
        "C": np.full(T, 0.25, dtype=float),
    }
    base_pen = {"A": np.zeros(T), "B": np.zeros(T), "C": np.zeros(T)}
    high_dn = {"A": np.ones(T) * 0.9, "B": np.zeros(T), "C": np.zeros(T)}

    _, w0 = arb_weights(scores, alpha=2.0, inertia=0.0, downside_penalty=base_pen)
    _, w1 = arb_weights(scores, alpha=2.0, inertia=0.0, downside_penalty=high_dn)

    a0 = float(np.mean(w0[:, 0]))
    a1 = float(np.mean(w1[:, 0]))
    assert a1 < a0


def test_arb_weights_crowding_penalty_reduces_weight():
    T = 120
    scores = {
        "A": np.full(T, 0.30, dtype=float),
        "B": np.full(T, 0.30, dtype=float),
        "C": np.full(T, 0.30, dtype=float),
    }
    base_pen = {"A": np.zeros(T), "B": np.zeros(T), "C": np.zeros(T)}
    high_cr = {"A": np.ones(T) * 0.95, "B": np.zeros(T), "C": np.zeros(T)}

    _, w0 = arb_weights(scores, alpha=2.0, inertia=0.0, crowding_penalty=base_pen)
    _, w1 = arb_weights(scores, alpha=2.0, inertia=0.0, crowding_penalty=high_cr)

    a0 = float(np.mean(w0[:, 0]))
    a1 = float(np.mean(w1[:, 0]))
    assert a1 < a0


def test_arb_weights_step_turnover_cap_limits_adjacent_moves():
    T = 180
    rng = np.random.default_rng(17)
    scores = {
        "A": rng.normal(0.0, 1.4, T),
        "B": rng.normal(0.0, 1.4, T),
        "C": rng.normal(0.0, 1.4, T),
    }
    _, w_uncapped = arb_weights(scores, alpha=3.0, inertia=0.0)
    _, w_capped = arb_weights(scores, alpha=3.0, inertia=0.0, max_step_turnover=0.30)

    uncapped_turn = np.sum(np.abs(np.diff(w_uncapped, axis=0)), axis=1)
    capped_turn = np.sum(np.abs(np.diff(w_capped, axis=0)), axis=1)

    assert float(np.max(capped_turn)) <= 0.30 + 1e-6
    assert float(np.mean(capped_turn)) <= float(np.mean(uncapped_turn))


def test_arb_weights_rolling_turnover_budget_limits_window_sum():
    T = 200
    rng = np.random.default_rng(21)
    scores = {
        "A": rng.normal(0.0, 1.6, T),
        "B": rng.normal(0.0, 1.6, T),
        "C": rng.normal(0.0, 1.6, T),
    }
    _, w_unbudgeted = arb_weights(scores, alpha=3.2, inertia=0.0, max_step_turnover=2.0)
    _, w_budgeted = arb_weights(
        scores,
        alpha=3.2,
        inertia=0.0,
        max_step_turnover=2.0,
        rolling_turnover_window=5,
        rolling_turnover_limit=0.75,
    )

    t_unbudgeted = np.sum(np.abs(np.diff(w_unbudgeted, axis=0)), axis=1)
    t_budgeted = np.sum(np.abs(np.diff(w_budgeted, axis=0)), axis=1)
    roll = np.array([np.sum(t_budgeted[max(0, i - 4) : i + 1]) for i in range(len(t_budgeted))], dtype=float)

    assert float(np.max(roll)) <= 0.75 + 1e-6
    assert float(np.mean(t_budgeted)) <= float(np.mean(t_unbudgeted))
