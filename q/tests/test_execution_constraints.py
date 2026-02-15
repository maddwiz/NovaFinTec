import numpy as np

import tools.run_execution_constraints as rec


def test_step_turnover_basic():
    w = np.array(
        [
            [0.0, 0.0],
            [1.0, -1.0],
            [2.0, -2.0],
        ],
        dtype=float,
    )
    t = rec._step_turnover(w)
    assert np.allclose(t, np.array([2.0, 2.0], dtype=float))


def test_apply_asset_delta_cap_limits_each_asset_step():
    w = np.array(
        [
            [0.0, 0.0],
            [0.30, -0.40],
            [0.10, -0.10],
        ],
        dtype=float,
    )
    out = rec._apply_asset_delta_cap(w, cap=0.15)
    d = np.diff(out, axis=0)
    assert float(np.max(np.abs(d))) <= 0.15 + 1e-12
    assert np.allclose(out[1], np.array([0.15, -0.15], dtype=float))
    assert np.allclose(out[2], np.array([0.10, -0.10], dtype=float))


def test_apply_turnover_caps_step_limit():
    w = np.array(
        [
            [0.0, 0.0],
            [1.0, -1.0],
            [2.0, -2.0],
        ],
        dtype=float,
    )
    out, before, after = rec._apply_turnover_caps(w, max_step_turnover=1.0)

    assert np.allclose(before, np.array([2.0, 2.0], dtype=float))
    assert np.all(after <= 1.0 + 1e-12)
    assert np.allclose(after, np.array([1.0, 1.0], dtype=float))
    assert np.allclose(out[-1], np.array([1.0, -1.0], dtype=float))


def test_apply_turnover_caps_rolling_budget():
    w = np.array(
        [
            [0.0, 0.0],
            [1.0, -1.0],
            [2.0, -2.0],
        ],
        dtype=float,
    )
    out, _before, after = rec._apply_turnover_caps(
        w,
        max_step_turnover=None,
        rolling_window=2,
        rolling_limit=1.5,
    )

    assert np.allclose(after, np.array([1.5, 0.0], dtype=float))
    assert np.allclose(out[1], np.array([0.75, -0.75], dtype=float))
    assert np.allclose(out[2], out[1])
