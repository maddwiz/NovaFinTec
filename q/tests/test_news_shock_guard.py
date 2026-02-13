import numpy as np

from qmods.news_shock_guard import apply_shock_guard, shock_mask


def test_shock_mask_detects_and_persists_shocks():
    v = np.array([0.01] * 40 + [0.20, 0.25, 0.22] + [0.01] * 20, dtype=float)
    m = shock_mask(v, z=2.0, min_len=2, lookback=20, cooldown=2, quantile=0.95)
    assert m.shape == v.shape
    assert np.max(m) == 1
    # Should remain on briefly after spike due to cooldown.
    assert int(np.sum(m)) >= 3


def test_apply_shock_guard_scales_signal():
    s = np.array([1.0, -1.0, 0.5, 0.5], dtype=float)
    m = np.array([0, 1, 1, 0], dtype=int)
    g = apply_shock_guard(s, m, alpha=0.4)
    assert np.allclose(g, np.array([1.0, -0.6, 0.3, 0.5]))
