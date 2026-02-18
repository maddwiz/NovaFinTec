import numpy as np

from qengine.adaptive_signals import (
    decompose_and_signal,
    empirical_mode_decomposition,
    estimate_dominant_period,
)


def test_emd_extracts_multiple_components_from_mixed_signal():
    t = np.arange(512, dtype=float)
    x = np.sin(2.0 * np.pi * t / 12.0) + 0.6 * np.sin(2.0 * np.pi * t / 48.0)
    imfs = empirical_mode_decomposition(x, max_imfs=6)
    assert len(imfs) >= 2

    periods = [estimate_dominant_period(imf) for imf in imfs]
    assert any(8 <= p <= 18 for p in periods)
    assert any(30 <= p <= 70 for p in periods)


def test_monotonic_series_produces_positive_trend_signal_tail():
    c = np.linspace(100.0, 180.0, 300)
    out = decompose_and_signal(c)
    trend = np.asarray(out["trend_signal"], float)
    assert trend.shape[0] == c.shape[0]
    assert float(np.abs(np.mean(trend[-20:]))) > 0.1


def test_stationary_oscillation_produces_finite_cycle_signal():
    t = np.arange(300, dtype=float)
    c = 100.0 + 2.0 * np.sin(2.0 * np.pi * t / 16.0)
    out = decompose_and_signal(c)
    cyc = np.asarray(out["cycle_signal"], float)
    assert cyc.shape[0] == c.shape[0]
    assert np.isfinite(cyc).all()
    assert float(np.max(np.abs(cyc))) <= 1.0 + 1e-9


def test_flat_series_graceful_output():
    c = np.full(128, 100.0, dtype=float)
    out = decompose_and_signal(c)
    for k in ["trend_signal", "cycle_signal", "composite"]:
        v = np.asarray(out[k], float)
        assert v.shape[0] == c.shape[0]
        assert np.isfinite(v).all()
