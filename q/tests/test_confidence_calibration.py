import numpy as np

from qmods.confidence_calibration import (
    apply_empirical_calibrator,
    fit_empirical_calibrator,
    reliability_governor_from_calibrated,
)


def test_empirical_calibrator_monotone_and_bounded():
    rng = np.random.default_rng(9)
    c = np.linspace(0.0, 1.0, 600)
    p = 0.25 + 0.60 * c
    y = (rng.uniform(0.0, 1.0, size=len(c)) < p).astype(float)

    cal = fit_empirical_calibrator(c, y, n_bins=12, min_count=30)
    x = np.asarray(cal["bin_centers"], float)
    h = np.asarray(cal["bin_hit_rate"], float)

    assert len(x) >= 2
    assert len(x) == len(h)
    assert np.all(np.diff(x) >= -1e-12)
    assert np.all(np.diff(h) >= -1e-12)
    assert np.min(h) >= 0.0
    assert np.max(h) <= 1.0


def test_calibrated_confidence_and_governor_shapes():
    rng = np.random.default_rng(5)
    c = rng.uniform(0.0, 1.0, size=420)
    y = (rng.uniform(0.0, 1.0, size=420) < (0.40 + 0.40 * c)).astype(float)
    cal = fit_empirical_calibrator(c, y, n_bins=10, min_count=20)
    cc = apply_empirical_calibrator(c, cal)
    g = reliability_governor_from_calibrated(cc, lo=0.72, hi=1.16, smooth=0.85)

    assert cc.shape == c.shape
    assert g.shape == c.shape
    assert np.isfinite(cc).all()
    assert np.isfinite(g).all()
    assert float(np.min(cc)) >= 0.0 - 1e-9
    assert float(np.max(cc)) <= 1.0 + 1e-9
    assert float(np.min(g)) >= 0.72 - 1e-9
    assert float(np.max(g)) <= 1.16 + 1e-9
