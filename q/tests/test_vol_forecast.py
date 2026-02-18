import numpy as np

from qengine.vol_forecast import compute_har_features, fit_har, forecast_vol


def test_compute_har_features_columns_present():
    r = np.random.default_rng(7).normal(0.0, 0.01, size=300)
    f = compute_har_features(r)
    assert list(f.columns) == ["rv_d", "rv_w", "rv_m", "jump", "rsv"]
    assert len(f) == len(r)


def test_fit_har_returns_coefficients():
    r = np.random.default_rng(11).normal(0.0, 0.012, size=400)
    f = compute_har_features(r)
    target = (r ** 2)
    m = fit_har(f, target, min_obs=120)
    assert "coefficients" in m
    assert m["n"] >= 0


def test_forecast_vol_output_finite_and_positive():
    rng = np.random.default_rng(42)
    # Vol clustering: low-vol then high-vol block.
    r1 = rng.normal(0.0, 0.008, size=320)
    r2 = rng.normal(0.0, 0.020, size=320)
    r = np.concatenate([r1, r2])

    fc = forecast_vol(r, train_window=252, step=21)
    assert fc.shape[0] == r.shape[0]
    assert np.isfinite(fc).all()
    assert float(np.nanmin(fc)) >= 0.0
    assert float(np.mean(fc[-80:])) >= float(np.mean(fc[:80])) * 0.8
