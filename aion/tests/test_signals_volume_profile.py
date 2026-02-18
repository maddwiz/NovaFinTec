import numpy as np
import pandas as pd

from aion.brain.signals import compute_volume_profile


def _session_df(close: np.ndarray, volume: np.ndarray) -> pd.DataFrame:
    c = pd.Series(close, dtype=float)
    return pd.DataFrame(
        {
            "high": c + 0.05,
            "low": c - 0.05,
            "close": c,
            "volume": pd.Series(volume, dtype=float),
        }
    )


def test_volume_profile_uniform_volume_poc_near_vwap():
    close = np.linspace(100.0, 101.0, 120)
    vol = np.full(120, 1000.0)
    df = _session_df(close, vol)

    va_high, va_low, poc = compute_volume_profile(df, n_bins=20)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vwap = float((tp * df["volume"]).sum() / (df["volume"].sum() + 1e-12))

    assert va_high > va_low
    assert abs(poc - vwap) < 0.08


def test_volume_profile_concentrated_highs_pushes_poc_up():
    close = np.linspace(100.0, 101.0, 120)
    vol = (np.linspace(1.0, 3.5, 120) ** 2) * 500.0
    df = _session_df(close, vol)

    _, _, poc = compute_volume_profile(df, n_bins=20)
    assert poc > 100.70


def test_value_area_covers_approximately_seventy_percent_volume():
    x = np.linspace(0.0, 8.0 * np.pi, 300)
    close = 100.0 + 0.8 * np.sin(x) + 0.3 * np.sin(0.35 * x)
    vol = 700.0 + 300.0 * (1.0 + np.sin(0.7 * x))
    df = _session_df(close, vol)

    va_high, va_low, _ = compute_volume_profile(df, n_bins=24)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    inside = (tp >= va_low) & (tp <= va_high)
    share = float(df.loc[inside, "volume"].sum() / (df["volume"].sum() + 1e-12))

    assert 0.60 <= share <= 0.90
