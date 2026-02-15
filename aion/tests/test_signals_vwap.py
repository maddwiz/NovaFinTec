from types import SimpleNamespace

import pandas as pd

from aion.brain.signals import _confirmation_component, compute_features


def _cfg():
    return SimpleNamespace(
        EMA_FAST=12,
        EMA_SLOW=26,
        RSI_LEN=14,
        ATR_LEN=14,
        ADX_LEN=14,
        BB_LEN=20,
        BB_STD=2.0,
        STOCH_LEN=14,
        STOCH_SMOOTH=3,
        VWAP_LEN=20,
        DIVERGENCE_LOOKBACK=8,
        DIVERGENCE_PRICE_MOVE_MIN=0.006,
        DIVERGENCE_RSI_DELTA_MIN=4.0,
        DIVERGENCE_OBV_DELTA_MIN=0.0,
        REGIME_ADX_TREND_MIN=20.0,
        REGIME_ATR_PCT_HIGH=0.045,
        REGIME_ATR_PCT_LOW=0.008,
        REGIME_BB_SQUEEZE_PCT=0.06,
    )


def test_compute_features_emits_vwap_columns():
    cfg = _cfg()
    n = 52
    close = pd.Series([100.0 + i * 0.10 for i in range(n)])
    df = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": pd.Series([800_000 + (i % 9) * 30_000 for i in range(n)]),
        }
    )
    out = compute_features(df, cfg)
    for col in ["vwap", "vwap_gap_pct", "vwap_cross_up", "vwap_cross_down"]:
        assert col in out.columns
    assert str(out["vwap_cross_up"].dtype) == "bool"
    assert str(out["vwap_cross_down"].dtype) == "bool"


def test_confirmation_component_scores_bullish_vwap_setup():
    row = pd.Series(
        {
            "vwap_cross_up": True,
            "vwap_cross_down": False,
            "volume_rel": 1.25,
            "vwap_gap_pct": 0.006,
            "close": 101.0,
            "open": 100.4,
        }
    )
    long, short, reasons_l, reasons_s = _confirmation_component(row)
    assert long > 0.0
    assert short == 0.0
    assert "VWAP bullish cross" in reasons_l
    assert "Holding above VWAP on volume" in reasons_l
    assert reasons_s == []


def test_confirmation_component_scores_bearish_vwap_setup():
    row = pd.Series(
        {
            "vwap_cross_up": False,
            "vwap_cross_down": True,
            "volume_rel": 1.30,
            "vwap_gap_pct": -0.007,
            "close": 99.4,
            "open": 100.0,
        }
    )
    long, short, reasons_l, reasons_s = _confirmation_component(row)
    assert short > 0.0
    assert long == 0.0
    assert "VWAP bearish cross" in reasons_s
    assert "Holding below VWAP on volume" in reasons_s
    assert reasons_l == []
