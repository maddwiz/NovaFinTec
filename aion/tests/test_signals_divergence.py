from types import SimpleNamespace

import pandas as pd

from aion.brain.signals import _confirmation_component, _divergence_features, compute_features


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
        DIVERGENCE_LOOKBACK=8,
        DIVERGENCE_PRICE_MOVE_MIN=0.006,
        DIVERGENCE_RSI_DELTA_MIN=4.0,
        DIVERGENCE_OBV_DELTA_MIN=0.0,
        REGIME_ADX_TREND_MIN=20.0,
        REGIME_ATR_PCT_HIGH=0.045,
        REGIME_ATR_PCT_LOW=0.008,
        REGIME_BB_SQUEEZE_PCT=0.06,
    )


def test_divergence_features_detect_bullish_and_bearish_cases():
    cfg = _cfg()

    bull_df = pd.DataFrame(
        {
            "close": [100, 101, 102, 103, 104, 105, 106, 107, 98, 97, 96, 95],
            "rsi": [40, 41, 42, 43, 44, 45, 46, 47, 52, 53, 54, 55],
            "obv": [0, 10, 20, 30, 40, 50, 60, 70, 90, 100, 110, 120],
        }
    )
    bull = _divergence_features(bull_df, cfg)
    assert bool(bull["rsi_bull_div"].iloc[-1]) is True
    assert bool(bull["obv_bull_div"].iloc[-1]) is True
    assert bool(bull["rsi_bear_div"].iloc[-1]) is False
    assert bool(bull["obv_bear_div"].iloc[-1]) is False

    bear_df = pd.DataFrame(
        {
            "close": [100, 99, 98, 97, 96, 95, 94, 93, 102, 103, 104, 105],
            "rsi": [60, 59, 58, 57, 56, 55, 54, 53, 49, 48, 47, 46],
            "obv": [200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100, 90],
        }
    )
    bear = _divergence_features(bear_df, cfg)
    assert bool(bear["rsi_bear_div"].iloc[-1]) is True
    assert bool(bear["obv_bear_div"].iloc[-1]) is True
    assert bool(bear["rsi_bull_div"].iloc[-1]) is False
    assert bool(bear["obv_bull_div"].iloc[-1]) is False


def test_confirmation_component_scores_divergence_reasons():
    row = pd.Series(
        {
            "rsi_bull_div": True,
            "rsi_bear_div": False,
            "obv_bull_div": True,
            "obv_bear_div": False,
            "macd_cross_up": False,
            "macd_cross_down": False,
            "stoch_cross_up": False,
            "stoch_cross_down": False,
            "stoch_k": 50.0,
            "rsi_cross_50_up": False,
            "rsi_cross_50_down": False,
            "cci": 0.0,
            "mfi": 50.0,
            "roc": 0.0,
            "volume_rel": 1.0,
            "close": 100.0,
            "open": 100.0,
            "dist_high_20": 1.0,
            "dist_low_20": 1.0,
            "breakout_up": False,
            "breakout_down": False,
        }
    )
    long, short, reasons_l, reasons_s = _confirmation_component(row)
    assert long > 0.0
    assert short == 0.0
    assert "Bullish RSI divergence" in reasons_l
    assert "Bullish OBV divergence" in reasons_l
    assert reasons_s == []


def test_compute_features_emits_divergence_columns():
    cfg = _cfg()
    n = 48
    close = pd.Series([100.0 + i * 0.05 for i in range(n)])
    df = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": pd.Series([1_000_000 + (i % 7) * 20_000 for i in range(n)]),
        }
    )
    out = compute_features(df, cfg)
    for col in ["rsi_bull_div", "rsi_bear_div", "obv_bull_div", "obv_bear_div"]:
        assert col in out.columns
        assert str(out[col].dtype) == "bool"
    assert "head_and_shoulders_top" in out.columns
    assert "inverse_head_and_shoulders_bottom" in out.columns
    assert "falling_wedge_breakout" in out.columns
    assert "rising_wedge_breakdown" in out.columns
    assert str(out["head_and_shoulders_top"].dtype) == "bool"
    assert str(out["inverse_head_and_shoulders_bottom"].dtype) == "bool"
    assert str(out["falling_wedge_breakout"].dtype) == "bool"
    assert str(out["rising_wedge_breakdown"].dtype) == "bool"
