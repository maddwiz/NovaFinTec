from types import SimpleNamespace

import pandas as pd

from aion.brain.signals import _breakout_component, compute_features


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
        FIB_TOLERANCE=0.003,
        BREAKOUT_RETEST_LOOKBACK=4,
        BREAKOUT_RETEST_TOL=0.0025,
        DIVERGENCE_LOOKBACK=8,
        DIVERGENCE_PRICE_MOVE_MIN=0.006,
        DIVERGENCE_RSI_DELTA_MIN=4.0,
        DIVERGENCE_OBV_DELTA_MIN=0.0,
        REGIME_ADX_TREND_MIN=20.0,
        REGIME_ATR_PCT_HIGH=0.045,
        REGIME_ATR_PCT_LOW=0.008,
        REGIME_BB_SQUEEZE_PCT=0.06,
    )


def test_compute_features_emits_breakout_retest_columns():
    cfg = _cfg()
    n = 80
    close = pd.Series([100.0 + i * 0.12 for i in range(n)])
    df = pd.DataFrame(
        {
            "open": close - 0.12,
            "high": close + 0.35,
            "low": close - 0.35,
            "close": close,
            "volume": pd.Series([900_000 + (i % 11) * 25_000 for i in range(n)]),
        }
    )
    out = compute_features(df, cfg)
    for col in ["breakout_retest_up", "breakout_retest_down"]:
        assert col in out.columns
        assert str(out[col].dtype) == "bool"


def test_breakout_component_scores_retest_reasons():
    row = pd.Series(
        {
            "breakout_up": False,
            "breakout_down": False,
            "breakout_retest_up": True,
            "breakout_retest_down": False,
            "double_bottom": False,
            "double_top": False,
            "inverse_head_and_shoulders_bottom": False,
            "head_and_shoulders_top": False,
        }
    )
    long, short, reasons_l, reasons_s = _breakout_component(row)
    assert long > 0.0
    assert short == 0.0
    assert "Breakout retest held" in reasons_l
    assert reasons_s == []
