from types import SimpleNamespace

import pandas as pd

from aion.brain.signals import intraday_entry_alignment


def _cfg():
    return SimpleNamespace(
        HIST_BAR_SIZE="1 min",
        INTRADAY_OPEN_RANGE_MIN=15,
        INTRADAY_BREAK_TOL=0.0012,
        INTRADAY_VOLUME_REL_MIN=1.12,
        INTRADAY_RECENT_BARS=6,
    )


def test_intraday_entry_alignment_scores_bullish_breakout():
    cfg = _cfg()
    idx = pd.date_range("2026-02-17 09:30:00", periods=70, freq="min")

    # Opening range stays narrow, then breaks higher with stronger prints.
    open_rng = [100.00 + (i % 4) * 0.03 for i in range(15)]
    trend = [100.25 + i * 0.04 for i in range(55)]
    close = pd.Series(open_rng + trend)
    high = close + 0.08
    low = close - 0.07
    open_px = close.shift(1).fillna(close.iloc[0] - 0.02)
    volume = pd.Series([6000 + (i % 5) * 250 for i in range(15)] + [9000 + (i % 7) * 500 for i in range(55)])
    vwap = close.rolling(20, min_periods=1).mean() - 0.05

    df = pd.DataFrame(
        {
            "date": idx,
            "open": open_px,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "vwap": vwap,
        }
    )

    score, reasons = intraday_entry_alignment(df, "LONG", cfg)
    assert score >= 0.70
    assert any("breakout" in r.lower() for r in reasons)
    assert any("vwap" in r.lower() for r in reasons)


def test_intraday_entry_alignment_penalizes_failed_long_setup():
    cfg = _cfg()
    idx = pd.date_range("2026-02-17 09:30:00", periods=70, freq="min")

    # Weak down drift, no opening-range breakout and below VWAP.
    close = pd.Series([100.3 - i * 0.01 for i in range(70)])
    high = close + 0.05
    low = close - 0.06
    open_px = close.shift(1).fillna(close.iloc[0] + 0.01)
    volume = pd.Series([5200 + (i % 4) * 150 for i in range(70)])
    vwap = close.rolling(20, min_periods=1).mean() + 0.10

    df = pd.DataFrame(
        {
            "date": idx,
            "open": open_px,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "vwap": vwap,
        }
    )

    score, reasons = intraday_entry_alignment(df, "LONG", cfg)
    assert score <= 0.55
    assert any("no opening-range breakout" in r.lower() for r in reasons)
    assert any("below vwap" in r.lower() for r in reasons)
