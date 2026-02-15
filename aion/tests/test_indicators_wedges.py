import pandas as pd

from aion.brain.indicators import falling_wedge_breakout, rising_wedge_breakdown


def test_rising_wedge_breakdown_detects_constructed_pattern():
    close = pd.Series(
        [
            100.0,
            101.0,
            102.0,
            103.0,
            104.0,
            105.0,
            106.0,
            106.4,
            106.8,
            107.1,
            107.3,
            107.45,
            107.55,
            107.62,
            107.68,
            107.72,
            107.75,
            107.78,
            99.0,
        ]
    )
    n = len(close)
    upper_pad = pd.Series([2.3 - 1.5 * i / max(1, n - 1) for i in range(n)])
    lower_pad = pd.Series([2.0 - 1.2 * i / max(1, n - 1) for i in range(n)])
    df = pd.DataFrame(
        {
            "close": close,
            "high": close + upper_pad,
            "low": close - lower_pad,
        }
    )
    sig = rising_wedge_breakdown(df, tol=0.004)
    assert bool(sig.iloc[-1]) is True


def test_falling_wedge_breakout_detects_constructed_pattern():
    close = pd.Series(
        [
            110.0,
            109.0,
            108.0,
            107.0,
            106.0,
            105.0,
            104.0,
            103.7,
            103.4,
            103.1,
            102.9,
            102.75,
            102.62,
            102.55,
            102.50,
            102.46,
            102.43,
            102.41,
            114.0,
        ]
    )
    n = len(close)
    upper_pad = pd.Series([2.2 - 1.4 * i / max(1, n - 1) for i in range(n)])
    lower_pad = pd.Series([2.1 - 1.3 * i / max(1, n - 1) for i in range(n)])
    df = pd.DataFrame(
        {
            "close": close,
            "high": close + upper_pad,
            "low": close - lower_pad,
        }
    )
    sig = falling_wedge_breakout(df, tol=0.004)
    assert bool(sig.iloc[-1]) is True
