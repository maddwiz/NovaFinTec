import pandas as pd

from aion.brain.indicators import inside_bar_breakdown, inside_bar_breakout


def test_inside_bar_breakout_detects_pattern():
    df = pd.DataFrame(
        {
            "high": [100.0, 102.0, 101.0, 103.2],
            "low": [98.0, 99.0, 99.5, 99.2],
            "close": [99.4, 100.2, 100.0, 103.4],
        }
    )
    sig = inside_bar_breakout(df)
    assert bool(sig.iloc[-1]) is True
    assert bool(sig.iloc[-2]) is False


def test_inside_bar_breakdown_detects_pattern():
    df = pd.DataFrame(
        {
            "high": [100.0, 102.0, 101.0, 100.6],
            "low": [98.0, 99.0, 99.5, 98.6],
            "close": [99.4, 100.2, 100.0, 98.4],
        }
    )
    sig = inside_bar_breakdown(df)
    assert bool(sig.iloc[-1]) is True
    assert bool(sig.iloc[-2]) is False
