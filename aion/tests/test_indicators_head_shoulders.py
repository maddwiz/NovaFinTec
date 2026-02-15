import pandas as pd

from aion.brain.indicators import head_and_shoulders_top, inverse_head_and_shoulders_bottom


def test_head_and_shoulders_top_detects_constructed_pattern():
    close = pd.Series(
        [
            100.0,
            101.0,
            102.0,
            103.0,
            105.0,  # left shoulder
            103.0,
            101.0,  # trough
            104.0,
            106.0,
            111.0,  # head
            109.0,
            107.0,
            102.0,  # trough
            105.0,  # right shoulder
            104.0,
            103.0,
            100.0,
            98.0,  # neckline break
        ]
    )
    sig = head_and_shoulders_top(close, lookback=18)
    assert bool(sig.iloc[-1]) is True


def test_inverse_head_and_shoulders_bottom_detects_constructed_pattern():
    close = pd.Series(
        [
            100.0,
            99.0,
            98.0,
            97.0,
            95.0,  # left shoulder trough
            97.0,
            99.0,  # peak
            96.0,
            94.0,
            90.0,  # head trough
            92.0,
            95.0,
            99.0,  # peak
            95.0,  # right shoulder trough
            96.0,
            97.0,
            100.0,
            102.0,  # neckline break
        ]
    )
    sig = inverse_head_and_shoulders_bottom(close, lookback=18)
    assert bool(sig.iloc[-1]) is True
