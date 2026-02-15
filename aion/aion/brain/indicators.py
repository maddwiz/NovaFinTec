import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    gain = up.ewm(alpha=1 / length, adjust=False).mean()
    loss = down.ewm(alpha=1 / length, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    tr = pd.concat(
        [
            (df["high"] - df["low"]),
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_s = tr.ewm(alpha=1 / length, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / length, adjust=False).mean() / (atr_s + 1e-9))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / length, adjust=False).mean() / (atr_s + 1e-9))

    dx = ((plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9)) * 100
    return dx.ewm(alpha=1 / length, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(series: pd.Series, length: int = 20, std_mult: float = 2.0):
    mid = sma(series, length)
    std = series.rolling(length).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    width = (upper - lower) / (mid.abs() + 1e-9)
    return upper, mid, lower, width


def stochastic(df: pd.DataFrame, length: int = 14, smooth: int = 3):
    lowest = df["low"].rolling(length).min()
    highest = df["high"].rolling(length).max()
    k = 100 * ((df["close"] - lowest) / ((highest - lowest) + 1e-9))
    d = k.rolling(smooth).mean()
    return k, d


def obv(df: pd.DataFrame) -> pd.Series:
    vol = df.get("volume")
    if vol is None:
        return pd.Series(0.0, index=df.index)
    sign = np.sign(df["close"].diff().fillna(0))
    return (sign * vol.fillna(0)).cumsum()


def cci(df: pd.DataFrame, length: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    ma = tp.rolling(length).mean()
    md = (tp - ma).abs().rolling(length).mean()
    return (tp - ma) / (0.015 * (md + 1e-9))


def mfi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    vol = df.get("volume")
    if vol is None:
        return pd.Series(50.0, index=df.index)

    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    rmf = tp * vol.fillna(0)

    up = pd.Series(np.where(tp > tp.shift(1), rmf, 0.0), index=df.index)
    dn = pd.Series(np.where(tp < tp.shift(1), rmf, 0.0), index=df.index)

    up_sum = up.rolling(length).sum()
    dn_sum = dn.rolling(length).sum()
    mr = up_sum / (dn_sum + 1e-9)
    return 100 - (100 / (1 + mr))


def roc(series: pd.Series, length: int = 10) -> pd.Series:
    prev = series.shift(length)
    return (series - prev) / (prev.abs() + 1e-9)


# Candlestick patterns

def is_bull_engulf(df: pd.DataFrame) -> pd.Series:
    o, c = df["open"], df["close"]
    prev_o, prev_c = o.shift(), c.shift()
    return (c > o) & (prev_c < prev_o) & (c >= prev_o) & (o <= prev_c)


def is_bear_engulf(df: pd.DataFrame) -> pd.Series:
    o, c = df["open"], df["close"]
    prev_o, prev_c = o.shift(), c.shift()
    return (c < o) & (prev_c > prev_o) & (c <= prev_o) & (o >= prev_c)


def is_hammer(df: pd.DataFrame, body_ratio: float = 0.33) -> pd.Series:
    o, c, h, l = df["open"], df["close"], df["high"], df["low"]
    body = (c - o).abs()
    range_ = (h - l).replace(0, np.nan)
    lower_wick = np.minimum(o, c) - l
    upper_wick = h - np.maximum(o, c)
    return (lower_wick > (range_ * 0.45)) & (upper_wick < body) & (body < (range_ * body_ratio))


def is_shooting_star(df: pd.DataFrame, body_ratio: float = 0.33) -> pd.Series:
    o, c, h, l = df["open"], df["close"], df["high"], df["low"]
    body = (c - o).abs()
    range_ = (h - l).replace(0, np.nan)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    return (upper_wick > (range_ * 0.45)) & (lower_wick < body) & (body < (range_ * body_ratio))


def is_doji(df: pd.DataFrame, max_body_pct: float = 0.12) -> pd.Series:
    body = (df["close"] - df["open"]).abs()
    range_ = (df["high"] - df["low"]).replace(0, np.nan)
    return (body / range_) <= max_body_pct


def is_morning_star(df: pd.DataFrame) -> pd.Series:
    o, c = df["open"], df["close"]
    body = (c - o).abs()
    prev_body = body.shift(1)
    prev2_body = body.shift(2)
    bearish_first = c.shift(2) < o.shift(2)
    small_middle = prev_body < (prev2_body * 0.7)
    bullish_last = c > o
    reclaim = c >= ((o.shift(2) + c.shift(2)) / 2)
    return bearish_first & small_middle & bullish_last & reclaim


def is_evening_star(df: pd.DataFrame) -> pd.Series:
    o, c = df["open"], df["close"]
    body = (c - o).abs()
    prev_body = body.shift(1)
    prev2_body = body.shift(2)
    bullish_first = c.shift(2) > o.shift(2)
    small_middle = prev_body < (prev2_body * 0.7)
    bearish_last = c < o
    reclaim = c <= ((o.shift(2) + c.shift(2)) / 2)
    return bullish_first & small_middle & bearish_last & reclaim


def is_bull_harami(df: pd.DataFrame) -> pd.Series:
    o, c = df["open"], df["close"]
    prev_o, prev_c = o.shift(1), c.shift(1)
    prev_bear = prev_c < prev_o
    curr_bull = c > o
    body_inside = (o >= prev_c) & (c <= prev_o)
    return prev_bear & curr_bull & body_inside


def is_bear_harami(df: pd.DataFrame) -> pd.Series:
    o, c = df["open"], df["close"]
    prev_o, prev_c = o.shift(1), c.shift(1)
    prev_bull = prev_c > prev_o
    curr_bear = c < o
    body_inside = (o <= prev_c) & (c >= prev_o)
    return prev_bull & curr_bear & body_inside


def is_three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    o, c, h = df["open"], df["close"], df["high"]
    b0 = c > o
    b1 = c.shift(1) > o.shift(1)
    b2 = c.shift(2) > o.shift(2)
    higher_closes = (c > c.shift(1)) & (c.shift(1) > c.shift(2))
    opens_within_prev_body = (o <= c.shift(1)) & (o >= o.shift(1))
    short_upper_wick = (h - c) <= (c - o).abs() * 0.6
    return b0 & b1 & b2 & higher_closes & opens_within_prev_body & short_upper_wick


def is_three_black_crows(df: pd.DataFrame) -> pd.Series:
    o, c, l = df["open"], df["close"], df["low"]
    b0 = c < o
    b1 = c.shift(1) < o.shift(1)
    b2 = c.shift(2) < o.shift(2)
    lower_closes = (c < c.shift(1)) & (c.shift(1) < c.shift(2))
    opens_within_prev_body = (o >= c.shift(1)) & (o <= o.shift(1))
    short_lower_wick = (c - l) <= (c - o).abs() * 0.6
    return b0 & b1 & b2 & lower_closes & opens_within_prev_body & short_lower_wick


# Structural patterns

def breakout_up(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    prior_high = df["high"].shift(1).rolling(lookback).max()
    return df["close"] > prior_high


def breakout_down(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    prior_low = df["low"].shift(1).rolling(lookback).min()
    return df["close"] < prior_low


def _double_top_window(arr: np.ndarray, tolerance: float = 0.006) -> float:
    if len(arr) < 12:
        return 0.0
    split = len(arr) // 2
    left_peak = np.max(arr[:split])
    right_peak = np.max(arr[split:])
    similar = abs(left_peak - right_peak) / (max(left_peak, right_peak) + 1e-9) <= tolerance
    confirmed = arr[-1] < min(left_peak, right_peak) * (1.0 - 0.004)
    return float(similar and confirmed)


def _double_bottom_window(arr: np.ndarray, tolerance: float = 0.006) -> float:
    if len(arr) < 12:
        return 0.0
    split = len(arr) // 2
    left_trough = np.min(arr[:split])
    right_trough = np.min(arr[split:])
    similar = abs(left_trough - right_trough) / (abs(min(left_trough, right_trough)) + 1e-9) <= tolerance
    confirmed = arr[-1] > max(left_trough, right_trough) * (1.0 + 0.004)
    return float(similar and confirmed)


def double_top(close: pd.Series, lookback: int = 24) -> pd.Series:
    return close.rolling(lookback).apply(_double_top_window, raw=True).fillna(0.0) > 0.5


def double_bottom(close: pd.Series, lookback: int = 24) -> pd.Series:
    return close.rolling(lookback).apply(_double_bottom_window, raw=True).fillna(0.0) > 0.5


def _head_and_shoulders_top_window(
    arr: np.ndarray,
    shoulder_tol: float = 0.025,
    min_head_height: float = 0.012,
    neckline_break: float = 0.004,
) -> float:
    if len(arr) < 18:
        return 0.0
    n = len(arr)
    seg = max(5, n // 3)
    left = arr[:seg]
    mid = arr[seg : 2 * seg]
    right = arr[2 * seg :]
    if len(mid) < 4 or len(right) < 4:
        return 0.0

    lpk = float(np.max(left))
    hpk = float(np.max(mid))
    rpk = float(np.max(right))

    shoulders_similar = abs(lpk - rpk) / (max(lpk, rpk) + 1e-9) <= shoulder_tol
    head_dominant = (hpk > lpk * (1.0 + min_head_height)) and (hpk > rpk * (1.0 + min_head_height))

    lt = float(np.min(arr[max(0, seg // 2) : min(n, seg + seg // 2)]))
    rt = float(np.min(arr[min(n - 1, seg + seg // 2) :]))
    neckline = 0.5 * (lt + rt)
    confirmed = float(arr[-1]) < neckline * (1.0 - neckline_break)
    return float(shoulders_similar and head_dominant and confirmed)


def _inverse_head_and_shoulders_bottom_window(
    arr: np.ndarray,
    shoulder_tol: float = 0.025,
    min_head_depth: float = 0.012,
    neckline_break: float = 0.004,
) -> float:
    if len(arr) < 18:
        return 0.0
    n = len(arr)
    seg = max(5, n // 3)
    left = arr[:seg]
    mid = arr[seg : 2 * seg]
    right = arr[2 * seg :]
    if len(mid) < 4 or len(right) < 4:
        return 0.0

    ltr = float(np.min(left))
    htr = float(np.min(mid))
    rtr = float(np.min(right))

    shoulders_similar = abs(ltr - rtr) / (abs(min(ltr, rtr)) + 1e-9) <= shoulder_tol
    head_dominant = (htr < ltr * (1.0 - min_head_depth)) and (htr < rtr * (1.0 - min_head_depth))

    lp = float(np.max(arr[max(0, seg // 2) : min(n, seg + seg // 2)]))
    rp = float(np.max(arr[min(n - 1, seg + seg // 2) :]))
    neckline = 0.5 * (lp + rp)
    confirmed = float(arr[-1]) > neckline * (1.0 + neckline_break)
    return float(shoulders_similar and head_dominant and confirmed)


def head_and_shoulders_top(close: pd.Series, lookback: int = 30) -> pd.Series:
    return (
        close.rolling(lookback)
        .apply(_head_and_shoulders_top_window, raw=True)
        .fillna(0.0)
        .astype(float)
        > 0.5
    )


def inverse_head_and_shoulders_bottom(close: pd.Series, lookback: int = 30) -> pd.Series:
    return (
        close.rolling(lookback)
        .apply(_inverse_head_and_shoulders_bottom_window, raw=True)
        .fillna(0.0)
        .astype(float)
        > 0.5
    )


def bull_flag_breakout(df: pd.DataFrame, pullback: int = 6, trend_span: int = 24) -> pd.Series:
    close = df["close"]
    trend_up = close > ema(close, trend_span)
    pullback = close.diff().rolling(pullback).sum() < 0
    flag_range = (df["high"].rolling(6).max() - df["low"].rolling(6).min()) / (close.abs() + 1e-9)
    tight_flag = flag_range < 0.035
    breakout = close > df["high"].shift(1).rolling(12).max()
    return trend_up & pullback & tight_flag & breakout


def bear_flag_breakdown(df: pd.DataFrame, pullback: int = 6, trend_span: int = 24) -> pd.Series:
    close = df["close"]
    trend_down = close < ema(close, trend_span)
    pullup = close.diff().rolling(pullback).sum() > 0
    flag_range = (df["high"].rolling(6).max() - df["low"].rolling(6).min()) / (close.abs() + 1e-9)
    tight_flag = flag_range < 0.035
    breakdown = close < df["low"].shift(1).rolling(12).min()
    return trend_down & pullup & tight_flag & breakdown


def ascending_triangle_breakout(df: pd.DataFrame, tol: float = 0.004) -> pd.Series:
    highs_now = df["high"].rolling(6).max()
    highs_prev = highs_now.shift(6)
    flat_top = (highs_now - highs_prev).abs() / (highs_now.abs() + 1e-9) <= tol

    lows_now = df["low"].rolling(6).min()
    lows_prev = lows_now.shift(6)
    rising_lows = lows_now > (lows_prev * (1.0 + tol * 0.45))

    breakout = df["close"] > df["high"].shift(1).rolling(18).max()
    return flat_top & rising_lows & breakout


def descending_triangle_breakdown(df: pd.DataFrame, tol: float = 0.004) -> pd.Series:
    lows_now = df["low"].rolling(6).min()
    lows_prev = lows_now.shift(6)
    flat_bottom = (lows_now - lows_prev).abs() / (lows_now.abs() + 1e-9) <= tol

    highs_now = df["high"].rolling(6).max()
    highs_prev = highs_now.shift(6)
    falling_highs = highs_now < (highs_prev * (1.0 - tol * 0.45))

    breakdown = df["close"] < df["low"].shift(1).rolling(18).min()
    return flat_bottom & falling_highs & breakdown


def swing_high_low(close: pd.Series, lookback: int = 60):
    recent = close.iloc[-lookback:]
    return recent.max(), recent.min()


def fib_levels(high: float, low: float):
    diff = high - low
    return {
        "0.236": high - 0.236 * diff,
        "0.382": high - 0.382 * diff,
        "0.500": high - 0.500 * diff,
        "0.618": high - 0.618 * diff,
        "0.786": high - 0.786 * diff,
    }
