from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import (
    adx,
    ascending_triangle_breakout,
    atr,
    bear_flag_breakdown,
    bollinger_bands,
    breakout_down,
    breakout_up,
    bull_flag_breakout,
    cci,
    descending_triangle_breakdown,
    double_bottom,
    double_top,
    ema,
    fib_levels,
    is_bear_harami,
    is_bear_engulf,
    is_bull_harami,
    is_bull_engulf,
    is_doji,
    is_evening_star,
    is_hammer,
    is_morning_star,
    is_shooting_star,
    is_three_black_crows,
    is_three_white_soldiers,
    macd,
    mfi,
    obv,
    roc,
    rsi,
    stochastic,
)


def compute_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    out = df.copy()

    out["ema_fast"] = ema(out["close"], cfg.EMA_FAST)
    out["ema_slow"] = ema(out["close"], cfg.EMA_SLOW)
    out["rsi"] = rsi(out["close"], cfg.RSI_LEN)
    out["atr"] = atr(out, cfg.ATR_LEN)
    out["adx"] = adx(out, cfg.ADX_LEN)
    out["cci"] = cci(out, 20)
    out["mfi"] = mfi(out, 14)
    out["roc"] = roc(out["close"], 10)

    out["macd"], out["macd_signal"], out["macd_hist"] = macd(out["close"])
    out["bb_upper"], out["bb_mid"], out["bb_lower"], out["bb_width"] = bollinger_bands(out["close"], cfg.BB_LEN, cfg.BB_STD)
    out["stoch_k"], out["stoch_d"] = stochastic(out, cfg.STOCH_LEN, cfg.STOCH_SMOOTH)
    out["obv"] = obv(out)
    out["obv_prev"] = out["obv"].shift(1)
    div = _divergence_features(out, cfg)
    out["rsi_bull_div"] = div["rsi_bull_div"]
    out["rsi_bear_div"] = div["rsi_bear_div"]
    out["obv_bull_div"] = div["obv_bull_div"]
    out["obv_bear_div"] = div["obv_bear_div"]

    out["bull_engulf"] = is_bull_engulf(out)
    out["bear_engulf"] = is_bear_engulf(out)
    out["hammer"] = is_hammer(out)
    out["shooting_star"] = is_shooting_star(out)
    out["doji"] = is_doji(out)
    out["morning_star"] = is_morning_star(out)
    out["evening_star"] = is_evening_star(out)
    out["bull_harami"] = is_bull_harami(out)
    out["bear_harami"] = is_bear_harami(out)
    out["three_white_soldiers"] = is_three_white_soldiers(out)
    out["three_black_crows"] = is_three_black_crows(out)

    out["breakout_up"] = breakout_up(out, lookback=20)
    out["breakout_down"] = breakout_down(out, lookback=20)
    out["double_top"] = double_top(out["close"], lookback=24)
    out["double_bottom"] = double_bottom(out["close"], lookback=24)
    out["bull_flag_breakout"] = bull_flag_breakout(out)
    out["bear_flag_breakdown"] = bear_flag_breakdown(out)
    out["ascending_triangle_breakout"] = ascending_triangle_breakout(out)
    out["descending_triangle_breakdown"] = descending_triangle_breakdown(out)

    out["macd_cross_up"] = (out["macd"] > out["macd_signal"]) & (out["macd"].shift(1) <= out["macd_signal"].shift(1))
    out["macd_cross_down"] = (out["macd"] < out["macd_signal"]) & (out["macd"].shift(1) >= out["macd_signal"].shift(1))
    out["stoch_cross_up"] = (out["stoch_k"] > out["stoch_d"]) & (out["stoch_k"].shift(1) <= out["stoch_d"].shift(1))
    out["stoch_cross_down"] = (out["stoch_k"] < out["stoch_d"]) & (out["stoch_k"].shift(1) >= out["stoch_d"].shift(1))
    out["rsi_cross_50_up"] = (out["rsi"] > 50) & (out["rsi"].shift(1) <= 50)
    out["rsi_cross_50_down"] = (out["rsi"] < 50) & (out["rsi"].shift(1) >= 50)

    vol = out.get("volume")
    if vol is None:
        out["volume_rel"] = 1.0
    else:
        out["volume_rel"] = vol / (vol.rolling(20).mean() + 1e-9)

    out["pivot_high_20"] = out["high"].shift(1).rolling(20).max()
    out["pivot_low_20"] = out["low"].shift(1).rolling(20).min()
    out["dist_high_20"] = (out["pivot_high_20"] - out["close"]) / (out["close"].abs() + 1e-9)
    out["dist_low_20"] = (out["close"] - out["pivot_low_20"]) / (out["close"].abs() + 1e-9)

    out["atr_pct"] = out["atr"] / (out["close"].abs() + 1e-9)
    out["ema_gap_pct"] = (out["ema_fast"] - out["ema_slow"]) / (out["close"].abs() + 1e-9)

    trend_cond = (out["adx"] >= cfg.REGIME_ADX_TREND_MIN) & (out["ema_gap_pct"].abs() > 0.001)
    chop_cond = (out["atr_pct"] >= cfg.REGIME_ATR_PCT_HIGH) & (out["adx"] < cfg.REGIME_ADX_TREND_MIN)
    squeeze_cond = out["bb_width"] <= cfg.REGIME_BB_SQUEEZE_PCT
    calm_cond = out["atr_pct"] <= cfg.REGIME_ATR_PCT_LOW

    out["regime"] = np.select(
        [chop_cond, trend_cond, squeeze_cond, calm_cond],
        ["high_vol_chop", "trending", "squeeze", "calm_range"],
        default="mixed",
    )

    return out


def fib_confluence_signal(row_price: float, high: float, low: float, tol: float):
    levels = fib_levels(high, low)
    for name, level in levels.items():
        dist = abs(row_price - level) / max(1e-6, row_price)
        if dist <= tol:
            return True, name, level
    return False, None, None


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _divergence_features(df: pd.DataFrame, cfg) -> dict[str, pd.Series]:
    empty = pd.Series(False, index=df.index)
    if df.empty or "close" not in df.columns:
        return {
            "rsi_bull_div": empty,
            "rsi_bear_div": empty,
            "obv_bull_div": empty,
            "obv_bear_div": empty,
        }

    lb = max(3, int(getattr(cfg, "DIVERGENCE_LOOKBACK", 8)))
    price_move_min = max(0.0, float(getattr(cfg, "DIVERGENCE_PRICE_MOVE_MIN", 0.006)))
    rsi_delta_min = max(0.0, float(getattr(cfg, "DIVERGENCE_RSI_DELTA_MIN", 4.0)))
    obv_delta_min = max(0.0, float(getattr(cfg, "DIVERGENCE_OBV_DELTA_MIN", 0.0)))

    close = pd.to_numeric(df.get("close"), errors="coerce")
    prev_close = close.shift(lb)
    price_chg = (close - prev_close) / (prev_close.abs() + 1e-9)
    bull_price = price_chg <= -price_move_min
    bear_price = price_chg >= price_move_min

    rsi_s = pd.to_numeric(df.get("rsi"), errors="coerce")
    rsi_chg = rsi_s - rsi_s.shift(lb)
    rsi_bull = (bull_price & (rsi_chg >= rsi_delta_min)).fillna(False)
    rsi_bear = (bear_price & (rsi_chg <= -rsi_delta_min)).fillna(False)

    obv_s = pd.to_numeric(df.get("obv"), errors="coerce")
    obv_chg = obv_s - obv_s.shift(lb)
    obv_bull = (bull_price & (obv_chg >= obv_delta_min)).fillna(False)
    obv_bear = (bear_price & (obv_chg <= -obv_delta_min)).fillna(False)

    return {
        "rsi_bull_div": rsi_bull.astype(bool),
        "rsi_bear_div": rsi_bear.astype(bool),
        "obv_bull_div": obv_bull.astype(bool),
        "obv_bear_div": obv_bear.astype(bool),
    }


def _trend_component(row: pd.Series, price: float):
    long = 0.0
    short = 0.0
    reasons_l = []
    reasons_s = []

    ema_fast = float(row["ema_fast"])
    ema_slow = float(row["ema_slow"])
    adx_v = float(row["adx"])
    macd_hist = float(row["macd_hist"])

    if ema_fast > ema_slow:
        v = 0.50 + 0.40 * _clamp01((adx_v - 12) / 25)
        long += v
        reasons_l.append("Trend up (EMA/ADX)")
    elif ema_fast < ema_slow:
        v = 0.50 + 0.40 * _clamp01((adx_v - 12) / 25)
        short += v
        reasons_s.append("Trend down (EMA/ADX)")

    if macd_hist > 0:
        long += 0.25
        reasons_l.append("MACD momentum up")
    elif macd_hist < 0:
        short += 0.25
        reasons_s.append("MACD momentum down")

    return long, short, reasons_l, reasons_s


def _mean_reversion_component(row: pd.Series, price: float):
    long = 0.0
    short = 0.0
    reasons_l = []
    reasons_s = []

    rsi_v = float(row["rsi"])
    stoch_k = float(row["stoch_k"])
    bb_upper = float(row["bb_upper"])
    bb_lower = float(row["bb_lower"])

    if rsi_v < 32 and stoch_k < 25:
        long += 0.55
        reasons_l.append("Oversold RSI/Stoch")
    if rsi_v > 68 and stoch_k > 75:
        short += 0.55
        reasons_s.append("Overbought RSI/Stoch")

    if price < bb_lower:
        long += 0.30
        reasons_l.append("Below lower Bollinger")
    if price > bb_upper:
        short += 0.30
        reasons_s.append("Above upper Bollinger")

    return long, short, reasons_l, reasons_s


def _breakout_component(row: pd.Series):
    long = 0.0
    short = 0.0
    reasons_l = []
    reasons_s = []

    if bool(row["breakout_up"]):
        long += 0.75
        reasons_l.append("Price breakout up")
    if bool(row["breakout_down"]):
        short += 0.75
        reasons_s.append("Price breakout down")

    if bool(row["double_bottom"]):
        long += 0.45
        reasons_l.append("Double bottom")
    if bool(row["double_top"]):
        short += 0.45
        reasons_s.append("Double top")

    return long, short, reasons_l, reasons_s


def _confirmation_component(row: pd.Series):
    long = 0.0
    short = 0.0
    reasons_l = []
    reasons_s = []

    if bool(row.get("macd_cross_up", False)):
        long += 0.34
        reasons_l.append("MACD bullish cross")
    if bool(row.get("macd_cross_down", False)):
        short += 0.34
        reasons_s.append("MACD bearish cross")

    if bool(row.get("stoch_cross_up", False)) and float(row.get("stoch_k", 50.0)) < 40:
        long += 0.24
        reasons_l.append("Stoch bullish cross")
    if bool(row.get("stoch_cross_down", False)) and float(row.get("stoch_k", 50.0)) > 60:
        short += 0.24
        reasons_s.append("Stoch bearish cross")

    if bool(row.get("rsi_cross_50_up", False)):
        long += 0.20
        reasons_l.append("RSI crossed above 50")
    if bool(row.get("rsi_cross_50_down", False)):
        short += 0.20
        reasons_s.append("RSI crossed below 50")

    cci_v = float(row.get("cci", 0.0))
    if cci_v > 100:
        long += 0.18
        reasons_l.append("CCI strength > 100")
    elif cci_v < -100:
        short += 0.18
        reasons_s.append("CCI weakness < -100")

    mfi_v = float(row.get("mfi", 50.0))
    if mfi_v < 20:
        long += 0.18
        reasons_l.append("MFI oversold")
    elif mfi_v > 80:
        short += 0.18
        reasons_s.append("MFI overbought")

    roc_v = float(row.get("roc", 0.0))
    if roc_v > 0.008:
        long += 0.14
        reasons_l.append("Positive rate-of-change")
    elif roc_v < -0.008:
        short += 0.14
        reasons_s.append("Negative rate-of-change")

    if bool(row.get("rsi_bull_div", False)):
        long += 0.28
        reasons_l.append("Bullish RSI divergence")
    if bool(row.get("rsi_bear_div", False)):
        short += 0.28
        reasons_s.append("Bearish RSI divergence")
    if bool(row.get("obv_bull_div", False)):
        long += 0.24
        reasons_l.append("Bullish OBV divergence")
    if bool(row.get("obv_bear_div", False)):
        short += 0.24
        reasons_s.append("Bearish OBV divergence")

    vol_rel = float(row.get("volume_rel", 1.0))
    if vol_rel >= 1.6:
        if float(row.get("close", 0.0)) >= float(row.get("open", 0.0)):
            long += 0.16
            reasons_l.append("Volume expansion on up bar")
        else:
            short += 0.16
            reasons_s.append("Volume expansion on down bar")

    dist_high = float(row.get("dist_high_20", 1.0))
    dist_low = float(row.get("dist_low_20", 1.0))
    if dist_high <= 0.004 and bool(row.get("breakout_up", False)):
        long += 0.20
        reasons_l.append("20-bar resistance breakout")
    if dist_low <= 0.004 and bool(row.get("breakout_down", False)):
        short += 0.20
        reasons_s.append("20-bar support breakdown")

    return long, short, reasons_l, reasons_s


def _pattern_component(row: pd.Series, price: float, high: float, low: float, cfg):
    long = 0.0
    short = 0.0
    reasons_l = []
    reasons_s = []

    if bool(row["bull_engulf"]) or bool(row["hammer"]) or bool(row["morning_star"]) or bool(row.get("bull_harami", False)):
        long += 0.65
        reasons_l.append("Bullish candlestick pattern")
    if bool(row["bear_engulf"]) or bool(row["shooting_star"]) or bool(row["evening_star"]) or bool(row.get("bear_harami", False)):
        short += 0.65
        reasons_s.append("Bearish candlestick pattern")

    if bool(row.get("three_white_soldiers", False)):
        long += 0.52
        reasons_l.append("Three white soldiers")
    if bool(row.get("three_black_crows", False)):
        short += 0.52
        reasons_s.append("Three black crows")

    if bool(row.get("bull_flag_breakout", False)):
        long += 0.58
        reasons_l.append("Bull flag breakout")
    if bool(row.get("bear_flag_breakdown", False)):
        short += 0.58
        reasons_s.append("Bear flag breakdown")

    if bool(row.get("ascending_triangle_breakout", False)):
        long += 0.62
        reasons_l.append("Ascending triangle breakout")
    if bool(row.get("descending_triangle_breakdown", False)):
        short += 0.62
        reasons_s.append("Descending triangle breakdown")

    if bool(row["doji"]):
        long += 0.12
        short += 0.12
        reasons_l.append("Doji indecision")
        reasons_s.append("Doji indecision")

    is_fib, fib_name, fib_level = fib_confluence_signal(price, high, low, cfg.FIB_TOLERANCE)
    if is_fib:
        if price >= fib_level:
            long += 0.30
            reasons_l.append(f"Fib support {fib_name}")
        else:
            short += 0.30
            reasons_s.append(f"Fib resistance {fib_name}")

    obv_prev = float(row.get("obv_prev", row["obv"]))
    obv_now = float(row["obv"])
    if obv_now > obv_prev:
        long += 0.15
        reasons_l.append("OBV rising")
    elif obv_now < obv_prev:
        short += 0.15
        reasons_s.append("OBV falling")

    return long, short, reasons_l, reasons_s


def _confluence_component(row: pd.Series):
    close_v = float(row.get("close", 0.0))
    open_v = float(row.get("open", close_v))
    ema_fast = float(row.get("ema_fast", close_v))
    ema_slow = float(row.get("ema_slow", close_v))

    bull_pattern = any(
        bool(row.get(k, False))
        for k in [
            "bull_engulf",
            "hammer",
            "morning_star",
            "bull_harami",
            "three_white_soldiers",
            "bull_flag_breakout",
            "ascending_triangle_breakout",
        ]
    )
    bear_pattern = any(
        bool(row.get(k, False))
        for k in [
            "bear_engulf",
            "shooting_star",
            "evening_star",
            "bear_harami",
            "three_black_crows",
            "bear_flag_breakdown",
            "descending_triangle_breakdown",
        ]
    )

    votes_l = [
        ema_fast > ema_slow,
        float(row.get("macd_hist", 0.0)) > 0,
        float(row.get("rsi", 50.0)) >= 50,
        float(row.get("stoch_k", 50.0)) >= float(row.get("stoch_d", 50.0)),
        float(row.get("cci", 0.0)) > 0,
        float(row.get("mfi", 50.0)) >= 50,
        float(row.get("roc", 0.0)) >= 0,
        float(row.get("obv", 0.0)) >= float(row.get("obv_prev", 0.0)),
        close_v >= float(row.get("bb_mid", close_v)),
        (float(row.get("volume_rel", 1.0)) >= 1.0 and close_v >= open_v),
        bool(row.get("breakout_up", False) or row.get("double_bottom", False)),
        bull_pattern,
        bool(row.get("rsi_bull_div", False) or row.get("obv_bull_div", False)),
    ]

    votes_s = [
        ema_fast < ema_slow,
        float(row.get("macd_hist", 0.0)) < 0,
        float(row.get("rsi", 50.0)) <= 50,
        float(row.get("stoch_k", 50.0)) <= float(row.get("stoch_d", 50.0)),
        float(row.get("cci", 0.0)) < 0,
        float(row.get("mfi", 50.0)) <= 50,
        float(row.get("roc", 0.0)) <= 0,
        float(row.get("obv", 0.0)) <= float(row.get("obv_prev", 0.0)),
        close_v <= float(row.get("bb_mid", close_v)),
        (float(row.get("volume_rel", 1.0)) >= 1.0 and close_v <= open_v),
        bool(row.get("breakout_down", False) or row.get("double_top", False)),
        bear_pattern,
        bool(row.get("rsi_bear_div", False) or row.get("obv_bear_div", False)),
    ]

    long_votes = sum(1 for v in votes_l if v)
    short_votes = sum(1 for v in votes_s if v)
    n_votes = max(1, len(votes_l))

    confluence_l = _clamp01(long_votes / n_votes)
    confluence_s = _clamp01(short_votes / n_votes)
    reasons_l = [f"Confluence long {long_votes}/{n_votes}"]
    reasons_s = [f"Confluence short {short_votes}/{n_votes}"]

    trend_structure_l = (ema_fast > ema_slow) and bool(
        row.get("breakout_up", False)
        or row.get("bull_flag_breakout", False)
        or row.get("ascending_triangle_breakout", False)
    )
    trend_structure_s = (ema_fast < ema_slow) and bool(
        row.get("breakout_down", False)
        or row.get("bear_flag_breakdown", False)
        or row.get("descending_triangle_breakdown", False)
    )
    if trend_structure_l:
        confluence_l = _clamp01(confluence_l + 0.08)
        reasons_l.append("Trend + structure alignment")
    if trend_structure_s:
        confluence_s = _clamp01(confluence_s + 0.08)
        reasons_s.append("Trend + structure alignment")

    disagreement = min(confluence_l, confluence_s)
    return confluence_l, confluence_s, reasons_l, reasons_s, long_votes, short_votes, disagreement


def _regime_weights(regime: str):
    if regime == "trending":
        return {"trend": 0.40, "breakout": 0.24, "mean_reversion": 0.06, "pattern": 0.18, "confirm": 0.12}
    if regime == "squeeze":
        return {"trend": 0.16, "breakout": 0.36, "mean_reversion": 0.09, "pattern": 0.23, "confirm": 0.16}
    if regime == "calm_range":
        return {"trend": 0.14, "breakout": 0.08, "mean_reversion": 0.42, "pattern": 0.22, "confirm": 0.14}
    if regime == "high_vol_chop":
        return {"trend": 0.08, "breakout": 0.10, "mean_reversion": 0.26, "pattern": 0.16, "confirm": 0.10}
    return {"trend": 0.30, "breakout": 0.20, "mean_reversion": 0.18, "pattern": 0.20, "confirm": 0.12}


def score_signal(row: pd.Series, price: float, high: float, low: float, cfg):
    regime = str(row.get("regime", "mixed"))
    w = _regime_weights(regime)

    t_l, t_s, t_rl, t_rs = _trend_component(row, price)
    m_l, m_s, m_rl, m_rs = _mean_reversion_component(row, price)
    b_l, b_s, b_rl, b_rs = _breakout_component(row)
    p_l, p_s, p_rl, p_rs = _pattern_component(row, price, high, low, cfg)
    c_l, c_s, c_rl, c_rs = _confirmation_component(row)

    long_raw_base = (
        w["trend"] * t_l
        + w["mean_reversion"] * m_l
        + w["breakout"] * b_l
        + w["pattern"] * p_l
        + w["confirm"] * c_l
    )
    short_raw_base = (
        w["trend"] * t_s
        + w["mean_reversion"] * m_s
        + w["breakout"] * b_s
        + w["pattern"] * p_s
        + w["confirm"] * c_s
    )

    confl_l, confl_s, confl_rl, confl_rs, votes_l, votes_s, disagreement = _confluence_component(row)
    long_raw = long_raw_base * (0.72 + 0.56 * confl_l)
    short_raw = short_raw_base * (0.72 + 0.56 * confl_s)

    # When both sides are similarly strong, reduce conviction to avoid whipsaw.
    ambiguity_penalty = max(0.76, 1.0 - 0.30 * disagreement)
    long_raw *= ambiguity_penalty
    short_raw *= ambiguity_penalty

    if regime == "high_vol_chop":
        long_raw *= 0.62
        short_raw *= 0.62

    long_conf = _clamp01(long_raw)
    short_conf = _clamp01(short_raw)

    return {
        "regime": regime,
        "long_conf": long_conf,
        "short_conf": short_conf,
        "components": {
            "trend": {"long": t_l, "short": t_s},
            "mean_reversion": {"long": m_l, "short": m_s},
            "breakout": {"long": b_l, "short": b_s},
            "pattern": {"long": p_l, "short": p_s},
            "confirm": {"long": c_l, "short": c_s},
            "confluence": {"long": confl_l, "short": confl_s},
        },
        "long_reasons": (t_rl + m_rl + b_rl + p_rl + c_rl + confl_rl),
        "short_reasons": (t_rs + m_rs + b_rs + p_rs + c_rs + confl_rs),
        "pattern_hits_long": len(p_rl),
        "pattern_hits_short": len(p_rs),
        "indicator_hits_long": len(t_rl) + len(m_rl) + len(b_rl) + len(c_rl),
        "indicator_hits_short": len(t_rs) + len(m_rs) + len(b_rs) + len(c_rs),
        "confluence_long": confl_l,
        "confluence_short": confl_s,
        "confluence_votes_long": votes_l,
        "confluence_votes_short": votes_s,
    }


def multi_timeframe_alignment(df_1h: pd.DataFrame, df_4h: pd.DataFrame, side: str, cfg):
    if df_1h is None or df_4h is None or df_1h.empty or df_4h.empty:
        return 0.5, ["No MTF data"]

    if len(df_1h) < 30 or len(df_4h) < 30:
        return 0.5, ["Insufficient MTF bars"]

    c1 = df_1h["close"]
    c4 = df_4h["close"]
    e1f, e1s = ema(c1, cfg.EMA_FAST), ema(c1, cfg.EMA_SLOW)
    e4f, e4s = ema(c4, cfg.EMA_FAST), ema(c4, cfg.EMA_SLOW)

    slope1 = float(e1f.iloc[-1] - e1f.iloc[-4])
    slope4 = float(e4f.iloc[-1] - e4f.iloc[-4])

    up1 = e1f.iloc[-1] > e1s.iloc[-1] and slope1 > 0
    up4 = e4f.iloc[-1] > e4s.iloc[-1] and slope4 > 0
    dn1 = e1f.iloc[-1] < e1s.iloc[-1] and slope1 < 0
    dn4 = e4f.iloc[-1] < e4s.iloc[-1] and slope4 < 0

    reasons = []
    if side == "LONG":
        score = 0.35 + (0.30 if up1 else 0.0) + (0.35 if up4 else 0.0)
        if up1:
            reasons.append("1H uptrend")
        if up4:
            reasons.append("4H uptrend")
        if not reasons:
            reasons.append("MTF misaligned for long")
        return _clamp01(score), reasons

    if side == "SHORT":
        score = 0.35 + (0.30 if dn1 else 0.0) + (0.35 if dn4 else 0.0)
        if dn1:
            reasons.append("1H downtrend")
        if dn4:
            reasons.append("4H downtrend")
        if not reasons:
            reasons.append("MTF misaligned for short")
        return _clamp01(score), reasons

    return 0.5, ["No side"]


def _dynamic_thresholds(cfg, profile: dict | None):
    long_th = cfg.ENTRY_THRESHOLD_LONG
    short_th = cfg.ENTRY_THRESHOLD_SHORT
    opp_exit = cfg.OPPOSITE_EXIT_THRESHOLD
    if profile:
        long_th = float(profile.get("entry_threshold_long", long_th))
        short_th = float(profile.get("entry_threshold_short", short_th))
        opp_exit = float(profile.get("opposite_exit_threshold", opp_exit))
    return long_th, short_th, opp_exit


def _clamp_threshold(x: float) -> float:
    return max(0.40, min(0.92, float(x)))


def _regime_threshold_adjustments(regime: str, cfg):
    long_shift = 0.0
    short_shift = 0.0
    opp_shift = 0.0
    margin_shift = 0.0

    if regime == "trending":
        long_shift = cfg.REGIME_TH_SHIFT_TRENDING
        short_shift = cfg.REGIME_TH_SHIFT_TRENDING
        opp_shift = cfg.REGIME_OPP_EXIT_SHIFT_TRENDING
        margin_shift = cfg.REGIME_MARGIN_SHIFT_TRENDING
    elif regime == "squeeze":
        long_shift = cfg.REGIME_TH_SHIFT_SQUEEZE
        short_shift = cfg.REGIME_TH_SHIFT_SQUEEZE
        opp_shift = cfg.REGIME_OPP_EXIT_SHIFT_SQUEEZE
        margin_shift = cfg.REGIME_MARGIN_SHIFT_SQUEEZE
    elif regime == "calm_range":
        long_shift = cfg.REGIME_TH_SHIFT_CALM_RANGE
        short_shift = cfg.REGIME_TH_SHIFT_CALM_RANGE
        opp_shift = cfg.REGIME_OPP_EXIT_SHIFT_CALM_RANGE
        margin_shift = cfg.REGIME_MARGIN_SHIFT_CALM_RANGE
    elif regime == "high_vol_chop":
        long_shift = cfg.REGIME_TH_SHIFT_HIGH_VOL_CHOP
        short_shift = cfg.REGIME_TH_SHIFT_HIGH_VOL_CHOP
        opp_shift = cfg.REGIME_OPP_EXIT_SHIFT_HIGH_VOL_CHOP
        margin_shift = cfg.REGIME_MARGIN_SHIFT_HIGH_VOL_CHOP

    return long_shift, short_shift, opp_shift, margin_shift


def _adaptive_reason_factor(reasons: list[str], profile: dict | None, cfg):
    if not profile:
        return 1.0, 0
    raw = profile.get("reason_multipliers")
    if not isinstance(raw, dict):
        return 1.0, 0

    lut = {}
    for k, v in raw.items():
        try:
            lut[str(k).strip().lower()] = float(v)
        except Exception:
            continue
    if not lut:
        return 1.0, 0

    matches = []
    for r in reasons:
        key = str(r).strip().lower()
        if key in lut:
            matches.append(lut[key])

    if not matches:
        return 1.0, 0

    factor = float(sum(matches) / len(matches))
    factor = max(cfg.REASON_ADAPT_MIN_MULT, min(cfg.REASON_ADAPT_MAX_MULT, factor))
    return factor, len(matches)


def _regime_multipliers(regime: str, cfg):
    stop = cfg.STOP_ATR_MULT
    target = cfg.TARGET_ATR_MULT
    trail = cfg.TRAIL_ATR_MULT

    if regime == "trending":
        target *= 1.20
        trail *= 1.15
    elif regime == "squeeze":
        target *= 1.28
        stop *= 0.95
    elif regime == "calm_range":
        target *= 0.90
        stop *= 0.90
    elif regime == "high_vol_chop":
        stop *= 1.15
        target *= 0.80

    return stop, target, trail


def _apply_external_overlay(
    long_conf: float,
    short_conf: float,
    long_th: float,
    short_th: float,
    external: dict | None,
    cfg,
):
    if not external or not getattr(cfg, "EXT_SIGNAL_ENABLED", False):
        return long_conf, short_conf, long_th, short_th, None

    try:
        bias = float(external.get("bias", 0.0))
        conf = float(external.get("confidence", 0.0))
    except Exception:
        return long_conf, short_conf, long_th, short_th, None

    bias = max(-float(cfg.EXT_SIGNAL_MAX_BIAS), min(float(cfg.EXT_SIGNAL_MAX_BIAS), bias))
    conf = _clamp01(conf)
    signed = bias * conf
    if abs(signed) < 1e-9:
        return long_conf, short_conf, long_th, short_th, {"bias": bias, "confidence": conf, "signed": 0.0}

    long_conf = _clamp01(long_conf + float(cfg.EXT_SIGNAL_CONF_BOOST) * signed)
    short_conf = _clamp01(short_conf - float(cfg.EXT_SIGNAL_CONF_BOOST) * signed)
    long_th = _clamp_threshold(long_th - float(cfg.EXT_SIGNAL_THRESHOLD_SHIFT) * signed)
    short_th = _clamp_threshold(short_th + float(cfg.EXT_SIGNAL_THRESHOLD_SHIFT) * signed)

    return long_conf, short_conf, long_th, short_th, {"bias": bias, "confidence": conf, "signed": signed}


def build_trade_signal(
    row: pd.Series,
    price: float,
    high: float,
    low: float,
    cfg,
    profile: dict | None = None,
    external: dict | None = None,
):
    scored = score_signal(row, price, high, low, cfg)
    long_th, short_th, opp_exit = _dynamic_thresholds(cfg, profile)

    regime = str(scored["regime"])
    conf_l = float(scored.get("confluence_long", 0.0))
    conf_s = float(scored.get("confluence_short", 0.0))
    long_shift, short_shift, opp_shift, margin_shift = _regime_threshold_adjustments(regime, cfg)

    # Regime-aware threshold shifts with confluence-aware easing.
    long_th = _clamp_threshold(long_th + long_shift)
    short_th = _clamp_threshold(short_th + short_shift)
    long_th = _clamp_threshold(long_th - cfg.CONFLUENCE_BOOST_MAX * max(0.0, conf_l - cfg.CONFLUENCE_LONG_MIN))
    short_th = _clamp_threshold(short_th - cfg.CONFLUENCE_BOOST_MAX * max(0.0, conf_s - cfg.CONFLUENCE_SHORT_MIN))
    opp_exit = _clamp_threshold(opp_exit + opp_shift)
    margin_min = max(0.03, cfg.SIGNAL_MIN_MARGIN + margin_shift)

    long_conf = float(scored["long_conf"])
    short_conf = float(scored["short_conf"])
    long_conf, short_conf, long_th, short_th, ext = _apply_external_overlay(
        long_conf=long_conf,
        short_conf=short_conf,
        long_th=long_th,
        short_th=short_th,
        external=external,
        cfg=cfg,
    )

    side = None
    confidence = 0.0
    reasons = []

    if regime == "high_vol_chop":
        return {
            "side": None,
            "confidence": 0.0,
            "regime": regime,
            "long_conf": long_conf,
            "short_conf": short_conf,
            "reasons": [
                "Regime filter: high_vol_chop",
                f"Confluence L/S {conf_l:.2f}/{conf_s:.2f}",
            ],
            "pattern_hits": 0,
            "indicator_hits": 0,
            "confluence_long": conf_l,
            "confluence_short": conf_s,
            "opposite_exit_threshold": opp_exit,
            "stop_atr_mult": cfg.STOP_ATR_MULT,
            "target_atr_mult": cfg.TARGET_ATR_MULT,
            "trail_atr_mult": cfg.TRAIL_ATR_MULT,
            "entry_threshold_long": long_th,
            "entry_threshold_short": short_th,
            "margin_min": margin_min,
            "components": scored["components"],
            "external_bias": float(ext.get("bias", 0.0)) if ext else 0.0,
            "external_confidence": float(ext.get("confidence", 0.0)) if ext else 0.0,
        }

    margin = abs(long_conf - short_conf)
    if (
        long_conf >= long_th
        and long_conf > short_conf
        and margin >= margin_min
        and conf_l >= cfg.CONFLUENCE_LONG_MIN
    ):
        side = "LONG"
        confidence = long_conf
        reasons = scored["long_reasons"]
    elif (
        short_conf >= short_th
        and short_conf > long_conf
        and margin >= margin_min
        and conf_s >= cfg.CONFLUENCE_SHORT_MIN
    ):
        side = "SHORT"
        confidence = short_conf
        reasons = scored["short_reasons"]

    if side is None and margin >= margin_min:
        if long_conf >= long_th and conf_l < cfg.CONFLUENCE_LONG_MIN:
            reasons = reasons + [f"LONG blocked by low confluence ({conf_l:.2f})"]
        if short_conf >= short_th and conf_s < cfg.CONFLUENCE_SHORT_MIN:
            reasons = reasons + [f"SHORT blocked by low confluence ({conf_s:.2f})"]

    if side:
        # Penalize edge cases where opposite confluence is still elevated.
        opposite_conf = conf_s if side == "LONG" else conf_l
        if opposite_conf > 0.58:
            confidence = _clamp01(confidence * (1.0 - cfg.CONFLUENCE_PENALTY_MAX * (opposite_conf - 0.58) / 0.42))
            reasons = reasons + [f"Opposite confluence penalty ({opposite_conf:.2f})"]

    if side and cfg.REASON_ADAPT_ENABLED:
        factor, matched = _adaptive_reason_factor(reasons, profile, cfg)
        confidence = _clamp01(confidence * factor)
        if matched > 0 and abs(factor - 1.0) >= 0.015:
            reasons = reasons + [f"Adaptive reason factor {factor:.2f}"]

    if ext and abs(float(ext.get("signed", 0.0))) >= 0.01:
        reasons = reasons + [
            f"External overlay bias {float(ext.get('bias', 0.0)):+.2f} @ {float(ext.get('confidence', 0.0)):.2f}"
        ]

    stop_mult, target_mult, trail_mult = _regime_multipliers(regime, cfg)

    return {
        "side": side,
        "confidence": confidence,
        "regime": regime,
        "long_conf": long_conf,
        "short_conf": short_conf,
        "reasons": reasons,
        "pattern_hits": int(scored["pattern_hits_long"] if side == "LONG" else scored["pattern_hits_short"] if side == "SHORT" else 0),
        "indicator_hits": int(scored["indicator_hits_long"] if side == "LONG" else scored["indicator_hits_short"] if side == "SHORT" else 0),
        "confluence_long": conf_l,
        "confluence_short": conf_s,
        "confluence_votes_long": int(scored.get("confluence_votes_long", 0)),
        "confluence_votes_short": int(scored.get("confluence_votes_short", 0)),
        "opposite_exit_threshold": opp_exit,
        "stop_atr_mult": stop_mult,
        "target_atr_mult": target_mult,
        "trail_atr_mult": trail_mult,
        "entry_threshold_long": long_th,
        "entry_threshold_short": short_th,
        "margin_min": margin_min,
        "components": scored["components"],
        "external_bias": float(ext.get("bias", 0.0)) if ext else 0.0,
        "external_confidence": float(ext.get("confidence", 0.0)) if ext else 0.0,
    }


def opposite_confidence(side: str, signal: dict) -> float:
    if side == "LONG":
        return float(signal.get("short_conf", 0.0))
    if side == "SHORT":
        return float(signal.get("long_conf", 0.0))
    return 0.0


def confidence_tag(conf: float) -> str:
    if conf >= 0.78:
        return "high"
    if conf >= 0.64:
        return "medium"
    if conf >= 0.55:
        return "low"
    return "none"
