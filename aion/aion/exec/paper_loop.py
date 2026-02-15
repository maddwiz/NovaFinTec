import datetime as dt
import json
import math
import time

from .. import config as cfg
from ..brain.signals import (
    build_trade_signal,
    compute_features,
    confidence_tag,
    multi_timeframe_alignment,
    opposite_confidence,
)
from ..brain.external_signals import (
    blend_external_signals,
    load_external_signal_bundle,
    runtime_overlay_scale,
)
from ..brain.novaspine_bridge import build_trade_event, emit_trade_event
from ..data.ib_client import disconnect, hist_bars_cached, ib
from ..execution.simulator import ExecutionSimulator
from ..ml.meta_label import MetaLabelModel
from ..monitoring.runtime_monitor import RuntimeMonitor
from ..portfolio.optimizer import allocate_candidates
from ..risk.event_filter import EventRiskFilter
from ..risk.kill_switch import KillSwitch
from ..risk.position_sizing import gross_leverage_ok, risk_qty
from ..utils.logging_utils import log_alert, log_equity, log_run, log_signal, log_trade

WATCHLIST_TXT = cfg.STATE_DIR / "watchlist.txt"
PROFILE_JSON = cfg.STATE_DIR / "strategy_profile.json"
RUNTIME_STATE_FILE = cfg.RUNTIME_STATE_FILE


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_watchlist():
    if not WATCHLIST_TXT.exists():
        return []
    return [s.strip().upper() for s in WATCHLIST_TXT.read_text().splitlines() if s.strip()]


def load_profile() -> dict:
    if not PROFILE_JSON.exists():
        return {}
    try:
        data = json.loads(PROFILE_JSON.read_text())
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _isfinite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _emit_trade_memory_event(
    event_type: str,
    symbol: str,
    side: str,
    qty: int,
    entry: float,
    exit_: float,
    pnl: float,
    reason: str,
    confidence: float,
    regime: str,
    monitor: RuntimeMonitor | None = None,
    extra: dict | None = None,
):
    if not bool(getattr(cfg, "MEMORY_ENABLE", False)):
        return

    try:
        ev = build_trade_event(
            event_type=str(event_type),
            symbol=str(symbol).upper(),
            side=str(side).upper(),
            qty=max(0, int(qty)),
            entry=float(entry),
            exit=float(exit_),
            pnl=float(pnl),
            reason=str(reason),
            confidence=float(confidence),
            regime=str(regime),
            extra=(extra if isinstance(extra, dict) else None),
        )
        res = emit_trade_event(ev, cfg)
        if not bool(res.get("ok", True)):
            msg = (
                f"NovaSpine emit issue event={event_type} symbol={symbol} "
                f"backend={res.get('backend')} error={res.get('error', 'unknown')}"
            )
            log_run(msg)
            if monitor is not None and cfg.MONITORING_ENABLED:
                monitor.record_system_event("novaspine_emit_fail", msg)
    except Exception as exc:
        msg = f"NovaSpine emit exception event={event_type} symbol={symbol}: {exc}"
        log_run(msg)
        if monitor is not None and cfg.MONITORING_ENABLED:
            monitor.record_system_event("novaspine_emit_exception", str(exc))


def _scale_external_signal(sig: dict | None, scale: float, max_bias: float):
    if not isinstance(sig, dict):
        return None
    s = max(0.0, min(2.0, _safe_float(scale, 1.0)))
    conf = max(0.0, min(1.0, _safe_float(sig.get("confidence"), 0.0) * s))
    bias = _safe_float(sig.get("bias"), 0.0) * math.sqrt(max(0.0, s))
    mb = abs(float(max_bias))
    bias = max(-mb, min(mb, bias))
    return {"bias": float(bias), "confidence": float(conf)}


def _normalize_position(raw: dict):
    if not isinstance(raw, dict):
        return None
    side = str(raw.get("side", "")).upper()
    if side not in {"LONG", "SHORT"}:
        return None
    qty = _safe_int(raw.get("qty"), 0)
    if qty <= 0:
        return None
    entry = _safe_float(raw.get("entry"), 0.0)
    if entry <= 0:
        return None

    out = {
        "qty": qty,
        "side": side,
        "entry": entry,
        "mark_price": _safe_float(raw.get("mark_price"), entry),
        "stop": _safe_float(raw.get("stop"), entry),
        "target": _safe_float(raw.get("target"), entry),
        "trail_stop": _safe_float(raw.get("trail_stop"), entry),
        "trail_mult": max(0.1, _safe_float(raw.get("trail_mult"), cfg.TRAIL_ATR_MULT)),
        "init_risk": max(1e-6, _safe_float(raw.get("init_risk"), entry * 0.0035)),
        "bars_held": max(0, _safe_int(raw.get("bars_held"), 0)),
        "partial_taken": bool(raw.get("partial_taken", False)),
        "peak_price": _safe_float(raw.get("peak_price"), entry),
        "trough_price": _safe_float(raw.get("trough_price"), entry),
        "confidence": _safe_float(raw.get("confidence"), 0.0),
        "regime": str(raw.get("regime", "mixed")),
        "atr_pct": max(0.0, _safe_float(raw.get("atr_pct"), 0.0)),
    }
    return out


def load_runtime_state(today: str):
    if not cfg.RESTORE_RUNTIME_STATE:
        return None
    if not RUNTIME_STATE_FILE.exists():
        return None
    try:
        payload = json.loads(RUNTIME_STATE_FILE.read_text())
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if str(payload.get("day", "")) != today:
        return None

    open_positions = {}
    raw_pos = payload.get("open_positions", {})
    if isinstance(raw_pos, dict):
        for sym, pos in raw_pos.items():
            n = _normalize_position(pos)
            if n is not None:
                open_positions[str(sym).upper()] = n

    cooldown = {}
    raw_cd = payload.get("cooldown", {})
    if isinstance(raw_cd, dict):
        for sym, val in raw_cd.items():
            v = _safe_int(val, 0)
            if v > 0:
                cooldown[str(sym).upper()] = v

    return {
        "cash": _safe_float(payload.get("cash"), cfg.EQUITY_START),
        "closed_pnl": _safe_float(payload.get("closed_pnl"), 0.0),
        "trades_today": max(0, _safe_int(payload.get("trades_today"), 0)),
        "open_positions": open_positions,
        "cooldown": cooldown,
    }


def save_runtime_state(today: str, cash: float, closed_pnl: float, trades_today: int, open_positions: dict, cooldown: dict):
    payload = {
        "saved_at": dt.datetime.now().isoformat(),
        "day": today,
        "cash": float(cash),
        "closed_pnl": float(closed_pnl),
        "trades_today": int(trades_today),
        "open_positions": open_positions,
        "cooldown": cooldown,
    }
    try:
        RUNTIME_STATE_FILE.write_text(json.dumps(payload, indent=2, default=str))
    except Exception:
        pass


def _position_pnl(position: dict, mark_price: float) -> float:
    qty = position["qty"]
    if position["side"] == "LONG":
        return (mark_price - position["entry"]) * qty
    return (position["entry"] - mark_price) * qty


def _mark_open_pnl(open_positions: dict, last_prices: dict) -> float:
    open_pnl = 0.0
    for sym, pos in open_positions.items():
        px = last_prices.get(sym, pos.get("mark_price", pos["entry"]))
        pos["mark_price"] = px
        open_pnl += _position_pnl(pos, px)
    return open_pnl


def _equity_from_cash_and_positions(cash: float, open_positions: dict, last_prices: dict) -> float:
    equity = float(cash)
    for sym, pos in open_positions.items():
        px = float(last_prices.get(sym, pos.get("mark_price", pos["entry"])))
        qty = int(pos.get("qty", 0))
        if qty <= 0:
            continue
        if pos.get("side") == "LONG":
            equity += px * qty
        else:
            equity -= px * qty
    return equity


def _close_position(
    open_positions: dict,
    symbol: str,
    fill_price: float,
    reason: str,
    cash: float,
    closed_pnl: float,
    ks: KillSwitch,
    today: str,
    fill_ratio: float = 1.0,
    slippage_bps: float = 0.0,
    monitor: RuntimeMonitor | None = None,
):
    pos = open_positions[symbol]
    qty = int(pos["qty"])

    if pos["side"] == "LONG":
        pnl = (fill_price - pos["entry"]) * qty
        cash += qty * fill_price
        side = "EXIT_SELL"
    else:
        pnl = (pos["entry"] - fill_price) * qty
        cash -= qty * fill_price
        side = "EXIT_BUY"

    closed_pnl += pnl
    ks.register_trade(pnl, today)

    log_trade(
        now(),
        symbol,
        side,
        qty,
        pos["entry"],
        fill_price,
        pnl,
        reason,
        confidence=float(pos.get("confidence", 0.0)),
        regime=str(pos.get("regime", "")),
        stop=float(pos.get("stop", 0.0)),
        target=float(pos.get("target", 0.0)),
        trail=float(pos.get("trail_stop", 0.0)),
        fill_ratio=fill_ratio,
        slippage_bps=slippage_bps,
    )
    _emit_trade_memory_event(
        event_type="trade.exit",
        symbol=symbol,
        side=str(pos["side"]),
        qty=qty,
        entry=float(pos["entry"]),
        exit_=float(fill_price),
        pnl=float(pnl),
        reason=str(reason),
        confidence=float(pos.get("confidence", 0.0)),
        regime=str(pos.get("regime", "")),
        monitor=monitor,
        extra={
            "fill_ratio": float(fill_ratio),
            "slippage_bps": float(slippage_bps),
            "stop": float(pos.get("stop", 0.0)),
            "target": float(pos.get("target", 0.0)),
            "trail": float(pos.get("trail_stop", 0.0)),
            "bars_held": int(pos.get("bars_held", 0)),
        },
    )
    del open_positions[symbol]
    return cash, closed_pnl


def _partial_close(
    pos: dict,
    symbol: str,
    price: float,
    exe: ExecutionSimulator,
    cash: float,
    closed_pnl: float,
    ks: KillSwitch,
    today: str,
    monitor: RuntimeMonitor | None = None,
):
    qty = int(pos["qty"])
    close_qty = max(1, int(math.floor(qty * cfg.PARTIAL_CLOSE_FRACTION)))
    close_qty = min(close_qty, qty - 1) if qty > 1 else qty
    if close_qty <= 0:
        return cash, closed_pnl, False

    side = "SELL" if pos["side"] == "LONG" else "BUY"
    fill = exe.execute(side=side, qty=close_qty, ref_price=price, atr_pct=float(pos.get("atr_pct", 0.0)), confidence=float(pos.get("confidence", 0.5)), allow_partial=False)

    if pos["side"] == "LONG":
        pnl = (fill.avg_fill - pos["entry"]) * close_qty
        cash += close_qty * fill.avg_fill
        event = "PARTIAL_SELL"
    else:
        pnl = (pos["entry"] - fill.avg_fill) * close_qty
        cash -= close_qty * fill.avg_fill
        event = "PARTIAL_BUY"

    closed_pnl += pnl
    ks.register_trade(pnl, today)

    pos["qty"] -= close_qty
    pos["partial_taken"] = True

    log_trade(
        now(),
        symbol,
        event,
        close_qty,
        pos["entry"],
        fill.avg_fill,
        pnl,
        "Partial take-profit",
        confidence=float(pos.get("confidence", 0.0)),
        regime=str(pos.get("regime", "")),
        stop=float(pos.get("stop", 0.0)),
        target=float(pos.get("target", 0.0)),
        trail=float(pos.get("trail_stop", 0.0)),
        fill_ratio=fill.fill_ratio,
        slippage_bps=fill.est_slippage_bps,
    )
    _emit_trade_memory_event(
        event_type="trade.partial_exit",
        symbol=symbol,
        side=str(pos["side"]),
        qty=int(close_qty),
        entry=float(pos["entry"]),
        exit_=float(fill.avg_fill),
        pnl=float(pnl),
        reason="Partial take-profit",
        confidence=float(pos.get("confidence", 0.0)),
        regime=str(pos.get("regime", "")),
        monitor=monitor,
        extra={
            "fill_ratio": float(fill.fill_ratio),
            "slippage_bps": float(fill.est_slippage_bps),
            "stop": float(pos.get("stop", 0.0)),
            "target": float(pos.get("target", 0.0)),
            "trail": float(pos.get("trail_stop", 0.0)),
            "bars_held": int(pos.get("bars_held", 0)),
            "remaining_qty": int(pos.get("qty", 0)),
        },
    )

    fully_closed = pos["qty"] <= 0
    return cash, closed_pnl, fully_closed


def main() -> int:
    equity = cfg.EQUITY_START
    cash = cfg.EQUITY_START
    open_positions = {}
    cooldown = {}
    closed_pnl = 0.0
    trades_today = 0
    day_key = dt.datetime.now().strftime("%Y-%m-%d")

    restored = load_runtime_state(day_key)
    if restored:
        cash = float(restored["cash"])
        closed_pnl = float(restored["closed_pnl"])
        trades_today = int(restored["trades_today"])
        open_positions = dict(restored["open_positions"])
        cooldown = dict(restored["cooldown"])
        log_run(
            f"Runtime state restored: positions={len(open_positions)} trades_today={trades_today} cash={cash:.2f}"
        )

    ks = KillSwitch(
        cfg.LOG_DIR / "killswitch.json",
        daily_limit=cfg.DAILY_DRAWDOWN_LIMIT,
        total_limit=cfg.TOTAL_DRAWDOWN_LIMIT,
        max_consecutive_losses=cfg.MAX_CONSECUTIVE_LOSSES,
    )
    ks.load()
    if cfg.AUTO_RESET_KILLSWITCH_ON_START and (
        ks.state.get("tripped_day") or ks.state.get("tripped_total")
    ):
        today0 = dt.datetime.now().strftime("%Y-%m-%d")
        ks.hard_reset(today=today0, equity=equity)
        log_run("KillSwitch state auto-reset on startup (paper mode).")

    exe = ExecutionSimulator(cfg)
    event_filter = EventRiskFilter(cfg)
    meta_model = MetaLabelModel(cfg)
    monitor = RuntimeMonitor(cfg)
    ib_fail_streak = 0
    last_ext_runtime_sig = None

    if cfg.META_LABEL_ENABLED:
        samples = meta_model.fit_from_trades(cfg.LOG_DIR / "shadow_trades.csv")
        if samples > 0:
            log_run(f"Meta-label model trained on {samples} samples.")

    log_run("Aion brain paper loop started (god-mode).")

    try:
        while True:
            try:
                ib()
                if ib_fail_streak > 0:
                    log_run(f"IB reconnected after {ib_fail_streak} failed cycle(s).")
                    if cfg.MONITORING_ENABLED:
                        monitor.record_system_event("ib_reconnected", f"recovered_after={ib_fail_streak}")
                ib_fail_streak = 0
            except Exception as exc:
                ib_fail_streak += 1
                if ib_fail_streak == 1 or (ib_fail_streak % max(1, cfg.IB_RECONNECT_LOG_EVERY) == 0):
                    log_run(
                        f"IB unavailable ({cfg.IB_HOST}:{cfg.IB_PORT}) cycle={ib_fail_streak}: {exc}"
                    )
                if cfg.MONITORING_ENABLED:
                    monitor.record_system_event("ib_connect_fail", str(exc))
                    for msg in monitor.check_alerts():
                        log_alert(msg)
                        log_run(f"ALERT: {msg}")
                save_runtime_state(day_key, cash, closed_pnl, trades_today, open_positions, cooldown)
                time.sleep(cfg.LOOP_SECONDS)
                continue

            profile = load_profile()
            if profile.get("trading_enabled") is False:
                log_run("Adaptive profile disabled trading; sleeping.")
                save_runtime_state(day_key, cash, closed_pnl, trades_today, open_positions, cooldown)
                time.sleep(cfg.LOOP_SECONDS)
                continue

            max_trades_cap = int(profile.get("max_trades_per_day", cfg.MAX_TRADES_PER_DAY))
            max_trades_cap_runtime = max_trades_cap

            wl = load_watchlist()
            if not wl:
                log_run("No watchlist found. Run universe_scan first. Sleeping.")
                save_runtime_state(day_key, cash, closed_pnl, trades_today, open_positions, cooldown)
                time.sleep(cfg.LOOP_SECONDS)
                continue

            today = dt.datetime.now().strftime("%Y-%m-%d")
            if today != day_key:
                day_key = today
                trades_today = 0
                if cfg.META_LABEL_ENABLED:
                    samples = meta_model.fit_from_trades(cfg.LOG_DIR / "shadow_trades.csv")
                    if samples > 0:
                        log_run(f"Meta-label model refreshed on day change ({samples} samples).")

            killswitch_block_new_entries = not ks.check(today, equity)
            if killswitch_block_new_entries:
                log_run(f"KillSwitch tripped: {ks.last_reason}")

            for sym in list(cooldown.keys()):
                cooldown[sym] = max(0, cooldown[sym] - 1)
                if cooldown[sym] == 0:
                    cooldown.pop(sym, None)

            last_prices = {}
            cycle_conf = []
            entry_candidates = []
            external_signals = {}
            global_external = None
            ext_runtime_scale = 1.0
            ext_runtime_diag = {"active": False, "flags": [], "degraded": False, "quality_gate_ok": True}
            if cfg.EXT_SIGNAL_ENABLED:
                ext_bundle = load_external_signal_bundle(
                    path=cfg.EXT_SIGNAL_FILE,
                    min_confidence=cfg.EXT_SIGNAL_MIN_CONFIDENCE,
                    max_bias=cfg.EXT_SIGNAL_MAX_BIAS,
                )
                external_signals = ext_bundle.get("signals", {}) if isinstance(ext_bundle, dict) else {}
                global_external = external_signals.get("__GLOBAL__")
                ext_runtime_scale, ext_runtime_diag = runtime_overlay_scale(
                    ext_bundle,
                    min_scale=cfg.EXT_SIGNAL_RUNTIME_MIN_SCALE,
                    max_scale=cfg.EXT_SIGNAL_RUNTIME_MAX_SCALE,
                    degraded_scale=cfg.EXT_SIGNAL_RUNTIME_DEGRADED_SCALE,
                    quality_fail_scale=cfg.EXT_SIGNAL_RUNTIME_QFAIL_SCALE,
                    flag_scale=cfg.EXT_SIGNAL_RUNTIME_FLAG_SCALE,
                )
                cap_scale = max(0.50, min(1.00, float(ext_runtime_scale)))
                max_trades_cap_runtime = max(1, int(round(max_trades_cap * cap_scale)))
                sig = (
                    round(float(ext_runtime_scale), 4),
                    tuple(sorted(str(x) for x in ext_runtime_diag.get("flags", []) if str(x))),
                    bool(ext_runtime_diag.get("degraded", False)),
                    bool(ext_runtime_diag.get("quality_gate_ok", True)),
                    str(ext_runtime_diag.get("regime", "unknown")),
                    str(ext_runtime_diag.get("source_mode", "unknown")),
                    int(max_trades_cap_runtime),
                )
                if sig != last_ext_runtime_sig:
                    flag_txt = ",".join(sig[1]) if sig[1] else "none"
                    log_run(
                        "External runtime overlay "
                        f"scale={sig[0]:.3f} flags={flag_txt} "
                        f"degraded={sig[2]} quality_ok={sig[3]} regime={sig[4]} source={sig[5]} "
                        f"max_trades={sig[6]}/{max_trades_cap}"
                    )
                    if cfg.MONITORING_ENABLED and (sig[2] or (not sig[3]) or bool(sig[1])):
                        monitor.record_system_event(
                            "external_overlay_runtime",
                            f"scale={sig[0]:.3f} flags={flag_txt} source={sig[5]} max_trades={sig[6]}/{max_trades_cap}",
                        )
                    last_ext_runtime_sig = sig

            for sym in wl:
                try:
                    df = hist_bars_cached(
                        sym,
                        duration=cfg.HIST_DURATION,
                        barSize=cfg.HIST_BAR_SIZE,
                        ttl_seconds=cfg.MAIN_BARS_CACHE_SEC,
                    )
                    if df.empty or len(df) < cfg.MIN_BARS:
                        continue

                    feats = compute_features(df, cfg)
                    last = feats.iloc[-1]

                    price = float(last["close"])
                    atr_val = float(last["atr"])
                    atr_pct = float(last["atr_pct"])
                    adx_val = float(last["adx"])
                    if not (_isfinite(price) and _isfinite(atr_val) and atr_val > 0):
                        continue

                    last_prices[sym] = price
                    lookback = min(cfg.SWING_LOOKBACK, len(df))
                    high = float(df["high"].iloc[-lookback:].max())
                    low = float(df["low"].iloc[-lookback:].min())

                    ext_sig = blend_external_signals(
                        external_signals.get(sym),
                        global_external,
                        max_bias=cfg.EXT_SIGNAL_MAX_BIAS,
                    )
                    ext_sig = _scale_external_signal(ext_sig, ext_runtime_scale, cfg.EXT_SIGNAL_MAX_BIAS)

                    signal = build_trade_signal(
                        last,
                        price,
                        high,
                        low,
                        cfg,
                        profile=profile,
                        external=ext_sig,
                    )
                    base_conf = float(signal.get("confidence", 0.0))
                    mtf_score = 1.0
                    mtf_reasons = []
                    meta_prob = 1.0

                    if signal["side"] and cfg.MTF_CONFIRM_ENABLED:
                        df_1h = hist_bars_cached(
                            sym,
                            duration=cfg.MTF_1H_DURATION,
                            barSize=cfg.MTF_1H_BAR,
                            ttl_seconds=cfg.MTF_1H_CACHE_SEC,
                        )
                        df_4h = hist_bars_cached(
                            sym,
                            duration=cfg.MTF_4H_DURATION,
                            barSize=cfg.MTF_4H_BAR,
                            ttl_seconds=cfg.MTF_4H_CACHE_SEC,
                        )
                        mtf_score, mtf_reasons = multi_timeframe_alignment(df_1h, df_4h, signal["side"], cfg)
                        if mtf_score < cfg.MTF_MIN_ALIGNMENT_SCORE:
                            signal["reasons"].append(f"MTF blocked ({mtf_score:.2f})")
                            signal["side"] = None
                        else:
                            signal["confidence"] = min(1.0, base_conf * (0.70 + 0.55 * mtf_score))
                            signal["reasons"].extend(mtf_reasons)

                    if signal["side"] and cfg.META_LABEL_ENABLED:
                        meta_prob = meta_model.predict_proba(
                            confidence=float(signal["confidence"]),
                            long_conf=float(signal["long_conf"]),
                            short_conf=float(signal["short_conf"]),
                            adx=adx_val,
                            atr_pct=atr_pct,
                            regime=str(signal["regime"]),
                        )
                        if meta_prob < cfg.META_LABEL_MIN_PROB:
                            signal["reasons"].append(f"Meta-label veto ({meta_prob:.2f})")
                            signal["side"] = None
                        else:
                            signal["confidence"] = min(1.0, float(signal["confidence"]) * (0.60 + 0.70 * meta_prob))

                    cycle_conf.append(max(float(signal["long_conf"]), float(signal["short_conf"])))
                    log_signal(
                        now(),
                        sym,
                        signal["regime"],
                        float(signal["long_conf"]),
                        float(signal["short_conf"]),
                        signal["side"] or "HOLD",
                        signal["reasons"],
                        meta_prob=meta_prob,
                        mtf_score=mtf_score,
                        pattern_hits=int(signal.get("pattern_hits", 0)),
                        indicator_hits=int(signal.get("indicator_hits", 0)),
                    )

                    pos = open_positions.get(sym)

                    # Manage open positions
                    if pos:
                        pos["bars_held"] += 1
                        pos["mark_price"] = price

                        if pos["side"] == "LONG":
                            pos["peak_price"] = max(pos["peak_price"], price)
                            trail_candidate = pos["peak_price"] - (atr_val * pos["trail_mult"])
                            pos["trail_stop"] = max(pos["trail_stop"], trail_candidate)

                            if (not pos["partial_taken"]) and price >= (pos["entry"] + pos["init_risk"] * cfg.PARTIAL_TAKE_R):
                                cash, closed_pnl, fully_closed = _partial_close(
                                    pos, sym, price, exe, cash, closed_pnl, ks, today, monitor=monitor
                                )
                                monitor.record_execution(cfg.SLIPPAGE_BPS)
                                if fully_closed:
                                    del open_positions[sym]
                                    cooldown[sym] = cfg.REENTRY_COOLDOWN_CYCLES
                                    continue
                                pos["stop"] = max(pos["stop"], pos["entry"]) if pos["qty"] > 0 else pos["stop"]

                            stop_trigger = price <= max(pos["stop"], pos["trail_stop"])
                            target_trigger = price >= pos["target"]
                        else:
                            pos["trough_price"] = min(pos["trough_price"], price)
                            trail_candidate = pos["trough_price"] + (atr_val * pos["trail_mult"])
                            pos["trail_stop"] = min(pos["trail_stop"], trail_candidate)

                            if (not pos["partial_taken"]) and price <= (pos["entry"] - pos["init_risk"] * cfg.PARTIAL_TAKE_R):
                                cash, closed_pnl, fully_closed = _partial_close(
                                    pos, sym, price, exe, cash, closed_pnl, ks, today, monitor=monitor
                                )
                                monitor.record_execution(cfg.SLIPPAGE_BPS)
                                if fully_closed:
                                    del open_positions[sym]
                                    cooldown[sym] = cfg.REENTRY_COOLDOWN_CYCLES
                                    continue
                                pos["stop"] = min(pos["stop"], pos["entry"]) if pos["qty"] > 0 else pos["stop"]

                            stop_trigger = price >= min(pos["stop"], pos["trail_stop"])
                            target_trigger = price <= pos["target"]

                        opposite = opposite_confidence(pos["side"], signal) >= float(signal["opposite_exit_threshold"])
                        timeout = pos["bars_held"] >= cfg.MAX_HOLD_CYCLES

                        if stop_trigger or target_trigger or opposite or timeout:
                            side = "SELL" if pos["side"] == "LONG" else "BUY"
                            fill = exe.execute(
                                side=side,
                                qty=int(pos["qty"]),
                                ref_price=price,
                                atr_pct=atr_pct,
                                confidence=float(pos.get("confidence", 0.5)),
                                allow_partial=False,
                            )
                            monitor.record_execution(fill.est_slippage_bps)

                            reason = "Stop/Trail hit" if stop_trigger else "Target hit" if target_trigger else "Opposite high-confidence signal" if opposite else "Time stop"
                            cash, closed_pnl = _close_position(
                                open_positions,
                                sym,
                                fill.avg_fill,
                                reason,
                                cash,
                                closed_pnl,
                                ks,
                                today,
                                fill_ratio=fill.fill_ratio,
                                slippage_bps=fill.est_slippage_bps,
                                monitor=monitor,
                            )
                            cooldown[sym] = cfg.REENTRY_COOLDOWN_CYCLES
                            continue

                        continue

                    # Candidate collection for portfolio allocator
                    if killswitch_block_new_entries:
                        continue
                    if sym in cooldown:
                        continue
                    if trades_today >= max_trades_cap_runtime:
                        continue
                    if signal["side"] is None:
                        continue
                    if len(open_positions) + len(entry_candidates) >= cfg.MAX_OPEN_POSITIONS:
                        continue

                    blocked, reason = event_filter.blocked(sym)
                    if blocked:
                        log_run(f"{sym}: event filter blocked entry ({reason})")
                        continue

                    rets = df["close"].pct_change().dropna().tail(80)
                    volatility = float(rets.std()) if not rets.empty else 0.01
                    margin = abs(float(signal["long_conf"]) - float(signal["short_conf"]))
                    expected_edge = float(signal["confidence"]) * (0.6 + margin)

                    entry_candidates.append(
                        {
                            "symbol": sym,
                            "side": signal["side"],
                            "signal": signal,
                            "price": price,
                            "atr": atr_val,
                            "atr_pct": atr_pct,
                            "returns": rets,
                            "volatility": max(1e-6, volatility),
                            "confidence": float(signal["confidence"]),
                            "expected_edge": expected_edge,
                        }
                    )

                except Exception as exc:
                    log_run(f"{sym}: loop error: {exc}")
                    continue

            # Portfolio-level entry selection and sizing
            allocations = {}
            if entry_candidates:
                if cfg.PORTFOLIO_ENABLE:
                    allocations = allocate_candidates(entry_candidates, equity, cfg)
                else:
                    gross = equity * cfg.MAX_GROSS_LEVERAGE
                    per = gross / max(len(entry_candidates), 1)
                    allocations = {
                        c["symbol"]: {"target_notional": per, "weight": 1.0 / len(entry_candidates), "side": c["side"]}
                        for c in entry_candidates
                    }

            for c in sorted(entry_candidates, key=lambda x: x["confidence"], reverse=True):
                if trades_today >= max_trades_cap_runtime:
                    break
                if len(open_positions) >= cfg.MAX_OPEN_POSITIONS:
                    break

                sym = c["symbol"]
                if sym in open_positions:
                    continue

                alloc = allocations.get(sym, {})
                target_notional = float(alloc.get("target_notional", equity * cfg.MAX_POSITION_NOTIONAL_PCT))
                qty_port = int(target_notional / max(c["price"], 1e-6))

                qty_risk = risk_qty(
                    equity=equity,
                    risk_per_trade=cfg.RISK_PER_TRADE,
                    atr=c["atr"],
                    price=c["price"],
                    confidence=c["confidence"],
                    stop_atr_mult=float(c["signal"]["stop_atr_mult"]),
                    max_notional_pct=cfg.MAX_POSITION_NOTIONAL_PCT,
                )
                qty = min(qty_port, qty_risk)
                if qty <= 0:
                    continue

                entry_side = "BUY" if c["side"] == "LONG" else "SELL"
                fill = exe.execute(
                    side=entry_side,
                    qty=qty,
                    ref_price=c["price"],
                    atr_pct=c["atr_pct"],
                    confidence=c["confidence"],
                    allow_partial=True,
                )
                monitor.record_execution(fill.est_slippage_bps)
                if fill.filled_qty <= 0:
                    continue

                notional = fill.filled_qty * fill.avg_fill
                if not gross_leverage_ok(
                    cash=cash,
                    open_positions=open_positions,
                    next_notional=notional,
                    max_gross_leverage=cfg.MAX_GROSS_LEVERAGE,
                    equity=max(equity, 1.0),
                ):
                    continue

                if entry_side == "BUY":
                    if cash < notional:
                        continue
                    cash -= notional
                else:
                    cash += notional

                init_risk = max(c["atr"] * float(c["signal"]["stop_atr_mult"]), fill.avg_fill * 0.0035)
                if c["side"] == "LONG":
                    stop = fill.avg_fill - init_risk
                    target = fill.avg_fill + (c["atr"] * float(c["signal"]["target_atr_mult"]))
                    trail_stop = stop
                    peak_price = fill.avg_fill
                    trough_price = fill.avg_fill
                else:
                    stop = fill.avg_fill + init_risk
                    target = fill.avg_fill - (c["atr"] * float(c["signal"]["target_atr_mult"]))
                    trail_stop = stop
                    peak_price = fill.avg_fill
                    trough_price = fill.avg_fill

                open_positions[sym] = {
                    "qty": int(fill.filled_qty),
                    "side": c["side"],
                    "entry": float(fill.avg_fill),
                    "mark_price": float(fill.avg_fill),
                    "stop": float(stop),
                    "target": float(target),
                    "trail_stop": float(trail_stop),
                    "trail_mult": float(c["signal"]["trail_atr_mult"]),
                    "init_risk": float(init_risk),
                    "bars_held": 0,
                    "partial_taken": False,
                    "peak_price": float(peak_price),
                    "trough_price": float(trough_price),
                    "confidence": float(c["confidence"]),
                    "regime": str(c["signal"]["regime"]),
                    "atr_pct": float(c["atr_pct"]),
                }
                trades_today += 1

                log_trade(
                    now(),
                    sym,
                    f"ENTRY_{entry_side}",
                    int(fill.filled_qty),
                    float(fill.avg_fill),
                    0.0,
                    0.0,
                    f"{confidence_tag(float(c['confidence']))} conf | {'; '.join(c['signal']['reasons'][:3])}",
                    confidence=float(c["confidence"]),
                    regime=str(c["signal"]["regime"]),
                    stop=float(stop),
                    target=float(target),
                    trail=float(trail_stop),
                    fill_ratio=float(fill.fill_ratio),
                    slippage_bps=float(fill.est_slippage_bps),
                )
                _emit_trade_memory_event(
                    event_type="trade.entry",
                    symbol=sym,
                    side=str(c["side"]),
                    qty=int(fill.filled_qty),
                    entry=float(fill.avg_fill),
                    exit_=0.0,
                    pnl=0.0,
                    reason=f"{confidence_tag(float(c['confidence']))} conf entry",
                    confidence=float(c["confidence"]),
                    regime=str(c["signal"]["regime"]),
                    monitor=monitor,
                    extra={
                        "stop": float(stop),
                        "target": float(target),
                        "trail": float(trail_stop),
                        "fill_ratio": float(fill.fill_ratio),
                        "slippage_bps": float(fill.est_slippage_bps),
                        "atr_pct": float(c["atr_pct"]),
                    },
                )

            open_pnl = _mark_open_pnl(open_positions, last_prices)
            equity = _equity_from_cash_and_positions(cash, open_positions, last_prices)
            log_equity(now(), equity, cash, open_pnl, closed_pnl)
            save_runtime_state(day_key, cash, closed_pnl, trades_today, open_positions, cooldown)

            if cfg.MONITORING_ENABLED:
                avg_conf = float(sum(cycle_conf) / len(cycle_conf)) if cycle_conf else 0.0
                monitor.record_cycle(equity=equity, avg_conf=avg_conf)
                alerts = monitor.check_alerts()
                for msg in alerts:
                    log_alert(msg)
                    log_run(f"ALERT: {msg}")

            time.sleep(cfg.LOOP_SECONDS)

    finally:
        disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
