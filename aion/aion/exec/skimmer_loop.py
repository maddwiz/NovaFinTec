"""
Day-skimmer control loop.

This loop is purpose-built for `AION_TRADING_MODE=day_skimmer` and uses the
intraday modules (bar engine, session analyzer, pattern stack, confluence,
and intraday risk) as the primary execution path.
"""

from __future__ import annotations

import datetime as dt
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as cfg
from ..brain.bar_engine import BarEngine
from ..brain.external_signals import load_external_signal_bundle
from ..brain.intraday_confluence import IntradaySignalBundle, score_intraday_entry
from ..brain.intraday_patterns import detect_all_intraday_patterns
from ..brain.intraday_risk import IntradayRiskManager, IntradayRiskParams, compute_position_size
from ..brain.novaspine_bridge import build_trade_event, emit_trade_event
from ..brain.session_analyzer import SessionAnalyzer
from ..data.ib_client import disconnect, hist_bars_cached, ib
from ..execution.simulator import ExecutionSimulator
from .skimmer_telemetry import SkimmerTelemetry
from ..utils.logging_utils import log_equity, log_run, log_trade


@dataclass
class TradeState:
    symbol: str
    side: str  # LONG or SHORT
    entry_price: float
    entry_qty: int
    current_qty: int
    risk_distance: float
    partial_taken: bool
    highest_since_entry: float
    lowest_since_entry: float
    entry_time: dt.datetime
    stop_price: float
    r_target_1: float


class SkimmerLoop:
    def __init__(self, cfg_mod=cfg):
        self.cfg = cfg_mod
        self.exe = ExecutionSimulator(cfg_mod)
        self.bar_engines: dict[str, BarEngine] = {}
        self.session_analyzers: dict[str, SessionAnalyzer] = {}
        self.open_trades: dict[str, TradeState] = {}
        self.last_prices: dict[str, float] = {}
        self.telemetry = SkimmerTelemetry(cfg_mod)

        self.cash = float(cfg_mod.EQUITY_START)
        self.closed_pnl = 0.0

        self.risk = IntradayRiskManager(
            equity=float(cfg_mod.EQUITY_START),
            params=IntradayRiskParams(
                stop_atr_multiple=float(cfg_mod.SKIMMER_STOP_ATR_MULTIPLE),
                risk_per_trade_pct=float(cfg_mod.SKIMMER_RISK_PER_TRADE),
                max_position_pct=float(cfg_mod.SKIMMER_MAX_POSITION_PCT),
                partial_profit_r=float(cfg_mod.SKIMMER_PARTIAL_PROFIT_R),
                partial_profit_fraction=float(cfg_mod.SKIMMER_PARTIAL_PROFIT_FRAC),
                trailing_stop_atr=float(cfg_mod.SKIMMER_TRAILING_STOP_ATR),
                max_trades_per_session=int(cfg_mod.SKIMMER_MAX_TRADES_SESSION),
                max_daily_loss_pct=float(cfg_mod.SKIMMER_MAX_DAILY_LOSS_PCT),
                max_open_positions=int(cfg_mod.SKIMMER_MAX_OPEN),
                no_new_entries_after_min=int(cfg_mod.SKIMMER_NO_ENTRY_BEFORE_CLOSE_MIN),
                force_close_all_at_min=int(cfg_mod.SKIMMER_FORCE_CLOSE_BEFORE_MIN),
            ),
        )

    @staticmethod
    def _watchlist_path() -> Path:
        return Path(cfg.STATE_DIR) / "watchlist.txt"

    def _active_symbols(self) -> list[str]:
        p = self._watchlist_path()
        if p.exists():
            try:
                syms = [str(x).strip().upper() for x in p.read_text(encoding="utf-8").splitlines() if str(x).strip()]
                if syms:
                    return syms
            except Exception:
                pass
        return ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "AMD", "GOOGL"]

    @staticmethod
    def _normalize_bars(df: pd.DataFrame | None) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        x = df.copy()
        col_map = {}
        for c in x.columns:
            lc = str(c).strip().lower()
            if lc in {"date", "datetime", "timestamp", "time"}:
                col_map[c] = "date"
            elif lc in {"open", "high", "low", "close", "volume"}:
                col_map[c] = lc
        if col_map:
            x = x.rename(columns=col_map)

        if "date" in x.columns:
            idx = pd.to_datetime(x["date"], errors="coerce")
            x = x.loc[~idx.isna()].copy()
            x.index = pd.DatetimeIndex(idx[~idx.isna()])
        elif not isinstance(x.index, pd.DatetimeIndex):
            idx = pd.to_datetime(x.index, errors="coerce")
            x = x.loc[~idx.isna()].copy()
            x.index = pd.DatetimeIndex(idx[~idx.isna()])

        x = x.sort_index()
        for c in ["open", "high", "low", "close"]:
            if c not in x.columns:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            x[c] = pd.to_numeric(x[c], errors="coerce")

        if "volume" not in x.columns:
            x["volume"] = 0.0
        x["volume"] = pd.to_numeric(x["volume"], errors="coerce").fillna(0.0)

        x = x.dropna(subset=["open", "high", "low", "close"])
        if x.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return x[["open", "high", "low", "close", "volume"]]

    @staticmethod
    def _minutes_to_close(ts: pd.Timestamp | None) -> int:
        if ts is None:
            return 390
        minute_of_day = int(ts.hour) * 60 + int(ts.minute)
        close_min = 16 * 60
        return max(0, close_min - minute_of_day)

    def _equity(self) -> float:
        eq = float(self.cash)
        for sym, st in self.open_trades.items():
            px = float(self.last_prices.get(sym, st.entry_price))
            if st.side == "LONG":
                eq += px * st.current_qty
            else:
                eq -= px * st.current_qty
        return float(eq)

    def _q_overlay_for_symbol(self, symbol: str, bundle: dict | None) -> tuple[float, float]:
        sigs = bundle.get("signals", {}) if isinstance(bundle, dict) else {}
        if not isinstance(sigs, dict):
            return 0.0, 0.0
        g = sigs.get("__GLOBAL__", {}) if isinstance(sigs.get("__GLOBAL__", {}), dict) else {}
        s = sigs.get(str(symbol).upper(), {}) if isinstance(sigs.get(str(symbol).upper(), {}), dict) else {}
        bias = float(s.get("bias", g.get("bias", 0.0)))
        conf = float(s.get("confidence", g.get("confidence", 0.0)))
        if not math.isfinite(bias):
            bias = 0.0
        if not math.isfinite(conf):
            conf = 0.0
        return float(np.clip(bias, -1.0, 1.0)), float(np.clip(conf, 0.0, 1.0))

    @staticmethod
    def _atr_5m(frames: dict[str, object]) -> float:
        tfd5 = frames.get("5m")
        if tfd5 is None or getattr(tfd5, "atr", None) is None or len(tfd5.atr) == 0:
            return 0.0
        try:
            v = float(tfd5.atr.iloc[-1])
        except Exception:
            return 0.0
        return float(max(0.0, v))

    @staticmethod
    def _atr_pct_1m(frames: dict[str, object]) -> float:
        tfd1 = frames.get("1m")
        if tfd1 is None or getattr(tfd1, "bars", None) is None or tfd1.bars.empty:
            return 0.0
        c = pd.to_numeric(tfd1.bars["close"], errors="coerce").dropna()
        if len(c) < 6:
            return 0.0
        r = c.pct_change().dropna().tail(30)
        if r.empty:
            return 0.0
        return float(max(0.0, np.std(r.values, ddof=1) if len(r) > 1 else 0.0))

    def _emit_trade_event(
        self,
        event_type: str,
        symbol: str,
        side: str,
        qty: int,
        entry: float,
        exit_px: float,
        pnl: float,
        reason: str,
        confidence: float,
        extra: dict | None = None,
    ):
        try:
            payload_extra = {"source": "aion_skimmer"}
            if isinstance(extra, dict) and extra:
                payload_extra.update(extra)
            ev = build_trade_event(
                event_type=event_type,
                symbol=symbol,
                side=side,
                qty=int(max(0, qty)),
                entry=float(entry),
                exit=float(exit_px),
                pnl=float(pnl),
                reason=str(reason),
                confidence=float(confidence),
                regime="intraday",
                extra=payload_extra,
            )
            emit_trade_event(ev, self.cfg)
        except Exception:
            pass

    def _close_qty(self, symbol: str, qty: int, reason: str, price: float, atr_pct: float) -> float:
        st = self.open_trades.get(symbol)
        if st is None:
            return 0.0
        q = int(max(0, min(int(qty), st.current_qty)))
        if q <= 0:
            return 0.0

        side_exec = "SELL" if st.side == "LONG" else "BUY"
        fill = self.exe.execute(
            side=side_exec,
            qty=q,
            ref_price=float(price),
            atr_pct=float(max(0.0, atr_pct)),
            confidence=0.6,
            allow_partial=False,
        )
        fq = int(max(0, min(fill.filled_qty, st.current_qty)))
        if fq <= 0:
            return 0.0

        if st.side == "LONG":
            self.cash += float(fill.avg_fill) * fq
            pnl = (float(fill.avg_fill) - float(st.entry_price)) * fq
        else:
            self.cash -= float(fill.avg_fill) * fq
            pnl = (float(st.entry_price) - float(fill.avg_fill)) * fq

        st.current_qty -= fq
        self.closed_pnl += float(pnl)

        log_trade(
            dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            symbol,
            f"EXIT_{reason.upper()}",
            fq,
            float(fill.avg_fill),
            float(pnl),
            float(self.closed_pnl),
            reason,
            confidence=0.6,
            regime="intraday",
            stop=float(st.stop_price),
            target=float(st.r_target_1),
            trail=float(st.stop_price),
            fill_ratio=float(fill.fill_ratio),
            slippage_bps=float(fill.est_slippage_bps),
        )
        self.telemetry.log_decision(
            symbol=symbol,
            action=f"EXIT_{reason.upper()}",
            confluence_score=0.0,
            session_phase="unknown",
            session_type="unknown",
            patterns_detected=[],
            entry_price=float(st.entry_price),
            stop_price=float(st.stop_price),
            shares=int(fq),
            risk_amount=float(fq * max(1e-9, st.risk_distance)),
            reasons=[str(reason)],
            extras={
                "pnl_realized": float(pnl),
                "remaining_qty": int(st.current_qty),
                "fill_price": float(fill.avg_fill),
            },
        )

        if reason in {"partial_profit_1R", "trailing_stop"}:
            self._emit_trade_event(
                event_type="trade.partial_exit" if reason == "partial_profit_1R" else "trade.exit",
                symbol=symbol,
                side=st.side,
                qty=fq,
                entry=float(st.entry_price),
                exit_px=float(fill.avg_fill),
                pnl=float(pnl),
                reason=reason,
                confidence=0.6,
            )

        if st.current_qty <= 0:
            self.risk.record_trade_result(float(pnl))
            self.open_trades.pop(symbol, None)

        return float(pnl)

    def _manage_position(self, symbol: str, frames: dict[str, object], price: float):
        st = self.open_trades.get(symbol)
        if st is None:
            return

        atr_5m = self._atr_5m(frames)
        atr_pct = self._atr_pct_1m(frames)
        if st.side == "LONG":
            st.highest_since_entry = max(st.highest_since_entry, float(price))
            if price <= st.stop_price:
                self._close_qty(symbol, st.current_qty, "initial_stop", price, atr_pct)
                return
            if (not st.partial_taken) and price >= st.r_target_1:
                close_qty = int(st.current_qty * float(self.risk.params.partial_profit_fraction))
                if close_qty > 0:
                    self._close_qty(symbol, close_qty, "partial_profit_1R", price, atr_pct)
                    st = self.open_trades.get(symbol)
                    if st is None:
                        return
                    st.partial_taken = True
                    st.stop_price = max(st.stop_price, st.entry_price)
            if st.partial_taken and st.current_qty > 0 and atr_5m > 0:
                trail = max(st.entry_price, st.highest_since_entry - atr_5m * float(self.risk.params.trailing_stop_atr))
                if price <= trail:
                    self._close_qty(symbol, st.current_qty, "trailing_stop", price, atr_pct)
        else:
            st.lowest_since_entry = min(st.lowest_since_entry, float(price))
            if price >= st.stop_price:
                self._close_qty(symbol, st.current_qty, "initial_stop", price, atr_pct)
                return
            if (not st.partial_taken) and price <= st.r_target_1:
                close_qty = int(st.current_qty * float(self.risk.params.partial_profit_fraction))
                if close_qty > 0:
                    self._close_qty(symbol, close_qty, "partial_profit_1R", price, atr_pct)
                    st = self.open_trades.get(symbol)
                    if st is None:
                        return
                    st.partial_taken = True
                    st.stop_price = min(st.stop_price, st.entry_price)
            if st.partial_taken and st.current_qty > 0 and atr_5m > 0:
                trail = min(st.entry_price, st.lowest_since_entry + atr_5m * float(self.risk.params.trailing_stop_atr))
                if price >= trail:
                    self._close_qty(symbol, st.current_qty, "trailing_stop", price, atr_pct)

    def _evaluate_entry(self, symbol: str, frames: dict[str, object], session_state, overlay_bundle: dict | None, now_ts: pd.Timestamp | None):
        minutes_to_close = self._minutes_to_close(now_ts)
        can_enter, gate_reason = self.risk.can_enter(minutes_to_close, len(self.open_trades))
        if not can_enter:
            self.telemetry.log_decision(
                symbol=symbol,
                action="SKIP_RISK_GATE",
                confluence_score=0.0,
                session_phase=str(getattr(getattr(session_state, "phase", None), "value", "unknown")),
                session_type=str(getattr(getattr(session_state, "session_type", None), "value", "unknown")),
                patterns_detected=[],
                reasons=[str(gate_reason)],
            )
            return

        tfd5 = frames.get("5m")
        if tfd5 is None or tfd5.bars is None or tfd5.bars.empty:
            return

        atr_5m = self._atr_5m(frames)
        if atr_5m <= 0:
            return
        tfd1 = frames.get("1m")
        if tfd1 is None or tfd1.bars is None or tfd1.bars.empty:
            return
        px = float(tfd1.bars["close"].iloc[-1])
        if not math.isfinite(px) or px <= 0:
            return

        patterns = detect_all_intraday_patterns(tfd5.bars, tfd5.atr)
        detected_patterns = [k for k, v in patterns.items() if isinstance(v, dict) and bool(v.get("detected", False))]
        q_bias, q_conf = self._q_overlay_for_symbol(symbol, overlay_bundle)

        best_side = None
        best_result = None
        for side in ["LONG", "SHORT"]:
            bundle = IntradaySignalBundle(
                symbol=symbol,
                side=side,
                session=session_state,
                patterns=patterns,
                bars=frames,
                q_overlay_bias=q_bias,
                q_overlay_confidence=q_conf,
            )
            result = score_intraday_entry(bundle, self.cfg)
            if (best_result is None) or (result.score > best_result.score):
                best_side = side
                best_result = result

        if best_side is None or best_result is None:
            return
        if not best_result.entry_allowed:
            self.telemetry.log_decision(
                symbol=symbol,
                action="NO_ENTRY",
                confluence_score=float(best_result.score),
                category_scores=best_result.category_scores,
                session_phase=str(getattr(getattr(session_state, "phase", None), "value", "unknown")),
                session_type=str(getattr(getattr(session_state, "session_type", None), "value", "unknown")),
                patterns_detected=detected_patterns,
                entry_price=float(px),
                reasons=list(best_result.reasons[:8]),
                extras={"candidate_side": str(best_side)},
            )
            return

        sizing = compute_position_size(
            side=best_side,
            entry_price=px,
            atr_5m=atr_5m,
            equity=max(1.0, self._equity()),
            params=self.risk.params,
        )

        side_exec = "BUY" if best_side == "LONG" else "SELL"
        fill = self.exe.execute(
            side=side_exec,
            qty=int(sizing.shares),
            ref_price=float(px),
            atr_pct=float(self._atr_pct_1m(frames)),
            confidence=float(np.clip(best_result.score, 0.0, 1.0)),
            allow_partial=True,
        )
        if fill.filled_qty <= 0:
            self.telemetry.log_decision(
                symbol=symbol,
                action="NO_FILL",
                confluence_score=float(best_result.score),
                category_scores=best_result.category_scores,
                session_phase=str(getattr(getattr(session_state, "phase", None), "value", "unknown")),
                session_type=str(getattr(getattr(session_state, "session_type", None), "value", "unknown")),
                patterns_detected=detected_patterns,
                entry_price=float(px),
                stop_price=float(sizing.stop_price),
                shares=int(sizing.shares),
                risk_amount=float(sizing.risk_amount),
                reasons=list(best_result.reasons[:8]),
                extras={"candidate_side": str(best_side)},
            )
            return

        notional = float(fill.avg_fill) * int(fill.filled_qty)
        if side_exec == "BUY":
            if self.cash < notional:
                return
            self.cash -= notional
        else:
            self.cash += notional

        ts_utc = dt.datetime.now(dt.timezone.utc)
        self.open_trades[symbol] = TradeState(
            symbol=symbol,
            side=best_side,
            entry_price=float(fill.avg_fill),
            entry_qty=int(fill.filled_qty),
            current_qty=int(fill.filled_qty),
            risk_distance=float(sizing.risk_distance),
            partial_taken=False,
            highest_since_entry=float(fill.avg_fill),
            lowest_since_entry=float(fill.avg_fill),
            entry_time=ts_utc,
            stop_price=float(sizing.stop_price),
            r_target_1=float(sizing.r_target_1),
        )

        self.telemetry.log_decision(
            symbol=symbol,
            action=f"ENTRY_{best_side}",
            confluence_score=float(best_result.score),
            category_scores=best_result.category_scores,
            session_phase=str(getattr(getattr(session_state, "phase", None), "value", "unknown")),
            session_type=str(getattr(getattr(session_state, "session_type", None), "value", "unknown")),
            patterns_detected=detected_patterns,
            entry_price=float(fill.avg_fill),
            stop_price=float(sizing.stop_price),
            shares=int(fill.filled_qty),
            risk_amount=float(sizing.risk_amount),
            reasons=list(best_result.reasons[:10]),
            extras={
                "symbol": str(symbol).upper(),
                "side": str(best_side),
                "q_overlay_bias": float(q_bias),
                "q_overlay_confidence": float(q_conf),
                "minutes_to_close": int(minutes_to_close),
            },
        )
        self._emit_trade_event(
            event_type="trade.entry",
            symbol=symbol,
            side=best_side,
            qty=int(fill.filled_qty),
            entry=float(fill.avg_fill),
            exit_px=float(fill.avg_fill),
            pnl=0.0,
            reason="intraday_entry",
            confidence=float(best_result.score),
            extra={
                "type": "intraday_entry",
                "confluence_score": float(best_result.score),
                "category_scores": dict(best_result.category_scores),
                "session_phase": str(getattr(getattr(session_state, "phase", None), "value", "unknown")),
                "session_type": str(getattr(getattr(session_state, "session_type", None), "value", "unknown")),
                "patterns": detected_patterns,
            },
        )

        log_trade(
            dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            symbol,
            f"ENTRY_{side_exec}",
            int(fill.filled_qty),
            float(fill.avg_fill),
            0.0,
            float(self.closed_pnl),
            "; ".join(best_result.reasons[:4]) or "skimmer_entry",
            confidence=float(best_result.score),
            regime=str(getattr(session_state, "session_type", "intraday")),
            stop=float(sizing.stop_price),
            target=float(sizing.r_target_2),
            trail=float(sizing.stop_price),
            fill_ratio=float(fill.fill_ratio),
            slippage_bps=float(fill.est_slippage_bps),
        )

    def _force_close_all(self, reason: str):
        for sym in list(self.open_trades.keys()):
            st = self.open_trades.get(sym)
            if st is None:
                continue
            px = float(self.last_prices.get(sym, st.entry_price))
            self._close_qty(sym, st.current_qty, reason, px, 0.0)

    def tick(self):
        ib()

        overlay_bundle = load_external_signal_bundle(
            path=Path(self.cfg.EXT_SIGNAL_FILE),
            min_confidence=0.0,
            max_bias=float(getattr(self.cfg, "EXT_SIGNAL_MAX_BIAS", 0.90)),
            max_age_hours=float(getattr(self.cfg, "EXT_SIGNAL_MAX_AGE_HOURS", 12.0)),
        )

        symbols = self._active_symbols()
        for symbol in symbols:
            try:
                bars_raw = hist_bars_cached(
                    symbol,
                    duration=str(getattr(self.cfg, "HIST_DURATION", "2 D")),
                    barSize="1 min",
                    ttl_seconds=int(max(0, getattr(self.cfg, "MAIN_BARS_CACHE_SEC", 25))),
                )
                bars = self._normalize_bars(bars_raw)
                if bars.empty:
                    continue

                be = self.bar_engines.get(symbol)
                if be is None:
                    be = BarEngine(self.cfg)
                    self.bar_engines[symbol] = be
                frames = be.update(bars)

                sa = self.session_analyzers.get(symbol)
                if sa is None:
                    sa = SessionAnalyzer(self.cfg)
                    self.session_analyzers[symbol] = sa
                session_state = sa.update(bars)

                price = be.latest_price()
                if price is None or (not math.isfinite(price)):
                    continue
                self.last_prices[symbol] = float(price)

                self._manage_position(symbol, frames, float(price))
                if symbol not in self.open_trades:
                    ts = bars.index[-1] if isinstance(bars.index, pd.DatetimeIndex) and len(bars.index) else None
                    self._evaluate_entry(symbol, frames, session_state, overlay_bundle, ts)
            except Exception as exc:
                log_run(f"{symbol}: skimmer tick error: {exc}")
                continue

        # Session-end flattening.
        any_ts = None
        if self.bar_engines:
            for eng in self.bar_engines.values():
                tfd = eng.get("1m")
                if tfd is not None and tfd.bars is not None and not tfd.bars.empty:
                    any_ts = tfd.bars.index[-1]
                    break
        if self.risk.should_force_close_all(self._minutes_to_close(any_ts)):
            self._force_close_all("session_end")

        eq = self._equity()
        open_pnl = float(eq - self.cash - self.closed_pnl)
        log_equity(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), float(eq), float(self.cash), float(open_pnl), float(self.closed_pnl))

    def run_forever(self):
        log_run(
            "Aion day_skimmer loop started "
            f"[bar=1 min loop={self.cfg.LOOP_SECONDS}s entry_th={self.cfg.SKIMMER_ENTRY_THRESHOLD:.2f}]"
        )
        try:
            while True:
                try:
                    self.tick()
                except Exception as exc:
                    log_run(f"skimmer loop cycle error: {exc}")
                time.sleep(max(1, int(self.cfg.LOOP_SECONDS)))
        finally:
            disconnect()


def run_day_skimmer_loop() -> int:
    SkimmerLoop(cfg).run_forever()
    return 0
