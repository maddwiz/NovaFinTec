"""
Day-skimmer control loop.

This loop is purpose-built for `AION_TRADING_MODE=day_skimmer` and uses the
intraday modules (bar engine, session analyzer, pattern stack, confluence,
and intraday risk) as the primary execution path.
"""

from __future__ import annotations

import datetime as dt
import json
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
from ..brain.watchlist import SkimmerWatchlistManager
from ..data.ib_client import disconnect, hist_bars_cached, ib
from ..execution.simulator import ExecutionSimulator
from ..risk.exposure_gate import check_exposure
from ..risk.governor_hierarchy import GovernorAction, resolve_governor_action
from .alerting import send_alert
from .audit_log import audit_log
from .kill_switch import KillSwitchWatcher
from .health_aggregator import write_system_health
from .order_state import save_order_state
from .reconciliation import reconcile_on_startup
from .telemetry_summary import write_telemetry_summary
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
    entry_category_scores: dict[str, float] | None = None


def _write_json_atomic(path: Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(p)


def _extract_open_orders(client) -> list[dict]:
    out: list[dict] = []
    try:
        trades = client.openTrades() or []
    except Exception:
        trades = []
    for t in trades:
        try:
            out.append(
                {
                    "order_id": int(getattr(getattr(t, "order", None), "orderId", 0)),
                    "symbol": str(getattr(getattr(t, "contract", None), "symbol", "")).upper(),
                    "action": str(getattr(getattr(t, "order", None), "action", "")).upper(),
                    "qty": int(getattr(getattr(t, "order", None), "totalQuantity", 0)),
                    "status": str(getattr(getattr(t, "orderStatus", None), "status", "")),
                }
            )
        except Exception:
            continue
    return out


def _ib_req_id(client) -> int:
    try:
        c = getattr(client, "client", None)
        getter = getattr(c, "getReqId", None)
        if callable(getter):
            return int(getter())
    except Exception:
        pass
    return 0


def _persist_ib_order_state(client) -> None:
    save_order_state(
        state_dir=Path(cfg.STATE_DIR),
        next_valid_id=_ib_req_id(client),
        open_orders=_extract_open_orders(client),
    )


def _ib_positions_market_value(client) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        positions = client.positions() or []
    except Exception:
        return out
    for p in positions:
        try:
            sym = str(getattr(getattr(p, "contract", None), "symbol", "")).upper()
            if not sym:
                continue
            if hasattr(p, "marketValue"):
                mv = float(getattr(p, "marketValue"))
            elif hasattr(p, "avgCost"):
                mv = float(getattr(p, "position", 0.0)) * float(getattr(p, "avgCost", 0.0))
            else:
                mv = float(getattr(p, "position", 0.0)) * float(getattr(p, "marketPrice", 0.0))
            out[sym] = float(mv)
        except Exception:
            continue
    return out


def _ib_net_liquidation(client, fallback: float) -> float:
    try:
        rows = client.accountSummary() or []
    except Exception:
        rows = []
    for row in rows:
        try:
            if str(getattr(row, "tag", "")).strip().lower() == "netliquidation":
                v = float(getattr(row, "value", 0.0))
                if math.isfinite(v) and v > 0:
                    return float(v)
        except Exception:
            continue
    return float(max(0.0, float(fallback)))


def _run_exposure_gate(client, symbol: str, side: str, qty: int, price: float, fallback_nlv: float) -> tuple[bool, str, dict]:
    if client is None:
        return True, "ib_unavailable", {
            "current_exposure_pct": 0.0,
            "proposed_exposure_pct": 0.0,
            "limit_pct": float(cfg.MAX_GROSS_EXPOSURE_PCT),
        }
    positions = _ib_positions_market_value(client)
    nlv = _ib_net_liquidation(client, fallback=fallback_nlv)
    sym = str(symbol).upper()
    signed_notional = (1.0 if str(side).upper() == "BUY" else -1.0) * max(0, int(qty)) * max(0.0, float(price))
    existing = float(positions.get(sym, 0.0))
    projected = existing + signed_notional

    if abs(projected) <= abs(existing):
        gross = sum(abs(v) for v in positions.values())
        pct = gross / max(1e-9, nlv if nlv > 0 else 1.0)
        return True, "reduces_existing_exposure", {
            "current_exposure_pct": float(pct),
            "proposed_exposure_pct": float(pct),
            "limit_pct": float(cfg.MAX_GROSS_EXPOSURE_PCT),
        }

    incremental = abs(projected) - abs(existing)
    gate = check_exposure(
        current_positions=positions,
        proposed_symbol=sym,
        proposed_value=float(incremental),
        net_liquidation=float(nlv),
        max_gross_exposure_pct=float(getattr(cfg, "MAX_GROSS_EXPOSURE_PCT", 0.95)),
        max_single_position_pct=float(getattr(cfg, "MAX_SINGLE_POSITION_PCT", 0.20)),
        max_correlated_exposure_pct=float(getattr(cfg, "MAX_CORRELATED_EXPOSURE_PCT", 0.40)),
        correlated_symbols=None,
    )
    return bool(gate.allowed), str(gate.reason), {
        "current_exposure_pct": float(gate.current_exposure_pct),
        "proposed_exposure_pct": float(gate.proposed_exposure_pct),
        "limit_pct": float(gate.limit_pct),
    }


class SkimmerLoop:
    def __init__(self, cfg_mod=cfg):
        self.cfg = cfg_mod
        self.exe = ExecutionSimulator(cfg_mod)
        self.bar_engines: dict[str, BarEngine] = {}
        self.session_analyzers: dict[str, SessionAnalyzer] = {}
        self.open_trades: dict[str, TradeState] = {}
        self.last_prices: dict[str, float] = {}
        self.telemetry = SkimmerTelemetry(cfg_mod)
        self._last_summary_ts = 0.0
        self.watchlist = SkimmerWatchlistManager(cfg_mod)

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
        self.kill_switch = KillSwitchWatcher(state_dir=Path(cfg_mod.STATE_DIR), poll_seconds=5.0)
        self._last_order_state_save_ts = 0.0
        self._order_state_save_interval_sec = 300.0
        self._canary_start_time = dt.datetime.now(dt.timezone.utc)
        self._canary_timeout_checked = False
        self._startup_safety_done = False
        self._overlay_bundle_cache: dict | None = None
        self._last_overlay_check_ts = 0.0
        self._ib_client = None

    def _startup_safety_bootstrap(self, ib_client) -> None:
        if self._startup_safety_done:
            return
        if ib_client is None:
            self._startup_safety_done = True
            return

        if bool(getattr(self.cfg, "AION_BLOCK_LIVE_ORDERS", True)) and (not bool(getattr(self.cfg, "AION_PAPER_MODE", True))):
            def _blocked_place_order(*_args, **_kwargs):
                log_run("PAPER-ONLY GUARD: placeOrder blocked")
                audit_log(
                    {"event": "LIVE_ORDER_BLOCKED", "reason": "AION_BLOCK_LIVE_ORDERS active"},
                    log_dir=Path(self.cfg.STATE_DIR),
                )
                return None

            try:
                setattr(ib_client, "placeOrder", _blocked_place_order)
                log_run("PAPER-ONLY GUARD enabled (skimmer): IB placeOrder patched")
            except Exception:
                pass

        rec = reconcile_on_startup(
            ib_client=ib_client,
            shadow_path=Path(self.cfg.STATE_DIR) / "shadow_trades.json",
            auto_fix=bool(getattr(self.cfg, "RECONCILE_AUTO_FIX", True)),
            max_auto_fix_value=float(getattr(self.cfg, "RECONCILE_MAX_AUTO_FIX_VALUE", 5000.0)),
        )
        _write_json_atomic(
            Path(self.cfg.STATE_DIR) / "reconciliation_result.json",
            {
                "passed": bool(rec.passed),
                "mismatches": list(rec.mismatches),
                "ibkr_positions": dict(rec.ibkr_positions),
                "shadow_positions": dict(rec.shadow_positions),
                "action_taken": str(rec.action_taken),
                "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )
        if rec.mismatches:
            for mm in rec.mismatches:
                audit_log(
                    {"event": "RECONCILIATION_MISMATCH", **mm, "action_taken": str(rec.action_taken)},
                    log_dir=Path(self.cfg.STATE_DIR),
                )
            send_alert(
                f"AION skimmer reconciliation mismatches={len(rec.mismatches)} action={rec.action_taken}",
                level="WARNING",
            )
            log_run(f"skimmer reconciliation mismatches={len(rec.mismatches)} action={rec.action_taken}")
        if (not rec.passed) and str(rec.action_taken) == "manual_review_required":
            raise RuntimeError("reconciliation requires manual review")

        _persist_ib_order_state(ib_client)
        self._last_order_state_save_ts = float(time.monotonic())
        self._startup_safety_done = True

    def _active_symbols(self, overlay_bundle: dict | None = None) -> list[str]:
        try:
            syms = self.watchlist.get_active_symbols(overlay_bundle=overlay_bundle)
            if syms:
                return syms
        except Exception as exc:
            log_run(f"skimmer watchlist refresh failed: {exc}")
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
        audit_log(
            {
                "event": "ORDER_INTENT",
                "symbol": str(symbol).upper(),
                "side": str(side_exec).upper(),
                "qty": int(q),
                "reason": str(reason),
            },
            log_dir=Path(self.cfg.STATE_DIR),
        )
        allowed, gate_reason, gate_diag = _run_exposure_gate(
            self._ib_client,
            symbol=symbol,
            side=side_exec,
            qty=q,
            price=float(price),
            fallback_nlv=max(1.0, float(self._equity())),
        )
        audit_log(
            {
                "event": ("EXPOSURE_GATE_PASS" if allowed else "EXPOSURE_GATE_VETO"),
                "symbol": str(symbol).upper(),
                "side": str(side_exec).upper(),
                "qty": int(q),
                "reason": str(gate_reason),
                **gate_diag,
            },
            log_dir=Path(self.cfg.STATE_DIR),
        )
        if not allowed:
            log_run(f"EXPOSURE GATE VETO: {symbol} {side_exec} qty={q} reason={gate_reason}")
            send_alert(f"Exposure gate vetoed skimmer close {symbol}: {gate_reason}", level="WARNING")
            return 0.0

        audit_log(
            {
                "event": "ORDER_SUBMITTED",
                "symbol": str(symbol).upper(),
                "side": str(side_exec).upper(),
                "qty": int(q),
                "reason": str(reason),
            },
            log_dir=Path(self.cfg.STATE_DIR),
        )
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
            audit_log(
                {
                    "event": "ORDER_REJECTED",
                    "symbol": str(symbol).upper(),
                    "side": str(side_exec).upper(),
                    "qty": int(q),
                    "reason": str(reason),
                },
                log_dir=Path(self.cfg.STATE_DIR),
            )
            return 0.0
        audit_log(
            {
                "event": "ORDER_PARTIAL_FILL" if fq < q else "ORDER_FILLED",
                "symbol": str(symbol).upper(),
                "side": str(side_exec).upper(),
                "qty_requested": int(q),
                "qty_filled": int(fq),
                "avg_fill_price": float(fill.avg_fill),
                "reason": str(reason),
            },
            log_dir=Path(self.cfg.STATE_DIR),
        )

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
                    "slippage_bps": float(fill.est_slippage_bps),
                    "estimated_slippage_bps": float(fill.est_slippage_bps),
                    "entry_category_scores": dict(st.entry_category_scores or {}),
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
        requested_qty = int(max(0, sizing.shares))
        if side_exec == "BUY":
            max_cash_qty = int(max(0.0, float(self.cash)) / max(1e-9, float(px)))
            if max_cash_qty <= 0:
                self.telemetry.log_decision(
                    symbol=symbol,
                    action="SKIP_CAPITAL",
                    confluence_score=float(best_result.score),
                    category_scores=best_result.category_scores,
                    session_phase=str(getattr(getattr(session_state, "phase", None), "value", "unknown")),
                    session_type=str(getattr(getattr(session_state, "session_type", None), "value", "unknown")),
                    patterns_detected=detected_patterns,
                    entry_price=float(px),
                    stop_price=float(sizing.stop_price),
                    shares=int(requested_qty),
                    risk_amount=float(requested_qty * max(1e-9, float(sizing.risk_distance))),
                    reasons=["Insufficient cash for long entry"],
                    extras={"candidate_side": str(best_side), "cash": float(self.cash)},
                )
                return
            requested_qty = min(requested_qty, max_cash_qty)

        if requested_qty <= 0:
            return

        audit_log(
            {
                "event": "ORDER_INTENT",
                "symbol": str(symbol).upper(),
                "side": str(side_exec).upper(),
                "qty": int(requested_qty),
                "reason": "intraday_entry",
                "confidence": float(best_result.score),
            },
            log_dir=Path(self.cfg.STATE_DIR),
        )
        allowed, gate_reason, gate_diag = _run_exposure_gate(
            self._ib_client,
            symbol=symbol,
            side=side_exec,
            qty=int(requested_qty),
            price=float(px),
            fallback_nlv=max(1.0, float(self._equity())),
        )
        audit_log(
            {
                "event": ("EXPOSURE_GATE_PASS" if allowed else "EXPOSURE_GATE_VETO"),
                "symbol": str(symbol).upper(),
                "side": str(side_exec).upper(),
                "qty": int(requested_qty),
                "reason": str(gate_reason),
                **gate_diag,
            },
            log_dir=Path(self.cfg.STATE_DIR),
        )
        if not allowed:
            self.telemetry.log_decision(
                symbol=symbol,
                action="SKIP_EXPOSURE_GATE",
                confluence_score=float(best_result.score),
                category_scores=best_result.category_scores,
                session_phase=str(getattr(getattr(session_state, "phase", None), "value", "unknown")),
                session_type=str(getattr(getattr(session_state, "session_type", None), "value", "unknown")),
                patterns_detected=detected_patterns,
                entry_price=float(px),
                stop_price=float(sizing.stop_price),
                shares=int(requested_qty),
                risk_amount=float(requested_qty * max(1e-9, float(sizing.risk_distance))),
                reasons=[f"Exposure gate veto: {gate_reason}"],
                extras={"candidate_side": str(best_side)},
            )
            log_run(f"EXPOSURE GATE VETO: {symbol} {side_exec} qty={requested_qty} reason={gate_reason}")
            send_alert(f"Exposure gate vetoed skimmer entry {symbol}: {gate_reason}", level="WARNING")
            return

        audit_log(
            {
                "event": "ORDER_SUBMITTED",
                "symbol": str(symbol).upper(),
                "side": str(side_exec).upper(),
                "qty": int(requested_qty),
                "reason": "intraday_entry",
            },
            log_dir=Path(self.cfg.STATE_DIR),
        )
        fill = self.exe.execute(
            side=side_exec,
            qty=int(requested_qty),
            ref_price=float(px),
            atr_pct=float(self._atr_pct_1m(frames)),
            confidence=float(np.clip(best_result.score, 0.0, 1.0)),
            allow_partial=True,
        )
        if fill.filled_qty <= 0:
            audit_log(
                {
                    "event": "ORDER_REJECTED",
                    "symbol": str(symbol).upper(),
                    "side": str(side_exec).upper(),
                    "qty": int(requested_qty),
                    "reason": "intraday_entry",
                },
                log_dir=Path(self.cfg.STATE_DIR),
            )
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
                shares=int(requested_qty),
                risk_amount=float(requested_qty * max(1e-9, float(sizing.risk_distance))),
                reasons=list(best_result.reasons[:8]),
                extras={
                    "candidate_side": str(best_side),
                    "requested_qty": int(requested_qty),
                    "sizing_qty": int(sizing.shares),
                },
            )
            return
        audit_log(
            {
                "event": "ORDER_PARTIAL_FILL" if int(fill.filled_qty) < int(requested_qty) else "ORDER_FILLED",
                "symbol": str(symbol).upper(),
                "side": str(side_exec).upper(),
                "qty_requested": int(requested_qty),
                "qty_filled": int(fill.filled_qty),
                "avg_fill_price": float(fill.avg_fill),
                "reason": "intraday_entry",
            },
            log_dir=Path(self.cfg.STATE_DIR),
        )

        notional = float(fill.avg_fill) * int(fill.filled_qty)
        if side_exec == "BUY":
            if self.cash < notional:
                self.telemetry.log_decision(
                    symbol=symbol,
                    action="SKIP_CAPITAL_POST_FILL",
                    confluence_score=float(best_result.score),
                    category_scores=best_result.category_scores,
                    session_phase=str(getattr(getattr(session_state, "phase", None), "value", "unknown")),
                    session_type=str(getattr(getattr(session_state, "session_type", None), "value", "unknown")),
                    patterns_detected=detected_patterns,
                    entry_price=float(fill.avg_fill),
                    stop_price=float(sizing.stop_price),
                    shares=int(fill.filled_qty),
                    risk_amount=float(int(fill.filled_qty) * max(1e-9, float(sizing.risk_distance))),
                    reasons=["Fill notional exceeds available cash"],
                    extras={
                        "candidate_side": str(best_side),
                        "cash": float(self.cash),
                        "fill_notional": float(notional),
                    },
                )
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
            entry_category_scores=dict(best_result.category_scores or {}),
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
            risk_amount=float(int(fill.filled_qty) * max(1e-9, float(sizing.risk_distance))),
            reasons=list(best_result.reasons[:10]),
            extras={
                "symbol": str(symbol).upper(),
                "side": str(best_side),
                "q_overlay_bias": float(q_bias),
                "q_overlay_confidence": float(q_conf),
                "minutes_to_close": int(minutes_to_close),
                "slippage_bps": float(fill.est_slippage_bps),
                "estimated_slippage_bps": float(fill.est_slippage_bps),
                "requested_qty": int(requested_qty),
                "sizing_qty": int(sizing.shares),
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

    def tick(self) -> bool:
        ib_client = ib()
        self._ib_client = ib_client
        self._startup_safety_bootstrap(ib_client)
        now_mono = float(time.monotonic())
        if (now_mono - float(self._last_order_state_save_ts)) >= float(self._order_state_save_interval_sec):
            _persist_ib_order_state(ib_client)
            self._last_order_state_save_ts = now_mono

        canary_block_new_entries = False
        if not self._canary_timeout_checked:
            canary_age_minutes = (dt.datetime.now(dt.timezone.utc) - self._canary_start_time).total_seconds() / 60.0
            if canary_age_minutes > float(getattr(self.cfg, "CANARY_MAX_LIFETIME_MINUTES", 45)):
                action = str(getattr(self.cfg, "CANARY_TIMEOUT_ACTION", "pass")).strip().lower() or "pass"
                result = "TIMEOUT_PASSED" if action == "pass" else "TIMEOUT_FAILED"
                audit_log(
                    {"event": "CANARY_TIMEOUT", "age_minutes": float(canary_age_minutes), "result": str(result)},
                    log_dir=Path(self.cfg.STATE_DIR),
                )
                log_run(
                    f"skimmer canary timeout age={canary_age_minutes:.0f}m "
                    f"limit={int(getattr(self.cfg, 'CANARY_MAX_LIFETIME_MINUTES', 45))}m result={result}"
                )
                send_alert(
                    f"AION skimmer canary timeout {canary_age_minutes:.0f}m result={result}",
                    level=("WARNING" if result == "TIMEOUT_PASSED" else "CRITICAL"),
                )
                canary_block_new_entries = (result == "TIMEOUT_FAILED")
                self._canary_timeout_checked = True

        if self.kill_switch.check():
            audit_log(
                {"event": "KILL_SWITCH_TRIGGERED", "open_positions": int(len(self.open_trades))},
                log_dir=Path(self.cfg.STATE_DIR),
            )
            log_run("KILL SWITCH (skimmer): flattening all positions and stopping")
            send_alert("AION skimmer KILL SWITCH triggered - flattening all positions", level="CRITICAL")
            self._force_close_all("kill_switch")
            self.kill_switch.acknowledge()
            _persist_ib_order_state(ib_client)
            try:
                write_system_health(state_dir=Path(self.cfg.STATE_DIR), log_dir=Path(self.cfg.LOG_DIR))
            except Exception:
                pass
            return False

        overlay_poll_sec = max(5.0, float(getattr(self.cfg, "EXT_SIGNAL_POLL_SECONDS", 300)))
        if (self._overlay_bundle_cache is None) or ((now_mono - float(self._last_overlay_check_ts)) >= overlay_poll_sec):
            overlay_bundle = load_external_signal_bundle(
                path=Path(self.cfg.EXT_SIGNAL_FILE),
                min_confidence=0.0,
                max_bias=float(getattr(self.cfg, "EXT_SIGNAL_MAX_BIAS", 0.90)),
                max_age_hours=float(getattr(self.cfg, "EXT_SIGNAL_MAX_AGE_HOURS", 12.0)),
            )
            self._overlay_bundle_cache = overlay_bundle if isinstance(overlay_bundle, dict) else {}
            self._last_overlay_check_ts = now_mono
            log_run("Overlay refreshed (skimmer)")
        overlay_bundle = self._overlay_bundle_cache if isinstance(self._overlay_bundle_cache, dict) else {}
        if isinstance(overlay_bundle, dict) and bool(overlay_bundle.get("overlay_rejected", False)):
            reason = str(overlay_bundle.get("overlay_rejection_reason", "invalid_overlay"))
            log_run(f"OVERLAY REJECTED (skimmer): {reason}")
            audit_log(
                {"event": "OVERLAY_REJECTED", "reason": reason, "path": str(self.cfg.EXT_SIGNAL_FILE)},
                log_dir=Path(self.cfg.STATE_DIR),
            )

        runtime_diag = overlay_bundle.get("runtime_diag", {}) if isinstance(overlay_bundle, dict) else {}
        flags = runtime_diag.get("flags", []) if isinstance(runtime_diag, dict) else []
        if not isinstance(flags, list):
            flags = []
        gov_results = []
        lock_reason = str(getattr(self.risk.state, "lock_reason", "")).strip().lower()
        if bool(self.risk.state.session_locked) and ("daily loss" in lock_reason):
            gov_results.append({"name": "daily_loss_limit", "score": 0.0, "threshold": 1.0})
        if isinstance(runtime_diag, dict) and (not bool(runtime_diag.get("quality_gate_ok", True))):
            gov_results.append({"name": "quality_governor", "score": 0.0, "threshold": 1.0})
        flag_to_governor = {
            "fracture_alert": "crisis_sentinel",
            "drift_alert": "shock_mask_guard",
            "exec_risk_hard": "exposure_gate",
        }
        for fl in flags:
            gname = flag_to_governor.get(str(fl).strip().lower())
            if gname:
                gov_results.append({"name": gname, "score": 0.0, "threshold": 1.0})
        gov_action = resolve_governor_action(gov_results) if gov_results else GovernorAction.PASS
        if gov_action >= GovernorAction.FLATTEN:
            triggered = [str(x.get("name", "")) for x in gov_results if x.get("name")]
            audit_log(
                {"event": "GOVERNOR_FLATTEN", "triggered": triggered},
                log_dir=Path(self.cfg.STATE_DIR),
            )
            try:
                (Path(self.cfg.STATE_DIR) / "KILL_SWITCH").write_text("GOVERNOR_FLATTEN", encoding="utf-8")
            except Exception:
                pass
            log_run(f"GOVERNOR FLATTEN (skimmer): triggered={','.join(triggered) if triggered else 'unknown'}")
            send_alert(
                f"AION skimmer governor FLATTEN triggered: {','.join(triggered) if triggered else 'unknown'}",
                level="CRITICAL",
            )
            self._force_close_all("governor_flatten")
            _persist_ib_order_state(ib_client)
            try:
                write_system_health(state_dir=Path(self.cfg.STATE_DIR), log_dir=Path(self.cfg.LOG_DIR))
            except Exception:
                pass
            return False
        if gov_action >= GovernorAction.VETO:
            canary_block_new_entries = True

        symbols = self._active_symbols(overlay_bundle=overlay_bundle)
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
                    if not canary_block_new_entries:
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
        self._maybe_update_telemetry_summary()
        try:
            write_system_health(state_dir=Path(self.cfg.STATE_DIR), log_dir=Path(self.cfg.LOG_DIR))
        except Exception:
            pass
        return True

    def _maybe_update_telemetry_summary(self):
        if not bool(getattr(self.cfg, "TELEMETRY_ENABLED", True)):
            return
        now = time.monotonic()
        if (now - float(self._last_summary_ts)) < 30.0:
            return
        self._last_summary_ts = now
        try:
            decisions_path = Path(self.cfg.STATE_DIR) / str(
                getattr(self.cfg, "TELEMETRY_DECISIONS_FILE", "trade_decisions.jsonl")
            )
            output_path = Path(getattr(self.cfg, "TELEMETRY_SUMMARY_FILE", Path(self.cfg.STATE_DIR) / "telemetry_summary.json"))
            window = int(max(1, getattr(self.cfg, "TELEMETRY_SUMMARY_WINDOW", 20)))
            write_telemetry_summary(
                decisions_path=decisions_path,
                output_path=output_path,
                rolling_window=window,
            )
        except Exception:
            return

    def run_forever(self):
        log_run(
            "Aion day_skimmer loop started "
            f"[bar=1 min loop={self.cfg.LOOP_SECONDS}s entry_th={self.cfg.SKIMMER_ENTRY_THRESHOLD:.2f}]"
        )
        try:
            while True:
                try:
                    keep_running = self.tick()
                    if keep_running is False:
                        break
                except Exception as exc:
                    log_run(f"skimmer loop cycle error: {exc}")
                    if "reconciliation requires manual review" in str(exc).lower():
                        send_alert("AION skimmer startup blocked: reconciliation requires manual review.", level="CRITICAL")
                        break
                time.sleep(max(1, int(self.cfg.LOOP_SECONDS)))
        finally:
            try:
                _persist_ib_order_state(ib())
            except Exception:
                pass
            disconnect()


def run_day_skimmer_loop() -> int:
    SkimmerLoop(cfg).run_forever()
    return 0
