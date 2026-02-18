import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from aion.exec import skimmer_loop as skl


class _WatchlistStub:
    def __init__(self, _cfg):
        pass

    def get_active_symbols(self, overlay_bundle=None):
        return ["AAPL"]


class _ExecStub:
    def __init__(self, _cfg):
        pass

    def execute(self, *, side, qty, ref_price, atr_pct, confidence, allow_partial):
        return SimpleNamespace(
            filled_qty=int(qty),
            avg_fill=float(ref_price),
            fill_ratio=1.0,
            est_slippage_bps=1.5,
        )


class _ExecNoFillStub:
    def __init__(self, _cfg):
        pass

    def execute(self, *, side, qty, ref_price, atr_pct, confidence, allow_partial):
        return SimpleNamespace(
            filled_qty=0,
            avg_fill=float(ref_price),
            fill_ratio=0.0,
            est_slippage_bps=1.5,
        )


def _bars_1m(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2026-01-06 09:30:00", periods=n, freq="min")
    base = 100.0 + np.linspace(0.0, 1.2, n) + 0.08 * np.sin(np.arange(n) / 6.0)
    close = base
    open_ = close - 0.03
    high = np.maximum(open_, close) + 0.05
    low = np.minimum(open_, close) - 0.05
    volume = 1_000_000 + (np.arange(n) % 10) * 2_000
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


def _bars_1m_near_close(n: int = 20) -> pd.DataFrame:
    start = pd.Timestamp("2026-01-06 15:36:00")
    idx = pd.date_range(start, periods=n, freq="min")
    base = 101.0 + np.linspace(0.0, 0.2, n)
    close = base
    open_ = close - 0.02
    high = np.maximum(open_, close) + 0.03
    low = np.minimum(open_, close) - 0.03
    volume = 900_000 + (np.arange(n) % 7) * 1_000
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


def _frames_for_price(price: float, atr_5m: float = 1.0):
    idx_1m = pd.date_range("2026-01-06 11:00:00", periods=3, freq="min")
    bars_1m = pd.DataFrame({"close": [price - 0.1, price, price]}, index=idx_1m)

    idx_5m = pd.date_range("2026-01-06 10:50:00", periods=3, freq="5min")
    bars_5m = pd.DataFrame({"close": [price - 0.3, price - 0.1, price]}, index=idx_5m)
    atr = pd.Series([atr_5m, atr_5m, atr_5m], index=idx_5m)

    return {
        "1m": SimpleNamespace(bars=bars_1m, atr=None),
        "5m": SimpleNamespace(bars=bars_5m, atr=atr),
    }


def _configure_cfg(monkeypatch, tmp_path):
    monkeypatch.setattr(skl.cfg, "EQUITY_START", 100_000.0, raising=False)
    monkeypatch.setattr(skl.cfg, "SKIMMER_STOP_ATR_MULTIPLE", 1.5, raising=False)
    monkeypatch.setattr(skl.cfg, "SKIMMER_RISK_PER_TRADE", 0.005, raising=False)
    monkeypatch.setattr(skl.cfg, "SKIMMER_MAX_POSITION_PCT", 0.03, raising=False)
    monkeypatch.setattr(skl.cfg, "SKIMMER_PARTIAL_PROFIT_R", 1.0, raising=False)
    monkeypatch.setattr(skl.cfg, "SKIMMER_PARTIAL_PROFIT_FRAC", 0.50, raising=False)
    monkeypatch.setattr(skl.cfg, "SKIMMER_TRAILING_STOP_ATR", 1.2, raising=False)
    monkeypatch.setattr(skl.cfg, "SKIMMER_MAX_TRADES_SESSION", 8, raising=False)
    monkeypatch.setattr(skl.cfg, "SKIMMER_MAX_DAILY_LOSS_PCT", 0.015, raising=False)
    monkeypatch.setattr(skl.cfg, "SKIMMER_MAX_OPEN", 3, raising=False)
    monkeypatch.setattr(skl.cfg, "SKIMMER_NO_ENTRY_BEFORE_CLOSE_MIN", 45, raising=False)
    monkeypatch.setattr(skl.cfg, "SKIMMER_FORCE_CLOSE_BEFORE_MIN", 10, raising=False)
    monkeypatch.setattr(skl.cfg, "EXT_SIGNAL_FILE", str(tmp_path / "q_signal_overlay.json"), raising=False)
    monkeypatch.setattr(skl.cfg, "EXT_SIGNAL_MAX_BIAS", 0.9, raising=False)
    monkeypatch.setattr(skl.cfg, "EXT_SIGNAL_MAX_AGE_HOURS", 12.0, raising=False)
    monkeypatch.setattr(skl.cfg, "HIST_DURATION", "2 D", raising=False)
    monkeypatch.setattr(skl.cfg, "MAIN_BARS_CACHE_SEC", 1, raising=False)
    monkeypatch.setattr(skl.cfg, "STATE_DIR", tmp_path, raising=False)
    monkeypatch.setattr(skl.cfg, "TELEMETRY_ENABLED", True, raising=False)
    monkeypatch.setattr(skl.cfg, "TELEMETRY_DECISIONS_FILE", "trade_decisions.jsonl", raising=False)
    monkeypatch.setattr(skl.cfg, "TELEMETRY_SUMMARY_FILE", tmp_path / "telemetry_summary.json", raising=False)
    monkeypatch.setattr(skl.cfg, "TELEMETRY_SUMMARY_WINDOW", 20, raising=False)
    monkeypatch.setattr(skl.cfg, "LOOP_SECONDS", 1, raising=False)


def test_tick_runs_intraday_modules_and_enters(monkeypatch, tmp_path):
    _configure_cfg(monkeypatch, tmp_path)
    bars = _bars_1m(140)
    calls = {"detect": 0, "score": []}

    monkeypatch.setattr(skl, "SkimmerWatchlistManager", _WatchlistStub)
    monkeypatch.setattr(skl, "ExecutionSimulator", _ExecStub)
    monkeypatch.setattr(skl, "ib", lambda: None)
    monkeypatch.setattr(skl, "hist_bars_cached", lambda *args, **kwargs: bars.copy())
    monkeypatch.setattr(skl, "load_external_signal_bundle", lambda **kwargs: {"signals": {"AAPL": {"bias": 0.4, "confidence": 0.8}}})
    monkeypatch.setattr(skl, "emit_trade_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(skl, "build_trade_event", lambda **kwargs: kwargs)
    monkeypatch.setattr(skl, "write_telemetry_summary", lambda **kwargs: None)
    monkeypatch.setattr(skl, "log_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(skl, "log_trade", lambda *args, **kwargs: None)
    monkeypatch.setattr(skl, "log_equity", lambda *args, **kwargs: None)

    def _fake_detect(bars_5m, atr):
        calls["detect"] += 1
        return {
            "pin_bar": {"detected": True, "direction": 1, "strength": 0.9, "level": "vwap"},
        }

    def _fake_score(bundle, _cfg):
        calls["score"].append(bundle.side)
        if bundle.side == "LONG":
            return SimpleNamespace(
                score=0.92,
                entry_allowed=True,
                reasons=["aligned"],
                category_scores={"session_structure": 0.9},
            )
        return SimpleNamespace(
            score=0.15,
            entry_allowed=False,
            reasons=["not_aligned"],
            category_scores={"session_structure": 0.1},
        )

    monkeypatch.setattr(skl, "detect_all_intraday_patterns", _fake_detect)
    monkeypatch.setattr(skl, "score_intraday_entry", _fake_score)

    loop = skl.SkimmerLoop(skl.cfg)
    loop.tick()

    assert calls["detect"] >= 1
    assert set(calls["score"]) == {"LONG", "SHORT"}
    assert "AAPL" in loop.open_trades
    assert loop.open_trades["AAPL"].side == "LONG"

    decisions = (tmp_path / "skimmer_decisions.jsonl").read_text(encoding="utf-8").splitlines()
    actions = [json.loads(line).get("action") for line in decisions if line.strip()]
    assert "ENTRY_LONG" in actions


def test_tick_uses_confluence_no_entry_path(monkeypatch, tmp_path):
    _configure_cfg(monkeypatch, tmp_path)
    bars = _bars_1m(130)
    calls = {"detect": 0, "score": 0}

    monkeypatch.setattr(skl, "SkimmerWatchlistManager", _WatchlistStub)
    monkeypatch.setattr(skl, "ExecutionSimulator", _ExecStub)
    monkeypatch.setattr(skl, "ib", lambda: None)
    monkeypatch.setattr(skl, "hist_bars_cached", lambda *args, **kwargs: bars.copy())
    monkeypatch.setattr(skl, "load_external_signal_bundle", lambda **kwargs: {"signals": {"AAPL": {"bias": 0.0, "confidence": 0.2}}})
    monkeypatch.setattr(skl, "emit_trade_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(skl, "build_trade_event", lambda **kwargs: kwargs)
    monkeypatch.setattr(skl, "write_telemetry_summary", lambda **kwargs: None)
    monkeypatch.setattr(skl, "log_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(skl, "log_trade", lambda *args, **kwargs: None)
    monkeypatch.setattr(skl, "log_equity", lambda *args, **kwargs: None)

    def _fake_detect(bars_5m, atr):
        calls["detect"] += 1
        return {}

    def _fake_score(bundle, _cfg):
        calls["score"] += 1
        return SimpleNamespace(
            score=0.42,
            entry_allowed=False,
            reasons=["below_threshold"],
            category_scores={"session_structure": 0.4},
        )

    monkeypatch.setattr(skl, "detect_all_intraday_patterns", _fake_detect)
    monkeypatch.setattr(skl, "score_intraday_entry", _fake_score)

    loop = skl.SkimmerLoop(skl.cfg)
    loop.tick()

    assert calls["detect"] >= 1
    assert calls["score"] >= 2
    assert "AAPL" not in loop.open_trades

    decisions = (tmp_path / "skimmer_decisions.jsonl").read_text(encoding="utf-8").splitlines()
    actions = [json.loads(line).get("action") for line in decisions if line.strip()]
    assert "NO_ENTRY" in actions


def test_manage_position_takes_partial_profit_then_keeps_remainder(monkeypatch, tmp_path):
    _configure_cfg(monkeypatch, tmp_path)

    monkeypatch.setattr(skl, "SkimmerWatchlistManager", _WatchlistStub)
    monkeypatch.setattr(skl, "ExecutionSimulator", _ExecStub)
    monkeypatch.setattr(skl, "emit_trade_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(skl, "build_trade_event", lambda **kwargs: kwargs)
    monkeypatch.setattr(skl, "log_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(skl, "log_trade", lambda *args, **kwargs: None)
    monkeypatch.setattr(skl, "log_equity", lambda *args, **kwargs: None)

    loop = skl.SkimmerLoop(skl.cfg)
    loop.open_trades["AAPL"] = skl.TradeState(
        symbol="AAPL",
        side="LONG",
        entry_price=100.0,
        entry_qty=10,
        current_qty=10,
        risk_distance=2.0,
        partial_taken=False,
        highest_since_entry=100.0,
        lowest_since_entry=100.0,
        entry_time=pd.Timestamp("2026-01-06T15:00:00Z").to_pydatetime(),
        stop_price=98.0,
        r_target_1=102.0,
    )

    frames = _frames_for_price(price=102.2, atr_5m=1.0)
    loop._manage_position("AAPL", frames, 102.2)

    assert "AAPL" in loop.open_trades
    st = loop.open_trades["AAPL"]
    assert st.partial_taken is True
    assert st.current_qty == 5
    assert st.stop_price >= st.entry_price


def test_manage_position_trailing_stop_only_after_partial(monkeypatch, tmp_path):
    _configure_cfg(monkeypatch, tmp_path)

    monkeypatch.setattr(skl, "SkimmerWatchlistManager", _WatchlistStub)
    monkeypatch.setattr(skl, "ExecutionSimulator", _ExecStub)
    monkeypatch.setattr(skl, "emit_trade_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(skl, "build_trade_event", lambda **kwargs: kwargs)
    monkeypatch.setattr(skl, "log_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(skl, "log_trade", lambda *args, **kwargs: None)
    monkeypatch.setattr(skl, "log_equity", lambda *args, **kwargs: None)

    loop = skl.SkimmerLoop(skl.cfg)
    loop.open_trades["AAPL"] = skl.TradeState(
        symbol="AAPL",
        side="LONG",
        entry_price=100.0,
        entry_qty=8,
        current_qty=8,
        risk_distance=2.0,
        partial_taken=False,
        highest_since_entry=103.0,
        lowest_since_entry=100.0,
        entry_time=pd.Timestamp("2026-01-06T15:00:00Z").to_pydatetime(),
        stop_price=95.0,
        r_target_1=104.0,
    )

    frames = _frames_for_price(price=100.0, atr_5m=1.0)

    # Before partial is taken, trailing stop must not fire.
    loop._manage_position("AAPL", frames, 100.0)
    assert "AAPL" in loop.open_trades
    assert loop.open_trades["AAPL"].current_qty == 8

    # After partial is marked taken, same price should trigger trailing stop.
    loop.open_trades["AAPL"].partial_taken = True
    loop._manage_position("AAPL", frames, 100.0)
    assert "AAPL" not in loop.open_trades


def test_tick_force_closes_open_position_near_session_end(monkeypatch, tmp_path):
    _configure_cfg(monkeypatch, tmp_path)
    bars = _bars_1m_near_close(22)

    monkeypatch.setattr(skl, "SkimmerWatchlistManager", _WatchlistStub)
    monkeypatch.setattr(skl, "ExecutionSimulator", _ExecStub)
    monkeypatch.setattr(skl, "ib", lambda: None)
    monkeypatch.setattr(skl, "hist_bars_cached", lambda *args, **kwargs: bars.copy())
    monkeypatch.setattr(skl, "load_external_signal_bundle", lambda **kwargs: {"signals": {}})
    monkeypatch.setattr(skl, "emit_trade_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(skl, "build_trade_event", lambda **kwargs: kwargs)
    monkeypatch.setattr(skl, "write_telemetry_summary", lambda **kwargs: None)
    monkeypatch.setattr(skl, "log_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(skl, "log_trade", lambda *args, **kwargs: None)
    monkeypatch.setattr(skl, "log_equity", lambda *args, **kwargs: None)
    monkeypatch.setattr(skl, "detect_all_intraday_patterns", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        skl,
        "score_intraday_entry",
        lambda *args, **kwargs: SimpleNamespace(
            score=0.10, entry_allowed=False, reasons=["skip"], category_scores={}
        ),
    )

    loop = skl.SkimmerLoop(skl.cfg)
    loop.open_trades["AAPL"] = skl.TradeState(
        symbol="AAPL",
        side="LONG",
        entry_price=101.0,
        entry_qty=12,
        current_qty=12,
        risk_distance=1.0,
        partial_taken=False,
        highest_since_entry=101.0,
        lowest_since_entry=101.0,
        entry_time=pd.Timestamp("2026-01-06T15:30:00Z").to_pydatetime(),
        stop_price=98.0,
        r_target_1=104.0,
    )

    loop.tick()
    assert "AAPL" not in loop.open_trades

    decisions = (tmp_path / "skimmer_decisions.jsonl").read_text(encoding="utf-8").splitlines()
    actions = [json.loads(line).get("action") for line in decisions if line.strip()]
    assert "EXIT_SESSION_END" in actions


def test_tick_skips_entry_when_risk_gate_blocks(monkeypatch, tmp_path):
    _configure_cfg(monkeypatch, tmp_path)
    bars = _bars_1m(130)
    calls = {"detect": 0}

    monkeypatch.setattr(skl, "SkimmerWatchlistManager", _WatchlistStub)
    monkeypatch.setattr(skl, "ExecutionSimulator", _ExecStub)
    monkeypatch.setattr(skl, "ib", lambda: None)
    monkeypatch.setattr(skl, "hist_bars_cached", lambda *args, **kwargs: bars.copy())
    monkeypatch.setattr(skl, "load_external_signal_bundle", lambda **kwargs: {"signals": {"AAPL": {"bias": 0.3, "confidence": 0.7}}})
    monkeypatch.setattr(skl, "emit_trade_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(skl, "build_trade_event", lambda **kwargs: kwargs)
    monkeypatch.setattr(skl, "write_telemetry_summary", lambda **kwargs: None)
    monkeypatch.setattr(skl, "log_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(skl, "log_trade", lambda *args, **kwargs: None)
    monkeypatch.setattr(skl, "log_equity", lambda *args, **kwargs: None)

    def _fake_detect(*args, **kwargs):
        calls["detect"] += 1
        return {}

    monkeypatch.setattr(skl, "detect_all_intraday_patterns", _fake_detect)

    loop = skl.SkimmerLoop(skl.cfg)
    loop.risk.state.session_locked = True
    loop.risk.state.lock_reason = "test_lock"
    loop.tick()

    assert "AAPL" not in loop.open_trades
    assert calls["detect"] == 0

    decisions = (tmp_path / "skimmer_decisions.jsonl").read_text(encoding="utf-8").splitlines()
    actions = [json.loads(line).get("action") for line in decisions if line.strip()]
    assert "SKIP_RISK_GATE" in actions


def test_tick_logs_no_fill_and_keeps_flat(monkeypatch, tmp_path):
    _configure_cfg(monkeypatch, tmp_path)
    bars = _bars_1m(140)

    monkeypatch.setattr(skl, "SkimmerWatchlistManager", _WatchlistStub)
    monkeypatch.setattr(skl, "ExecutionSimulator", _ExecNoFillStub)
    monkeypatch.setattr(skl, "ib", lambda: None)
    monkeypatch.setattr(skl, "hist_bars_cached", lambda *args, **kwargs: bars.copy())
    monkeypatch.setattr(skl, "load_external_signal_bundle", lambda **kwargs: {"signals": {"AAPL": {"bias": 0.5, "confidence": 0.9}}})
    monkeypatch.setattr(skl, "emit_trade_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(skl, "build_trade_event", lambda **kwargs: kwargs)
    monkeypatch.setattr(skl, "write_telemetry_summary", lambda **kwargs: None)
    monkeypatch.setattr(skl, "log_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(skl, "log_trade", lambda *args, **kwargs: None)
    monkeypatch.setattr(skl, "log_equity", lambda *args, **kwargs: None)
    monkeypatch.setattr(skl, "detect_all_intraday_patterns", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        skl,
        "score_intraday_entry",
        lambda bundle, _cfg: SimpleNamespace(
            score=0.85 if bundle.side == "LONG" else 0.10,
            entry_allowed=(bundle.side == "LONG"),
            reasons=["align"],
            category_scores={"session_structure": 0.8},
        ),
    )

    loop = skl.SkimmerLoop(skl.cfg)
    loop.tick()

    assert "AAPL" not in loop.open_trades
    decisions = (tmp_path / "skimmer_decisions.jsonl").read_text(encoding="utf-8").splitlines()
    actions = [json.loads(line).get("action") for line in decisions if line.strip()]
    assert "NO_FILL" in actions


def test_tick_caps_long_quantity_to_available_cash(monkeypatch, tmp_path):
    _configure_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr(skl.cfg, "EQUITY_START", 150.0, raising=False)
    bars = _bars_1m(140)

    monkeypatch.setattr(skl, "SkimmerWatchlistManager", _WatchlistStub)
    monkeypatch.setattr(skl, "ExecutionSimulator", _ExecStub)
    monkeypatch.setattr(skl, "ib", lambda: None)
    monkeypatch.setattr(skl, "hist_bars_cached", lambda *args, **kwargs: bars.copy())
    monkeypatch.setattr(skl, "load_external_signal_bundle", lambda **kwargs: {"signals": {"AAPL": {"bias": 0.5, "confidence": 0.9}}})
    monkeypatch.setattr(skl, "emit_trade_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(skl, "build_trade_event", lambda **kwargs: kwargs)
    monkeypatch.setattr(skl, "write_telemetry_summary", lambda **kwargs: None)
    monkeypatch.setattr(skl, "log_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(skl, "log_trade", lambda *args, **kwargs: None)
    monkeypatch.setattr(skl, "log_equity", lambda *args, **kwargs: None)
    monkeypatch.setattr(skl, "detect_all_intraday_patterns", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        skl,
        "score_intraday_entry",
        lambda bundle, _cfg: SimpleNamespace(
            score=0.88 if bundle.side == "LONG" else 0.05,
            entry_allowed=(bundle.side == "LONG"),
            reasons=["align"],
            category_scores={"session_structure": 0.8},
        ),
    )

    loop = skl.SkimmerLoop(skl.cfg)
    loop.tick()

    assert "AAPL" in loop.open_trades
    assert loop.open_trades["AAPL"].entry_qty == 1

    decisions = (tmp_path / "skimmer_decisions.jsonl").read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in decisions if line.strip()]
    entries = [row for row in parsed if row.get("action") == "ENTRY_LONG"]
    assert entries
    extras = entries[-1].get("extras", {})
    assert int(extras.get("requested_qty", 0)) == 1
    assert int(extras.get("sizing_qty", 0)) >= 1
