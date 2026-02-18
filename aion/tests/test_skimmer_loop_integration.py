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

