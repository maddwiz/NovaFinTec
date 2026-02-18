import aion.exec.paper_loop as pl


class _DummyKS:
    def __init__(self):
        self.rows = []

    def register_trade(self, pnl, today):
        self.rows.append((float(pnl), str(today)))


class _DummySink:
    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append(dict(record))
        return record


class _DummyFill:
    def __init__(self, avg_fill=0.0, fill_ratio=1.0, est_slippage_bps=0.0, filled_qty=0):
        self.avg_fill = float(avg_fill)
        self.fill_ratio = float(fill_ratio)
        self.est_slippage_bps = float(est_slippage_bps)
        self.filled_qty = int(filled_qty)


class _DummyExe:
    def __init__(self, fill: _DummyFill):
        self._fill = fill

    def execute(self, **_kwargs):
        return self._fill


def test_telemetry_log_trade_decision_writes_expected_shape(monkeypatch):
    sink = _DummySink()
    monkeypatch.setattr(pl.cfg, "TELEMETRY_ENABLED", True, raising=False)

    pl._telemetry_log_trade_decision(
        sink,
        symbol="aapl",
        decision="enter_long",
        q_overlay_bias=0.42,
        q_overlay_confidence=0.61,
        confluence_score=0.73,
        intraday_alignment_score=0.55,
        regime="trending",
        governor_compound_scalar=0.37,
        entry_price=185.2,
        stop_price=183.9,
        risk_distance=1.3,
        position_size_shares=120,
        reasons=["Aligned with trend"],
        slippage_bps=4.0,
        estimated_slippage_bps=3.5,
    )

    assert len(sink.records) == 1
    row = sink.records[0]
    assert row["symbol"] == "AAPL"
    assert row["decision"] == "ENTER_LONG"
    assert abs(float(row["q_overlay_bias"]) - 0.42) < 1e-12
    assert abs(float(row["confluence_score"]) - 0.73) < 1e-12
    assert row["position_size_shares"] == 120
    assert row["reasons"] == ["Aligned with trend"]


def test_close_position_emits_exit_telemetry(monkeypatch):
    monkeypatch.setattr(pl.cfg, "TELEMETRY_ENABLED", True, raising=False)
    monkeypatch.setattr(pl, "log_trade", lambda *args, **kwargs: None)
    monkeypatch.setattr(pl, "_emit_trade_memory_event", lambda *args, **kwargs: None)

    sink = _DummySink()
    ks = _DummyKS()
    open_positions = {
        "AAPL": {
            "qty": 10,
            "side": "LONG",
            "entry": 100.0,
            "entry_ts": "2026-02-18T14:00:00+00:00",
            "stop": 98.5,
            "target": 103.0,
            "trail_stop": 99.0,
            "init_risk": 1.5,
            "bars_held": 3,
            "confidence": 0.66,
            "regime": "trending",
            "q_overlay_bias": 0.31,
            "q_overlay_confidence": 0.58,
            "intraday_score": 0.40,
        }
    }

    cash, closed_pnl = pl._close_position(
        open_positions=open_positions,
        symbol="AAPL",
        fill_price=102.0,
        reason="target_hit",
        cash=0.0,
        closed_pnl=0.0,
        ks=ks,
        today="2026-02-18",
        fill_ratio=1.0,
        slippage_bps=5.0,
        monitor=None,
        memory_runtime_context=None,
        telemetry_sink=sink,
        governor_compound_scalar=0.44,
    )

    assert "AAPL" not in open_positions
    assert abs(float(cash) - 1020.0) < 1e-12
    assert abs(float(closed_pnl) - 20.0) < 1e-12
    assert len(sink.records) == 1
    row = sink.records[0]
    assert row["decision"] == "EXIT_TARGET_HIT"
    assert abs(float(row["pnl_realized"]) - 20.0) < 1e-12
    assert row["symbol"] == "AAPL"


def test_partial_close_emits_partial_profit_telemetry(monkeypatch):
    monkeypatch.setattr(pl.cfg, "TELEMETRY_ENABLED", True, raising=False)
    monkeypatch.setattr(pl.cfg, "PARTIAL_PROFIT_FRACTION", 0.50, raising=False)
    monkeypatch.setattr(pl, "log_trade", lambda *args, **kwargs: None)
    monkeypatch.setattr(pl, "_emit_trade_memory_event", lambda *args, **kwargs: None)

    sink = _DummySink()
    ks = _DummyKS()
    pos = {
        "qty": 10,
        "side": "LONG",
        "entry": 100.0,
        "entry_ts": "2026-02-18T14:00:00+00:00",
        "stop": 98.5,
        "target": 103.0,
        "trail_stop": 99.0,
        "init_risk": 1.5,
        "bars_held": 2,
        "partial_taken": False,
        "confidence": 0.61,
        "regime": "trending",
        "q_overlay_bias": 0.25,
        "q_overlay_confidence": 0.52,
        "intraday_score": 0.33,
        "atr_pct": 0.01,
    }
    exe = _DummyExe(_DummyFill(avg_fill=102.0, fill_ratio=1.0, est_slippage_bps=4.0, filled_qty=5))

    cash, closed_pnl, fully_closed = pl._partial_close(
        pos=pos,
        symbol="AAPL",
        price=102.0,
        exe=exe,
        cash=0.0,
        closed_pnl=0.0,
        ks=ks,
        today="2026-02-18",
        monitor=None,
        memory_runtime_context=None,
        telemetry_sink=sink,
        governor_compound_scalar=0.41,
    )

    assert fully_closed is False
    assert pos["qty"] == 5
    assert pos["partial_taken"] is True
    assert abs(float(cash) - 510.0) < 1e-12
    assert abs(float(closed_pnl) - 10.0) < 1e-12
    assert len(sink.records) == 1
    row = sink.records[0]
    assert row["decision"] == "PARTIAL_PROFIT_1R"
    assert abs(float(row["pnl_realized"]) - 10.0) < 1e-12
