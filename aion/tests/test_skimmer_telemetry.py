import datetime as dt
import json
from types import SimpleNamespace

from aion.exec.skimmer_telemetry import SkimmerTelemetry


def test_log_decision_writes_expected_fields(tmp_path):
    cfg = SimpleNamespace(STATE_DIR=tmp_path)
    tel = SkimmerTelemetry(cfg)
    ts = dt.datetime(2026, 1, 10, 14, 30, tzinfo=dt.timezone.utc)

    tel.log_decision(
        symbol="aapl",
        action="entry_long",
        confluence_score=0.72,
        category_scores={"session_structure": 0.8, "pattern_confluence": 0.7},
        session_phase="range_extension",
        session_type="trend_day",
        patterns_detected=["pin_bar", "momentum_burst"],
        entry_price=185.23,
        stop_price=184.10,
        shares=150,
        risk_amount=169.50,
        reasons=["All timeframes aligned"],
        extras={"q_overlay_bias": 0.62},
        timestamp=ts,
    )

    out = tmp_path / "skimmer_decisions.jsonl"
    assert out.exists()
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])

    assert row["ts"] == "2026-01-10T14:30:00Z"
    assert row["symbol"] == "AAPL"
    assert row["action"] == "ENTRY_LONG"
    assert row["session_phase"] == "range_extension"
    assert row["session_type"] == "trend_day"
    assert row["shares"] == 150
    assert row["entry_price"] == 185.23
    assert row["extras"]["q_overlay_bias"] == 0.62


def test_log_decision_rotates_daily_file(tmp_path):
    cfg = SimpleNamespace(STATE_DIR=tmp_path)
    tel = SkimmerTelemetry(cfg)

    tel.log_decision(
        symbol="AAPL",
        action="NO_ENTRY",
        confluence_score=0.40,
        reasons=["below threshold"],
        timestamp=dt.datetime(2026, 1, 10, 15, 0, tzinfo=dt.timezone.utc),
    )
    tel.log_decision(
        symbol="MSFT",
        action="NO_ENTRY",
        confluence_score=0.41,
        reasons=["below threshold"],
        timestamp=dt.datetime(2026, 1, 11, 15, 0, tzinfo=dt.timezone.utc),
    )

    rolled = tmp_path / "skimmer_decisions.20260110.jsonl"
    active = tmp_path / "skimmer_decisions.jsonl"
    assert rolled.exists()
    assert active.exists()

    rolled_row = json.loads(rolled.read_text(encoding="utf-8").splitlines()[0])
    active_row = json.loads(active.read_text(encoding="utf-8").splitlines()[0])
    assert rolled_row["symbol"] == "AAPL"
    assert active_row["symbol"] == "MSFT"
