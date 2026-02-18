import json
from pathlib import Path

from aion.exec.telemetry_summary import build_telemetry_summary, write_telemetry_summary


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def test_build_telemetry_summary_computes_core_metrics(tmp_path):
    decisions = tmp_path / "trade_decisions.jsonl"
    _write_jsonl(
        decisions,
        [
            {
                "timestamp": "2026-02-18T14:30:00Z",
                "symbol": "AAPL",
                "decision": "ENTRY_LONG",
                "regime": "trend_day",
                "reasons": ["Opening-range breakout aligned"],
                "slippage_bps": 4.0,
                "estimated_slippage_bps": 3.0,
            },
            {
                "timestamp": "2026-02-18T14:45:00Z",
                "symbol": "AAPL",
                "decision": "EXIT_TRAILING_STOP",
                "regime": "trend_day",
                "pnl_realized": 120.0,
                "reasons": ["Opening-range breakout aligned"],
                "entry_category_scores": {
                    "session_structure": 0.90,
                    "multi_timeframe": 0.80,
                    "pattern_confluence": 0.70,
                },
                "slippage_bps": 5.0,
                "estimated_slippage_bps": 4.0,
            },
            {
                "timestamp": "2026-02-18T15:10:00Z",
                "symbol": "MSFT",
                "decision": "EXIT_INITIAL_STOP",
                "regime": "range_day",
                "pnl_realized": -60.0,
                "reasons": ["No timeframe alignment"],
                "entry_category_scores": {
                    "multi_timeframe": 0.95,
                    "session_structure": 0.40,
                },
                "slippage_bps": 6.0,
                "estimated_slippage_bps": 5.0,
            },
        ],
    )

    out = build_telemetry_summary(decisions_path=decisions, rolling_window=20)
    assert out["closed_trade_events"] == 2
    assert out["rolling_hit_rate"] == 0.5
    assert out["total_hit_rate"] == 0.5
    assert out["most_profitable_regime"] == "trend_day"
    assert out["worst_regime"] == "range_day"
    assert out["avg_slippage_bps"] > out["avg_estimated_slippage_bps"]
    assert len(out["top_win_reasons"]) >= 1
    assert len(out["top_loss_reasons"]) >= 1
    assert out["top_win_signal_category"] == "session_structure"
    assert out["top_loss_signal_category"] == "multi_timeframe"


def test_write_telemetry_summary_creates_json_output(tmp_path):
    decisions = tmp_path / "trade_decisions.jsonl"
    output = tmp_path / "telemetry_summary.json"
    _write_jsonl(
        decisions,
        [
            {
                "timestamp": "2026-02-18T15:00:00Z",
                "symbol": "AAPL",
                "decision": "EXIT_TRAILING_STOP",
                "regime": "trend_day",
                "pnl_realized": 10.0,
                "reasons": ["Rising VWAP"],
            }
        ],
    )
    payload = write_telemetry_summary(decisions_path=decisions, output_path=output, rolling_window=5)
    assert output.exists()
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["closed_trade_events"] == 1
    assert saved["rolling_window"] == 5
    assert payload["closed_trade_events"] == 1
