import json
from pathlib import Path
from types import SimpleNamespace

from aion.brain import novaspine_bridge as nsb


def _cfg(**kwargs):
    base = {
        "MEMORY_ENABLE": True,
        "MEMORY_BACKEND": "filesystem",
        "MEMORY_NAMESPACE": "private/nova/actions",
        "MEMORY_OUTBOX_DIR": Path("logs") / "novaspine_outbox_aion",
        "MEMORY_TIMEOUT_SEC": 3.0,
        "MEMORY_SOURCE_PREFIX": "aion",
        "MEMORY_TOKEN": "",
        "MEMORY_NOVASPINE_URL": "http://127.0.0.1:8420",
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_build_trade_event_sanitizes_values():
    ev = nsb.build_trade_event(
        event_type="trade.entry",
        symbol="aapl",
        side="long",
        qty=-4,
        entry="198.11",
        exit="nan",
        pnl="inf",
        reason="test",
        confidence="0.75",
        regime="risk_on",
        extra={"fill_ratio": 0.9},
    )
    assert ev["event_type"] == "trade.entry"
    assert ev["payload"]["symbol"] == "AAPL"
    assert ev["payload"]["side"] == "LONG"
    assert ev["payload"]["qty"] == 0
    assert ev["payload"]["entry"] == 198.11
    assert ev["payload"]["exit"] == 0.0
    assert ev["payload"]["pnl"] == 0.0
    assert ev["payload"]["fill_ratio"] == 0.9


def test_emit_trade_event_disabled():
    out = nsb.emit_trade_event(
        {"event_type": "trade.entry", "payload": {"symbol": "AAPL"}},
        _cfg(MEMORY_ENABLE=False),
    )
    assert out["ok"] is True
    assert out["backend"] == "disabled"


def test_emit_trade_event_filesystem_outbox(tmp_path: Path):
    out = nsb.emit_trade_event(
        {"event_type": "trade.entry", "payload": {"symbol": "AAPL"}},
        _cfg(MEMORY_OUTBOX_DIR=tmp_path),
    )
    assert out["ok"] is True
    assert out["backend"] == "filesystem"
    p = Path(out["outbox_file"])
    assert p.exists()
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["event_type"] == "trade.entry"
    assert "ts_utc" in row


def test_emit_trade_event_novaspine_api_success(monkeypatch):
    calls = []

    def _fake_json_request(url, method="GET", payload=None, token=None, timeout_sec=6.0):
        calls.append((str(url), str(method)))
        if str(url).endswith("/api/v1/health"):
            return 200, {"ok": True}
        if str(url).endswith("/api/v1/memory/ingest"):
            return 200, {"ok": True}
        return 404, None

    monkeypatch.setattr(nsb, "_json_request", _fake_json_request)

    out = nsb.emit_trade_event(
        {"event_type": "trade.exit", "payload": {"symbol": "TSLA"}},
        _cfg(MEMORY_BACKEND="novaspine_api"),
    )
    assert out["ok"] is True
    assert out["backend"] == "novaspine_api"
    assert out["published"] == 1
    assert any(c[0].endswith("/api/v1/health") for c in calls)
    assert any(c[0].endswith("/api/v1/memory/ingest") for c in calls)


def test_emit_trade_event_novaspine_api_unreachable_fallback(tmp_path: Path, monkeypatch):
    def _fake_json_request(*_args, **_kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(nsb, "_json_request", _fake_json_request)

    out = nsb.emit_trade_event(
        {"event_type": "trade.partial_exit", "payload": {"symbol": "MSFT"}},
        _cfg(MEMORY_BACKEND="novaspine_api", MEMORY_OUTBOX_DIR=tmp_path),
    )
    assert out["ok"] is False
    assert out["backend"] == "novaspine_api"
    assert out["queued"] == 1
    assert "outbox_file" in out
    assert Path(out["outbox_file"]).exists()
    assert "unreachable" in str(out.get("error", ""))


def test_emit_trade_event_unknown_backend_falls_back_to_queue(tmp_path: Path):
    out = nsb.emit_trade_event(
        {"event_type": "trade.entry", "payload": {"symbol": "NFLX"}},
        _cfg(MEMORY_BACKEND="not_real", MEMORY_OUTBOX_DIR=tmp_path),
    )
    assert out["ok"] is False
    assert out["backend"] == "not_real"
    assert out["queued"] == 1
    assert Path(out["outbox_file"]).exists()
