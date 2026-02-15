import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aion.brain import novaspine_bridge as nsb


@pytest.fixture(autouse=True)
def _reset_bridge_state():
    nsb._LAST_API_FAIL_TS = 0.0
    yield
    nsb._LAST_API_FAIL_TS = 0.0


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
    assert isinstance(row.get("event_id"), str) and len(row["event_id"]) == 40


def test_normalize_event_generates_stable_event_id():
    ev = {"event_type": "trade.entry", "payload": {"symbol": "AAPL", "qty": 2}, "ts_utc": "2026-02-15T00:00:00+00:00"}
    n1 = nsb._normalize_event(ev)
    n2 = nsb._normalize_event(ev)
    assert n1["event_id"] == n2["event_id"]
    assert len(n1["event_id"]) == 40


def test_emit_trade_event_novaspine_api_success(monkeypatch):
    calls = []

    def _fake_json_request(url, method="GET", payload=None, token=None, timeout_sec=6.0):
        calls.append((str(url), str(method), payload))
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
    ingest = next(c for c in calls if c[0].endswith("/api/v1/memory/ingest"))
    assert isinstance(ingest[2], dict)
    assert isinstance(ingest[2].get("event_id"), str) and len(ingest[2]["event_id"]) == 40
    md = ingest[2].get("metadata", {})
    assert md.get("event_id") == ingest[2]["event_id"]
    assert str(ingest[2].get("source_id", "")).startswith("aion:trade.exit:")


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


def test_emit_trade_event_novaspine_api_cooldown_short_circuits(tmp_path: Path, monkeypatch):
    calls = {"n": 0}

    def _fake_json_request(*_args, **_kwargs):
        calls["n"] += 1
        raise RuntimeError("down")

    ticks = iter([100.0, 110.0])
    monkeypatch.setattr(nsb, "_json_request", _fake_json_request)
    monkeypatch.setattr(nsb.time, "monotonic", lambda: next(ticks))

    cfg = _cfg(
        MEMORY_BACKEND="novaspine_api",
        MEMORY_OUTBOX_DIR=tmp_path,
        MEMORY_FAIL_COOLDOWN_SEC=120.0,
    )
    out1 = nsb.emit_trade_event({"event_type": "trade.entry", "payload": {"symbol": "AAPL"}}, cfg)
    out2 = nsb.emit_trade_event({"event_type": "trade.entry", "payload": {"symbol": "AAPL"}}, cfg)

    assert out1["ok"] is False
    assert out2["ok"] is False
    assert "unreachable" in str(out1.get("error", ""))
    assert str(out2.get("error", "")).startswith("cooldown_active")
    assert calls["n"] == 1
