from pathlib import Path

import qmods.novaspine_adapter as na
from qmods.novaspine_adapter import publish_events, write_jsonl_outbox


def test_write_jsonl_outbox(tmp_path: Path):
    events = [{"event_type": "x", "payload": {"a": 1}}]
    p = write_jsonl_outbox(events, outbox_dir=tmp_path, prefix="test_batch")
    assert p.exists()
    txt = p.read_text(encoding="utf-8").strip()
    assert '"event_type":"x"' in txt


def test_publish_filesystem(tmp_path: Path):
    events = [{"event_type": "decision.signal_export", "payload": {"signals_count": 7}}]
    res = publish_events(
        events=events,
        backend="filesystem",
        namespace="private/nova/actions",
        outbox_dir=tmp_path,
    )
    assert res.failed == 0
    assert res.queued == 1
    assert res.published == 0
    assert res.outbox_file is not None
    assert Path(res.outbox_file).exists()


def test_publish_unknown_backend_fallback(tmp_path: Path):
    events = [{"event_type": "governance.health_gate", "payload": {"ok": True}}]
    res = publish_events(
        events=events,
        backend="made_up_backend",
        namespace="private/nova/actions",
        outbox_dir=tmp_path,
    )
    assert res.failed == 0
    assert res.queued == 1
    assert res.outbox_file is not None


class _FakeResp:
    def __init__(self, status: int = 200, body: bytes = b"{}"):
        self.status = status
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None


def test_publish_novaspine_api_success(tmp_path: Path, monkeypatch):
    calls = []

    def _fake_urlopen(req, timeout=0):
        calls.append((req.method, req.full_url, req.data))
        if req.full_url.endswith("/api/v1/health"):
            return _FakeResp(200, b'{"status":"ok"}')
        if req.full_url.endswith("/api/v1/memory/ingest"):
            return _FakeResp(200, b'{"chunk_ids":["x"],"count":1}')
        if req.full_url.endswith("/api/v1/events/track-batch"):
            return _FakeResp(200, b'{"count":2}')
        return _FakeResp(404, b"{}")

    monkeypatch.setattr(na.request, "urlopen", _fake_urlopen)
    events = [
        {"event_type": "decision.signal_export", "payload": {"signals_count": 20}, "trust": 0.8},
        {"event_type": "governance.health_gate", "payload": {"ok": True}, "trust": 0.9},
    ]
    res = publish_events(
        events=events,
        backend="novaspine_api",
        namespace="private/nova/actions",
        outbox_dir=tmp_path,
        novaspine_base_url="http://127.0.0.1:8420",
    )
    assert res.published == 2
    assert res.failed == 0
    assert res.queued == 0
    urls = [u for _, u, _ in calls]
    assert any(u.endswith("/api/v1/health") for u in urls)
    assert sum(1 for u in urls if u.endswith("/api/v1/memory/ingest")) == 2


def test_publish_novaspine_api_unreachable_falls_back_to_outbox(tmp_path: Path, monkeypatch):
    def _raise_urlopen(req, timeout=0):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(na.request, "urlopen", _raise_urlopen)
    events = [{"event_type": "decision.signal_export", "payload": {"signals_count": 10}, "trust": 0.7}]
    res = publish_events(
        events=events,
        backend="novaspine_api",
        namespace="private/nova/actions",
        outbox_dir=tmp_path,
        novaspine_base_url="http://127.0.0.1:8420",
    )
    assert res.published == 0
    assert res.failed == 0
    assert res.queued == 1
    assert res.outbox_file is not None
    assert Path(res.outbox_file).exists()
    assert str(res.error or "").startswith("unreachable:")


def test_event_namespace_preserved_in_novaspine_payload():
    ev = {
        "event_type": "governance.health_gate",
        "namespace": "private/c3/governance",
        "payload": {"ok": True},
        "trust": 0.9,
    }
    payload = na._event_to_novaspine_ingest(ev, namespace="private/nova/actions")
    assert payload["metadata"]["namespace"] == "private/c3/governance"
    assert payload["source_id"].startswith("novafintec:")
