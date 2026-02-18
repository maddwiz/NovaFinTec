import json
from pathlib import Path

from aion.exec.order_events import IBOrderStatusHandler, attach_order_status_handler
from aion.exec.shadow_state import load_shadow_positions


class _Obj:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _trade(
    *,
    order_id: int = 1,
    symbol: str = "AAPL",
    action: str = "BUY",
    total_qty: int = 10,
    status: str = "Submitted",
    filled: int = 0,
    remaining: int = 10,
    avg_fill: float = 0.0,
):
    return _Obj(
        order=_Obj(orderId=order_id, action=action, totalQuantity=total_qty),
        contract=_Obj(symbol=symbol),
        orderStatus=_Obj(status=status, filled=filled, remaining=remaining, avgFillPrice=avg_fill),
    )


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s:
            continue
        out.append(json.loads(s))
    return out


def test_order_status_handler_tracks_incremental_fill_delta(tmp_path: Path):
    handler = IBOrderStatusHandler(state_dir=tmp_path, shadow_path=tmp_path / "shadow_trades.json")

    handler(_trade(status="Submitted", filled=0, remaining=10))
    handler(_trade(status="Submitted", filled=4, remaining=6, avg_fill=100.0))
    handler(_trade(status="Cancelled", filled=4, remaining=6, avg_fill=100.0))
    handler(_trade(status="Filled", filled=10, remaining=0, avg_fill=101.0))

    shadow = load_shadow_positions(tmp_path / "shadow_trades.json")
    assert shadow["AAPL"]["qty"] == 10

    events = _read_events(tmp_path / "audit_orders.jsonl")
    names = [e.get("event") for e in events]
    assert "ORDER_SUBMITTED" in names
    assert "ORDER_PARTIAL_FILL" in names
    assert "ORDER_CANCELLED" in names
    assert "ORDER_FILLED" in names


class _DummyEvent:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self


class _DummyIB:
    def __init__(self):
        self.orderStatusEvent = _DummyEvent()


def test_attach_order_status_handler_idempotent(tmp_path: Path):
    ib = _DummyIB()
    h1 = attach_order_status_handler(ib, state_dir=tmp_path)
    h2 = attach_order_status_handler(ib, state_dir=tmp_path)
    assert h1 is not None
    assert h2 is h1
    assert len(ib.orderStatusEvent.handlers) == 1
