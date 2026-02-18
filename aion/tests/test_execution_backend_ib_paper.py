from types import SimpleNamespace

from aion.execution.simulator import ExecutionSimulator


def _cfg(backend: str):
    return SimpleNamespace(
        EXECUTION_BACKEND=str(backend),
        IB_ORDER_TIMEOUT_SEC=1.0,
        IB_ORDER_POLL_SEC=0.05,
        SPREAD_BPS_BASE=2.5,
        SPREAD_BPS_VOL_MULT=18.0,
        EXEC_QUEUE_IMPACT_BPS=3.0,
        EXEC_LATENCY_MS=250,
        SLIPPAGE_BPS=5.0,
        EXEC_PARTIAL_FILL_MIN=0.35,
        EXEC_PARTIAL_FILL_MAX=1.0,
    )


class _FakeIB:
    def __init__(self, *, fill_qty: int | None = None, avg_fill: float = 101.0, status: str = "Filled", fail: bool = False):
        self.fill_qty = fill_qty
        self.avg_fill = float(avg_fill)
        self.status = str(status)
        self.fail = bool(fail)
        self.trade = None

    def qualifyContracts(self, contract):
        return [contract]

    def placeOrder(self, _contract, order):
        if self.fail:
            raise RuntimeError("placeOrder failed")
        total = int(getattr(order, "totalQuantity", 0) or 0)
        filled = int(total if self.fill_qty is None else max(0, min(total, int(self.fill_qty))))
        remaining = int(max(0, total - filled))
        self.trade = SimpleNamespace(
            order=SimpleNamespace(orderId=77, totalQuantity=total, action=getattr(order, "action", "BUY")),
            orderStatus=SimpleNamespace(
                status=self.status,
                filled=filled,
                remaining=remaining,
                avgFillPrice=self.avg_fill,
            ),
        )
        return self.trade

    def waitOnUpdate(self, timeout=None):
        return True

    def cancelOrder(self, _order):
        if self.trade is not None:
            self.trade.orderStatus.status = "Cancelled"


def test_execute_simulator_backend_default_source():
    exe = ExecutionSimulator(_cfg("simulator"))
    fill = exe.execute(
        side="BUY",
        qty=10,
        ref_price=100.0,
        atr_pct=0.01,
        confidence=0.7,
        allow_partial=True,
        symbol="AAPL",
        ib_client=None,
    )
    assert fill.filled_qty > 0
    assert fill.source == "simulator"


def test_execute_ib_paper_backend_uses_ib_fill_when_available():
    exe = ExecutionSimulator(_cfg("ib_paper"))
    fake_ib = _FakeIB(fill_qty=7, avg_fill=101.25, status="Filled", fail=False)
    fill = exe.execute(
        side="BUY",
        qty=7,
        ref_price=100.0,
        atr_pct=0.01,
        confidence=0.8,
        allow_partial=False,
        symbol="AAPL",
        ib_client=fake_ib,
    )
    assert fill.source == "ib_paper"
    assert fill.filled_qty == 7
    assert abs(fill.avg_fill - 101.25) < 1e-9
    assert fill.order_id == 77


def test_execute_ib_paper_backend_falls_back_to_simulator_on_failure():
    exe = ExecutionSimulator(_cfg("ib_paper"))
    fake_ib = _FakeIB(fail=True)
    fill = exe.execute(
        side="SELL",
        qty=9,
        ref_price=100.0,
        atr_pct=0.01,
        confidence=0.6,
        allow_partial=True,
        symbol="MSFT",
        ib_client=fake_ib,
    )
    assert fill.filled_qty > 0
    assert fill.source == "simulator_fallback"


def test_simulate_fill_alias_stays_simulator_only():
    exe = ExecutionSimulator(_cfg("ib_paper"))
    fill = exe.simulate_fill(
        side="BUY",
        qty=5,
        ref_price=100.0,
        atr_pct=0.01,
        confidence=0.5,
        allow_partial=False,
    )
    assert fill.source == "simulator"
