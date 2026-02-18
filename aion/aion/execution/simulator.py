from __future__ import annotations

import math
import time
from dataclasses import dataclass

try:
    from ib_insync import MarketOrder, Stock
except Exception:  # pragma: no cover - optional import guard
    MarketOrder = None
    Stock = None


@dataclass
class FillResult:
    filled_qty: int
    avg_fill: float
    fill_ratio: float
    est_slippage_bps: float
    source: str = "simulator"
    status: str = "simulated"
    order_id: int = 0


class ExecutionSimulator:
    def __init__(self, cfg):
        self.cfg = cfg
        backend_raw = str(getattr(cfg, "EXECUTION_BACKEND", "simulator")).strip().lower()
        self.backend = backend_raw if backend_raw in {"simulator", "ib_paper"} else "simulator"
        self.ib_order_timeout_sec = float(max(0.5, float(getattr(cfg, "IB_ORDER_TIMEOUT_SEC", 4.0))))
        self.ib_order_poll_sec = float(max(0.05, float(getattr(cfg, "IB_ORDER_POLL_SEC", 0.2))))

    def _spread_bps(self, atr_pct: float, confidence: float) -> float:
        conf_boost = 1.0 - 0.25 * max(0.0, min(1.0, confidence))
        return (self.cfg.SPREAD_BPS_BASE + self.cfg.SPREAD_BPS_VOL_MULT * max(0.0, atr_pct)) * conf_boost

    def _simulate(
        self,
        *,
        side: str,
        qty: int,
        ref_price: float,
        atr_pct: float,
        confidence: float,
        allow_partial: bool = True,
        source: str = "simulator",
    ) -> FillResult:
        if qty <= 0 or ref_price <= 0:
            return FillResult(0, ref_price, 0.0, 0.0, source=source, status="rejected")

        spread_bps = self._spread_bps(atr_pct, confidence)
        queue_impact = self.cfg.EXEC_QUEUE_IMPACT_BPS * min(1.0, qty / 500.0)
        latency_impact = 0.8 * (self.cfg.EXEC_LATENCY_MS / 1000.0)
        slippage_bps = self.cfg.SLIPPAGE_BPS + queue_impact + latency_impact + (atr_pct * 10000.0 * 0.08)

        half_spread = ref_price * (spread_bps / 10000.0) * 0.5
        slip = ref_price * (slippage_bps / 10000.0)

        if side.upper() == "BUY":
            px = ref_price + half_spread + slip
        else:
            px = ref_price - half_spread - slip

        if allow_partial:
            raw_ratio = 0.45 + 0.55 * confidence - 0.75 * min(0.25, atr_pct)
            fill_ratio = max(self.cfg.EXEC_PARTIAL_FILL_MIN, min(self.cfg.EXEC_PARTIAL_FILL_MAX, raw_ratio))
        else:
            fill_ratio = 1.0

        filled_qty = max(0, int(math.floor(qty * fill_ratio)))
        if filled_qty == 0 and qty > 0 and fill_ratio > 0:
            filled_qty = 1

        return FillResult(
            filled_qty=filled_qty,
            avg_fill=float(px),
            fill_ratio=float(filled_qty / max(qty, 1)),
            est_slippage_bps=float(slippage_bps),
            source=source,
            status="simulated",
        )

    def _execute_ib_paper(
        self,
        *,
        ib_client,
        symbol: str,
        side: str,
        qty: int,
        ref_price: float,
        allow_partial: bool,
    ) -> FillResult | None:
        if ib_client is None or qty <= 0 or ref_price <= 0:
            return None
        if MarketOrder is None or Stock is None:
            return None
        sym = str(symbol).strip().upper()
        if not sym:
            return None

        action = "BUY" if str(side).strip().upper() == "BUY" else "SELL"
        contract = Stock(sym, "SMART", "USD")
        try:
            qualified = ib_client.qualifyContracts(contract)
            if qualified:
                contract = qualified[0]
        except Exception:
            # Continue with unqualified contract attempt.
            pass

        try:
            order = MarketOrder(action, int(max(1, qty)))
            order.tif = "DAY"
            trade = ib_client.placeOrder(contract, order)
        except Exception:
            return None

        deadline = time.monotonic() + self.ib_order_timeout_sec
        status = ""
        filled = 0
        remaining = int(max(0, qty))
        avg_fill = 0.0
        order_id = int(getattr(getattr(trade, "order", None), "orderId", 0) or 0)

        while time.monotonic() < deadline:
            try:
                ib_client.waitOnUpdate(timeout=self.ib_order_poll_sec)
            except Exception:
                time.sleep(self.ib_order_poll_sec)
            st = getattr(trade, "orderStatus", None)
            if st is None:
                continue
            status = str(getattr(st, "status", "") or "")
            filled = int(max(0, int(getattr(st, "filled", 0) or 0)))
            remaining = int(max(0, int(getattr(st, "remaining", max(0, qty - filled)) or 0)))
            avg_fill = float(getattr(st, "avgFillPrice", 0.0) or 0.0)
            if filled >= qty:
                break
            if status in {"Cancelled", "ApiCancelled", "Inactive"}:
                break
            if allow_partial and status in {"Filled", "Submitted", "PreSubmitted"} and filled > 0:
                break

        if remaining > 0:
            try:
                ib_client.cancelOrder(order)
            except Exception:
                pass
            # Short post-cancel wait for final status/fill delta.
            post_deadline = time.monotonic() + min(1.0, self.ib_order_timeout_sec * 0.25)
            while time.monotonic() < post_deadline:
                try:
                    ib_client.waitOnUpdate(timeout=self.ib_order_poll_sec)
                except Exception:
                    time.sleep(self.ib_order_poll_sec)
                st = getattr(trade, "orderStatus", None)
                if st is None:
                    continue
                status = str(getattr(st, "status", "") or status)
                filled = int(max(0, int(getattr(st, "filled", filled) or filled)))
                remaining = int(max(0, int(getattr(st, "remaining", max(0, qty - filled)) or max(0, qty - filled))))
                avg_fill = float(getattr(st, "avgFillPrice", avg_fill) or avg_fill)
                if status in {"Cancelled", "ApiCancelled", "Inactive", "Filled"}:
                    break

        filled = int(max(0, min(int(qty), int(filled))))
        if avg_fill <= 0:
            avg_fill = float(ref_price)
        slip_bps = 0.0
        if ref_price > 0:
            slip_bps = abs(float(avg_fill) - float(ref_price)) / float(ref_price) * 10000.0

        return FillResult(
            filled_qty=filled,
            avg_fill=float(avg_fill),
            fill_ratio=float(filled / max(1, int(qty))),
            est_slippage_bps=float(slip_bps),
            source="ib_paper",
            status=str(status or "submitted"),
            order_id=int(order_id),
        )

    def execute(
        self,
        side: str,
        qty: int,
        ref_price: float,
        atr_pct: float,
        confidence: float,
        allow_partial: bool = True,
        symbol: str | None = None,
        ib_client=None,
    ) -> FillResult:
        if self.backend == "ib_paper":
            ib_fill = self._execute_ib_paper(
                ib_client=ib_client,
                symbol=str(symbol or ""),
                side=side,
                qty=qty,
                ref_price=ref_price,
                allow_partial=bool(allow_partial),
            )
            if ib_fill is not None:
                return ib_fill
            return self._simulate(
                side=side,
                qty=qty,
                ref_price=ref_price,
                atr_pct=atr_pct,
                confidence=confidence,
                allow_partial=allow_partial,
                source="simulator_fallback",
            )
        return self._simulate(
            side=side,
            qty=qty,
            ref_price=ref_price,
            atr_pct=atr_pct,
            confidence=confidence,
            allow_partial=allow_partial,
            source="simulator",
        )

    def simulate_fill(
        self,
        *,
        side: str,
        qty: int,
        ref_price: float,
        atr_pct: float,
        confidence: float,
        allow_partial: bool = True,
    ) -> FillResult:
        return self._simulate(
            side=side,
            qty=qty,
            ref_price=ref_price,
            atr_pct=atr_pct,
            confidence=confidence,
            allow_partial=allow_partial,
            source="simulator",
        )
