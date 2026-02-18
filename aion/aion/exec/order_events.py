"""IB order-status callback handling for fills/cancels/rejects."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from .audit_log import audit_log
from .shadow_state import apply_shadow_fill
from ..utils.logging_utils import log_run


def _to_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _to_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


class IBOrderStatusHandler:
    """Tracks incremental fills and mirrors them into shadow state."""

    def __init__(self, *, state_dir: Path, shadow_path: Path | None = None):
        self.state_dir = Path(state_dir)
        self.shadow_path = Path(shadow_path) if shadow_path is not None else (self.state_dir / "shadow_trades.json")
        self._last_filled_by_order: dict[int, int] = {}

    def __call__(self, trade):
        try:
            self.handle(trade)
        except Exception as exc:
            log_run(f"order status callback error: {exc}")

    def handle(self, trade) -> None:
        now_utc = dt.datetime.now(dt.timezone.utc).isoformat()
        order = getattr(trade, "order", None)
        contract = getattr(trade, "contract", None)
        status_obj = getattr(trade, "orderStatus", None)

        order_id = _to_int(getattr(order, "orderId", 0), 0)
        symbol = str(getattr(contract, "symbol", "")).strip().upper()
        action = str(getattr(order, "action", "")).strip().upper()
        total_qty = max(0, _to_int(getattr(order, "totalQuantity", 0), 0))
        status = str(getattr(status_obj, "status", "")).strip()
        filled_total = max(0, _to_int(getattr(status_obj, "filled", 0), 0))
        remaining_qty = max(0, _to_int(getattr(status_obj, "remaining", 0), 0))
        avg_fill_price = _to_float(getattr(status_obj, "avgFillPrice", 0.0), 0.0)

        base = {
            "timestamp": now_utc,
            "order_id": int(order_id),
            "symbol": symbol,
            "status": status,
            "filled_qty": int(filled_total),
            "remaining_qty": int(remaining_qty),
            "avg_fill_price": float(avg_fill_price),
            "action": action,
            "total_qty": int(total_qty),
        }

        if status in {"Filled"}:
            status_event = "ORDER_FILLED"
        elif status in {"Cancelled", "ApiCancelled", "PendingCancel"}:
            status_event = "ORDER_CANCELLED"
        elif status in {"Inactive"}:
            status_event = "ORDER_REJECTED"
        elif status in {"PreSubmitted", "Submitted", "PendingSubmit"}:
            status_event = "ORDER_SUBMITTED"
        else:
            status_event = "ORDER_STATUS"

        prev_filled = int(self._last_filled_by_order.get(order_id, 0))
        fill_delta = max(0, filled_total - prev_filled)
        if filled_total >= prev_filled:
            self._last_filled_by_order[order_id] = int(filled_total)

        # Always log status transition.
        audit_log({"event": status_event, **base}, log_dir=self.state_dir)

        # Mirror fill deltas exactly once into shadow.
        if fill_delta > 0 and symbol:
            fill_event = "ORDER_PARTIAL_FILL" if remaining_qty > 0 else "ORDER_FILLED"
            audit_log(
                {
                    "event": fill_event,
                    **base,
                    "fill_delta_qty": int(fill_delta),
                },
                log_dir=self.state_dir,
            )
            apply_shadow_fill(
                self.shadow_path,
                symbol=symbol,
                action=action,
                filled_qty=int(fill_delta),
                avg_fill_price=float(avg_fill_price),
                timestamp=now_utc,
            )


def attach_order_status_handler(
    ib_client,
    *,
    state_dir: Path,
    shadow_path: Path | None = None,
) -> IBOrderStatusHandler | None:
    """Attach handler once; safe to call repeatedly on reconnect."""
    if ib_client is None:
        return None
    existing = getattr(ib_client, "_aion_order_status_handler", None)
    if isinstance(existing, IBOrderStatusHandler):
        return existing

    event = getattr(ib_client, "orderStatusEvent", None)
    if event is None:
        return None

    handler = IBOrderStatusHandler(state_dir=Path(state_dir), shadow_path=shadow_path)
    try:
        event += handler
    except Exception:
        return None
    try:
        setattr(ib_client, "_aion_order_status_handler", handler)
    except Exception:
        pass
    return handler
