"""Atomic shadow position state helpers for runtime reconciliation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def _normalize_action(action: str) -> str:
    a = str(action or "").strip().upper()
    return "BUY" if a == "BUY" else "SELL"


def _as_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _as_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def load_shadow_positions(shadow_path: Path) -> dict[str, dict]:
    p = Path(shadow_path)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}

    out: dict[str, dict] = {}
    for sym, info in raw.items():
        key = _normalize_symbol(sym)
        if not key or not isinstance(info, dict):
            continue
        out[key] = {
            "qty": _as_int(info.get("qty", 0)),
            "avg_price": _as_float(info.get("avg_price", 0.0)),
            "last_updated": str(info.get("last_updated", "")),
        }
    return out


def save_shadow_positions(shadow_path: Path, positions: dict[str, dict]) -> Path:
    p = Path(shadow_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, dict] = {}
    for sym, info in (positions or {}).items():
        key = _normalize_symbol(sym)
        if not key:
            continue
        qty = _as_int((info or {}).get("qty", 0))
        if qty == 0:
            continue
        payload[key] = {
            "qty": qty,
            "avg_price": _as_float((info or {}).get("avg_price", 0.0)),
            "last_updated": str((info or {}).get("last_updated", _utc_now())),
        }

    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(p)
    return p


def apply_shadow_fill(
    shadow_path: Path,
    *,
    symbol: str,
    action: str,
    filled_qty: int,
    avg_fill_price: float,
    timestamp: str | None = None,
) -> dict:
    """
    Apply a fill delta to the shadow position file atomically.

    Returns the updated symbol record, or {} when flat after update.
    """
    qty = max(0, _as_int(filled_qty, 0))
    if qty <= 0:
        return {}

    sym = _normalize_symbol(symbol)
    if not sym:
        return {}
    side = _normalize_action(action)
    px = max(0.0, _as_float(avg_fill_price, 0.0))
    ts = str(timestamp or _utc_now())

    positions = load_shadow_positions(Path(shadow_path))
    cur = positions.get(sym, {})
    old_qty = _as_int(cur.get("qty", 0))
    old_avg = _as_float(cur.get("avg_price", 0.0))
    delta = qty if side == "BUY" else -qty
    new_qty = old_qty + delta

    if new_qty == 0:
        positions.pop(sym, None)
        save_shadow_positions(Path(shadow_path), positions)
        return {}

    old_sign = 0 if old_qty == 0 else (1 if old_qty > 0 else -1)
    new_sign = 1 if new_qty > 0 else -1
    delta_sign = 1 if delta > 0 else -1

    if old_qty == 0:
        new_avg = px
    elif old_sign == delta_sign and old_sign == new_sign:
        # Increasing position in same direction: weighted average entry.
        new_avg = (abs(old_qty) * old_avg + abs(delta) * px) / max(1, abs(new_qty))
    elif old_sign == new_sign:
        # Reducing position without crossing zero: keep legacy cost basis.
        new_avg = old_avg
    else:
        # Position flipped direction: reset basis to new fill price.
        new_avg = px

    rec = {"qty": int(new_qty), "avg_price": float(new_avg), "last_updated": ts}
    positions[sym] = rec
    save_shadow_positions(Path(shadow_path), positions)
    return dict(rec)
