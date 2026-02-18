from pathlib import Path

import pytest

from aion.exec.shadow_state import apply_shadow_fill, load_shadow_positions, save_shadow_positions


def test_apply_shadow_fill_position_lifecycle(tmp_path: Path):
    shadow = tmp_path / "shadow_trades.json"

    rec = apply_shadow_fill(
        shadow,
        symbol="AAPL",
        action="BUY",
        filled_qty=100,
        avg_fill_price=10.0,
        timestamp="2026-01-01T00:00:00Z",
    )
    assert rec["qty"] == 100
    assert rec["avg_price"] == pytest.approx(10.0)

    rec = apply_shadow_fill(
        shadow,
        symbol="AAPL",
        action="BUY",
        filled_qty=50,
        avg_fill_price=12.0,
        timestamp="2026-01-01T00:01:00Z",
    )
    assert rec["qty"] == 150
    assert rec["avg_price"] == pytest.approx((100.0 * 10.0 + 50.0 * 12.0) / 150.0)

    rec = apply_shadow_fill(
        shadow,
        symbol="AAPL",
        action="SELL",
        filled_qty=40,
        avg_fill_price=11.0,
        timestamp="2026-01-01T00:02:00Z",
    )
    assert rec["qty"] == 110
    assert rec["avg_price"] == pytest.approx((100.0 * 10.0 + 50.0 * 12.0) / 150.0)

    rec = apply_shadow_fill(
        shadow,
        symbol="AAPL",
        action="SELL",
        filled_qty=200,
        avg_fill_price=9.0,
        timestamp="2026-01-01T00:03:00Z",
    )
    assert rec["qty"] == -90
    assert rec["avg_price"] == pytest.approx(9.0)

    rec = apply_shadow_fill(
        shadow,
        symbol="AAPL",
        action="BUY",
        filled_qty=90,
        avg_fill_price=8.0,
        timestamp="2026-01-01T00:04:00Z",
    )
    assert rec == {}
    assert load_shadow_positions(shadow) == {}


def test_save_shadow_positions_atomic_write(tmp_path: Path):
    shadow = tmp_path / "shadow_trades.json"
    save_shadow_positions(
        shadow,
        {
            "AAPL": {"qty": 10, "avg_price": 100.0, "last_updated": "x"},
            "MSFT": {"qty": 0, "avg_price": 0.0, "last_updated": "y"},
        },
    )
    assert shadow.exists()
    assert not (tmp_path / "shadow_trades.json.tmp").exists()
    data = load_shadow_positions(shadow)
    assert set(data.keys()) == {"AAPL"}
