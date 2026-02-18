import numpy as np

import aion.exec.paper_loop as pl


def test_long_stop_multiplier_wider_than_short(monkeypatch):
    monkeypatch.setattr(pl.cfg, "STOP_ATR_LONG", 2.5)
    monkeypatch.setattr(pl.cfg, "STOP_ATR_SHORT", 2.0)
    monkeypatch.setattr(pl.cfg, "STOP_VOL_ADAPTIVE", False)

    rets = 0.002 * np.sin(np.linspace(0.0, 12.0, 200))
    long_mult, long_expanded = pl._entry_stop_atr_multiple("LONG", rets)
    short_mult, short_expanded = pl._entry_stop_atr_multiple("SHORT", rets)

    assert long_expanded is False
    assert short_expanded is False
    assert abs(long_mult - 2.5) < 1e-12
    assert abs(short_mult - 2.0) < 1e-12
    assert long_mult > short_mult


def test_vol_adaptive_expansion_triggers_in_high_vol_regime(monkeypatch):
    monkeypatch.setattr(pl.cfg, "STOP_ATR_LONG", 2.5)
    monkeypatch.setattr(pl.cfg, "STOP_VOL_ADAPTIVE", True)
    monkeypatch.setattr(pl.cfg, "STOP_VOL_LOOKBACK", 20)
    monkeypatch.setattr(pl.cfg, "STOP_VOL_EXPANSION_MULT", 1.3)

    low = 0.0015 * np.sin(np.linspace(0.0, 30.0, 280))
    high = 0.03 * np.sign(np.sin(np.linspace(0.0, 25.0, 40)))
    rets = np.r_[low, high]

    mult, expanded = pl._entry_stop_atr_multiple("LONG", rets)
    assert expanded is True
    assert abs(mult - (2.5 * 1.3)) < 1e-9


def test_disabled_vol_adaptive_uses_static_multipliers(monkeypatch):
    monkeypatch.setattr(pl.cfg, "STOP_ATR_SHORT", 1.8)
    monkeypatch.setattr(pl.cfg, "STOP_VOL_ADAPTIVE", False)
    monkeypatch.setattr(pl.cfg, "STOP_VOL_LOOKBACK", 20)
    monkeypatch.setattr(pl.cfg, "STOP_VOL_EXPANSION_MULT", 1.3)

    low = 0.0015 * np.sin(np.linspace(0.0, 30.0, 280))
    high = 0.03 * np.sign(np.sin(np.linspace(0.0, 25.0, 40)))
    rets = np.r_[low, high]

    mult, expanded = pl._entry_stop_atr_multiple("SHORT", rets)
    assert expanded is False
    assert abs(mult - 1.8) < 1e-12
