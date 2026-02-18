import importlib

import aion.config as cfg_mod


def test_day_skimmer_mode_applies_intraday_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("AION_HOME", str(tmp_path / "aion_home"))
    monkeypatch.setenv("AION_TRADING_MODE", "day_skimmer")
    for k in [
        "AION_HIST_BAR_SIZE",
        "AION_HIST_DURATION",
        "AION_HIST_USE_RTH",
        "AION_LOOP_SECONDS",
        "AION_MAX_TRADES_PER_DAY",
        "AION_MAX_HOLD_CYCLES",
    ]:
        monkeypatch.delenv(k, raising=False)

    cfg = importlib.reload(cfg_mod)
    assert cfg.TRADING_MODE == "day_skimmer"
    assert cfg.HIST_BAR_SIZE == "1 min"
    assert cfg.HIST_DURATION == "3 D"
    assert cfg.HIST_USE_RTH is False
    assert int(cfg.LOOP_SECONDS) == 12
    assert int(cfg.MAX_TRADES_PER_DAY) == 30
    assert int(cfg.MAX_HOLD_CYCLES) == 6
    assert cfg.INTRADAY_CONFIRM_ENABLED is True
    assert float(cfg.INTRADAY_MIN_ALIGNMENT_SCORE) >= 0.60
    assert cfg.PARTIAL_PROFIT_ENABLED is True
    assert float(cfg.PARTIAL_PROFIT_R_MULTIPLE) > 0.0
    assert float(cfg.PARTIAL_PROFIT_FRACTION) > 0.0
    assert cfg.TRAILING_STOP_ENABLED is True
    assert float(cfg.TRAILING_STOP_ATR_MULTIPLE) > 0.0
    assert float(cfg.STOP_ATR_LONG) > 0.0
    assert float(cfg.STOP_ATR_SHORT) > 0.0
    assert cfg.STOP_VOL_ADAPTIVE is True
    assert int(cfg.STOP_VOL_LOOKBACK) > 0
    assert float(cfg.STOP_VOL_EXPANSION_MULT) >= 1.0


def test_long_term_mode_respects_explicit_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("AION_HOME", str(tmp_path / "aion_home"))
    monkeypatch.setenv("AION_TRADING_MODE", "long_term")
    monkeypatch.setenv("AION_HIST_BAR_SIZE", "15 mins")
    monkeypatch.setenv("AION_LOOP_SECONDS", "45")

    cfg = importlib.reload(cfg_mod)
    assert cfg.TRADING_MODE == "long_term"
    assert cfg.HIST_BAR_SIZE == "15 mins"
    assert int(cfg.LOOP_SECONDS) == 45
    assert cfg.INTRADAY_CONFIRM_ENABLED is False
