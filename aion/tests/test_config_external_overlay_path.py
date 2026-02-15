import importlib
from pathlib import Path

import aion.config as cfg_mod


def test_config_prefers_q_overlay_path_when_present(tmp_path, monkeypatch):
    aion_home = tmp_path / "aion_home"
    q_home = tmp_path / "q_home"
    (q_home / "runs_plus").mkdir(parents=True, exist_ok=True)
    (q_home / "runs_plus" / "q_signal_overlay.json").write_text("{}", encoding="utf-8")

    monkeypatch.setenv("AION_HOME", str(aion_home))
    monkeypatch.setenv("AION_Q_HOME", str(q_home))
    monkeypatch.delenv("AION_EXT_SIGNAL_FILE", raising=False)

    cfg = importlib.reload(cfg_mod)
    assert Path(cfg.EXT_SIGNAL_FILE) == (q_home / "runs_plus" / "q_signal_overlay.json")


def test_config_falls_back_to_state_overlay_path_when_q_missing(tmp_path, monkeypatch):
    aion_home = tmp_path / "aion_home"
    q_home = tmp_path / "q_home_missing"

    monkeypatch.setenv("AION_HOME", str(aion_home))
    monkeypatch.setenv("AION_Q_HOME", str(q_home))
    monkeypatch.delenv("AION_EXT_SIGNAL_FILE", raising=False)

    cfg = importlib.reload(cfg_mod)
    assert Path(cfg.EXT_SIGNAL_FILE) == (aion_home / "state" / "q_signal_overlay.json")


def test_config_reads_external_overlay_critical_flag(tmp_path, monkeypatch):
    aion_home = tmp_path / "aion_home"
    monkeypatch.setenv("AION_HOME", str(aion_home))
    monkeypatch.setenv("AION_EXT_SIGNAL_CRITICAL", "1")
    cfg = importlib.reload(cfg_mod)
    assert cfg.EXT_SIGNAL_CRITICAL is True
