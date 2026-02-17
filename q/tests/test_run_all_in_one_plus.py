from pathlib import Path
import os

import tools.run_all_in_one_plus as raip


def test_has_any_report_detects_report_file(tmp_path: Path):
    assert raip.has_any_report(tmp_path) is False
    (tmp_path / "report_plus.html").write_text("<html></html>", encoding="utf-8")
    assert raip.has_any_report(tmp_path) is True


def test_ensure_report_exists_builds_when_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(raip, "ROOT", tmp_path)

    calls = []

    def _fake_run_script(relpath: str, args=None):
        calls.append(relpath)
        if relpath == "tools/build_report_plus.py":
            (tmp_path / "report_plus.html").write_text("<html></html>", encoding="utf-8")
            return True, 0
        return False, 1

    monkeypatch.setattr(raip, "run_script", _fake_run_script)

    ok = raip.ensure_report_exists()
    assert ok is True
    assert calls and calls[0] == "tools/build_report_plus.py"


def test_ensure_report_exists_noop_when_present(tmp_path: Path, monkeypatch):
    (tmp_path / "report_best_plus.html").write_text("<html></html>", encoding="utf-8")
    monkeypatch.setattr(raip, "ROOT", tmp_path)

    called = {"n": 0}

    def _fake_run_script(_relpath: str, args=None):
        called["n"] += 1
        return False, 1

    monkeypatch.setattr(raip, "run_script", _fake_run_script)
    ok = raip.ensure_report_exists()
    assert ok is True
    assert called["n"] == 0


def test_default_aion_overlay_mirror_returns_path_when_aion_state_exists(tmp_path: Path, monkeypatch):
    root = tmp_path / "q"
    (tmp_path / "aion" / "state").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(raip, "ROOT", root)
    mirror = raip.default_aion_overlay_mirror()
    assert mirror.endswith("aion/state/q_signal_overlay.json")


def test_default_aion_overlay_mirror_empty_when_no_aion_state(tmp_path: Path, monkeypatch):
    root = tmp_path / "q"
    monkeypatch.setattr(raip, "ROOT", root)
    mirror = raip.default_aion_overlay_mirror()
    assert mirror == ""


def test_should_run_legacy_tune_when_missing(monkeypatch, tmp_path: Path):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(raip, "RUNS", runs)
    monkeypatch.delenv("Q_ENABLE_LEGACY_TUNE", raising=False)
    assert raip.should_run_legacy_tune() is True


def test_should_skip_legacy_tune_when_present(monkeypatch, tmp_path: Path):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "legacy_exposure.csv").write_text("1.0\n", encoding="utf-8")
    monkeypatch.setattr(raip, "RUNS", runs)
    monkeypatch.delenv("Q_ENABLE_LEGACY_TUNE", raising=False)
    assert raip.should_run_legacy_tune() is False


def test_should_force_legacy_tune_with_env(monkeypatch, tmp_path: Path):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "legacy_exposure.csv").write_text("1.0\n", encoding="utf-8")
    monkeypatch.setattr(raip, "RUNS", runs)
    monkeypatch.setenv("Q_ENABLE_LEGACY_TUNE", "1")
    assert raip.should_run_legacy_tune() is True


def test_apply_performance_defaults_sets_runtime_floor(monkeypatch):
    monkeypatch.delenv("Q_RUNTIME_TOTAL_FLOOR", raising=False)
    monkeypatch.setenv("Q_DEFAULT_RUNTIME_TOTAL_FLOOR", "0.18")
    monkeypatch.delenv("Q_DISABLE_GOVERNORS", raising=False)
    monkeypatch.delenv("Q_DEFAULT_DISABLE_GOVERNORS", raising=False)
    raip.apply_performance_defaults()
    assert os.environ.get("Q_RUNTIME_TOTAL_FLOOR") == "0.18"
    got = os.environ.get("Q_DISABLE_GOVERNORS", "")
    assert "global_governor" in got
    assert "heartbeat_scaler" in got


def test_apply_performance_defaults_merges_user_disables(monkeypatch):
    monkeypatch.setenv("Q_DISABLE_GOVERNORS", "quality_governor")
    monkeypatch.setenv("Q_DEFAULT_DISABLE_GOVERNORS", "global_governor")
    raip.apply_performance_defaults()
    got = os.environ.get("Q_DISABLE_GOVERNORS", "")
    assert "quality_governor" in got
    assert "global_governor" in got


def test_apply_performance_defaults_uses_selected_profile(monkeypatch, tmp_path: Path):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "runtime_profile_selected.json").write_text(
        '{"runtime_total_floor":0.16,"disable_governors":["x","y"]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(raip, "RUNS", runs)
    monkeypatch.delenv("Q_RUNTIME_TOTAL_FLOOR", raising=False)
    monkeypatch.delenv("Q_DEFAULT_RUNTIME_TOTAL_FLOOR", raising=False)
    monkeypatch.delenv("Q_DEFAULT_DISABLE_GOVERNORS", raising=False)
    monkeypatch.delenv("Q_DISABLE_GOVERNORS", raising=False)
    raip.apply_performance_defaults()
    assert os.environ.get("Q_RUNTIME_TOTAL_FLOOR") == "0.16"
    got = os.environ.get("Q_DISABLE_GOVERNORS", "")
    assert "x" in got and "y" in got


def test_apply_performance_defaults_prefers_active_profile(monkeypatch, tmp_path: Path):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "runtime_profile_selected.json").write_text(
        '{"runtime_total_floor":0.12,"disable_governors":["sel"]}',
        encoding="utf-8",
    )
    (runs / "runtime_profile_active.json").write_text(
        '{"runtime_total_floor":0.18,"disable_governors":["act"]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(raip, "RUNS", runs)
    monkeypatch.delenv("Q_RUNTIME_TOTAL_FLOOR", raising=False)
    monkeypatch.delenv("Q_DEFAULT_RUNTIME_TOTAL_FLOOR", raising=False)
    monkeypatch.delenv("Q_DEFAULT_DISABLE_GOVERNORS", raising=False)
    monkeypatch.delenv("Q_DISABLE_GOVERNORS", raising=False)
    raip.apply_performance_defaults()
    assert os.environ.get("Q_RUNTIME_TOTAL_FLOOR") == "0.18"
    got = os.environ.get("Q_DISABLE_GOVERNORS", "")
    assert "act" in got
    assert "sel" not in got
