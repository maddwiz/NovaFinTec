from pathlib import Path

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
