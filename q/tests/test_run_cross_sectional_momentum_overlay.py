import json
from pathlib import Path

import numpy as np

import tools.run_cross_sectional_momentum_overlay as csm


def test_run_cross_sectional_momentum_overlay_writes_outputs(tmp_path: Path, monkeypatch):
    root = tmp_path
    runs = root / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)

    t, n = 260, 36
    rng = np.random.default_rng(7)
    x = np.linspace(0.0, 14.0, t)
    loadings = np.linspace(-1.2, 1.3, n)
    fac = (0.0012 * np.sin(x) + 0.0009 * np.cos(0.5 * x))[:, None]
    arr = fac * loadings[None, :] + 0.0032 * rng.standard_normal((t, n))
    arr[-55:, : n // 2] -= 0.0020
    arr[-55:, n // 2 :] += 0.0018
    np.savetxt(runs / "asset_returns.csv", arr, delimiter=",")

    monkeypatch.setattr(csm, "ROOT", root)
    monkeypatch.setattr(csm, "RUNS", runs)
    monkeypatch.setenv("Q_DISABLE_REPORT_CARDS", "1")
    monkeypatch.setenv("Q_CSM_MIN_ASSETS", "12")

    rc = csm.main()
    assert rc == 0

    sig = np.loadtxt(runs / "cross_sectional_momentum_signal.csv", delimiter=",").ravel()
    ov = np.loadtxt(runs / "cross_sectional_momentum_overlay.csv", delimiter=",").ravel()
    info = json.loads((runs / "cross_sectional_momentum_info.json").read_text(encoding="utf-8"))

    assert len(sig) == t
    assert len(ov) == t
    assert float(np.min(sig)) >= -1.0 - 1e-9
    assert float(np.max(sig)) <= 1.0 + 1e-9
    assert float(np.min(ov)) >= float(info["params"]["floor"]) - 1e-9
    assert float(np.max(ov)) <= float(info["params"]["ceil"]) + 1e-9
    assert float(np.std(ov)) > 0.0
    assert info["ok"] is True


def test_run_cross_sectional_momentum_overlay_insufficient_shape_is_neutral(tmp_path: Path, monkeypatch):
    root = tmp_path
    runs = root / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    np.savetxt(runs / "asset_returns.csv", np.zeros((70, 4), dtype=float), delimiter=",")

    monkeypatch.setattr(csm, "ROOT", root)
    monkeypatch.setattr(csm, "RUNS", runs)
    monkeypatch.setenv("Q_DISABLE_REPORT_CARDS", "1")
    monkeypatch.setenv("Q_CSM_MIN_ASSETS", "10")

    rc = csm.main()
    assert rc == 0

    sig = np.loadtxt(runs / "cross_sectional_momentum_signal.csv", delimiter=",").ravel()
    ov = np.loadtxt(runs / "cross_sectional_momentum_overlay.csv", delimiter=",").ravel()
    info = json.loads((runs / "cross_sectional_momentum_info.json").read_text(encoding="utf-8"))

    assert len(sig) == 70
    assert len(ov) == 70
    assert float(np.max(np.abs(sig))) == 0.0
    assert float(np.min(ov)) == 1.0
    assert float(np.max(ov)) == 1.0
    assert info["ok"] is False
