import json
from pathlib import Path

import numpy as np

import tools.run_cost_stress_validation as csval


def test_cost_stress_validation_writes_outputs(tmp_path: Path, monkeypatch):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    t = 320
    n = 3
    rng = np.random.default_rng(7)
    a = rng.normal(0.0004, 0.01, size=(t, n))
    w = np.full((t, n), 1.0 / n, dtype=float)
    np.savetxt(runs / "asset_returns.csv", a, delimiter=",")
    np.savetxt(runs / "portfolio_weights_final.csv", w, delimiter=",")

    monkeypatch.setattr(csval, "ROOT", tmp_path)
    monkeypatch.setattr(csval, "RUNS", runs)
    monkeypatch.setenv("Q_COST_STRESS_BPS_LIST", "10,15")
    monkeypatch.setenv("Q_STRICT_OOS_MIN_TRAIN", "200")
    monkeypatch.setenv("Q_STRICT_OOS_MIN_TEST", "80")

    rc = csval.main()
    assert rc == 0
    out = json.loads((runs / "cost_stress_validation.json").read_text(encoding="utf-8"))
    assert "ok" in out
    assert "scenarios" in out
    assert len(out["scenarios"]) == 2
    assert (runs / "cost_stress_validation_rows.csv").exists()
