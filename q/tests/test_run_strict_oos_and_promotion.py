import json
from pathlib import Path

import numpy as np

import tools.run_q_promotion_gate as pg
import tools.run_strict_oos_validation as so


def test_strict_oos_metrics_basic():
    r = np.array([0.01, -0.01, 0.02, -0.02], dtype=float)
    m = so._metrics(r)
    assert m["n"] == 4
    assert abs(m["hit_rate"] - 0.5) < 1e-12
    assert m["vol_daily"] > 0.0


def test_strict_oos_metrics_uses_sample_volatility_ddof1():
    r = np.array([0.01, -0.01], dtype=float)
    m = so._metrics(r)
    expected = float(np.std(r, ddof=1))
    assert abs(float(m["vol_daily"]) - expected) < 1e-9


def test_q_promotion_gate_pass(tmp_path: Path, monkeypatch):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics_oos_net": {
            "sharpe": 1.2,
            "hit_rate": 0.51,
            "max_drawdown": -0.04,
            "n": 300,
        }
    }
    (runs / "strict_oos_validation.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(pg, "ROOT", tmp_path)
    monkeypatch.setattr(pg, "RUNS", runs)
    rc = pg.main()
    assert rc == 0
    out = json.loads((runs / "q_promotion_gate.json").read_text(encoding="utf-8"))
    assert out["ok"] is True
    assert out["reasons"] == []


def test_q_promotion_gate_prefers_robust_when_available(tmp_path: Path, monkeypatch):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics_oos_net": {
            "sharpe": 1.4,
            "hit_rate": 0.52,
            "max_drawdown": -0.03,
            "n": 400,
        },
        "metrics_oos_robust": {
            "sharpe": 1.1,
            "hit_rate": 0.50,
            "max_drawdown": -0.04,
            "n": 300,
        },
    }
    (runs / "strict_oos_validation.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(pg, "ROOT", tmp_path)
    monkeypatch.setattr(pg, "RUNS", runs)
    rc = pg.main()
    assert rc == 0
    out = json.loads((runs / "q_promotion_gate.json").read_text(encoding="utf-8"))
    assert out["ok"] is True
    assert out["metric_source"] == "metrics_oos_robust"


def test_q_promotion_gate_fail(tmp_path: Path, monkeypatch):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics_oos_net": {
            "sharpe": 0.6,
            "hit_rate": 0.45,
            "max_drawdown": -0.20,
            "n": 100,
        }
    }
    (runs / "strict_oos_validation.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(pg, "ROOT", tmp_path)
    monkeypatch.setattr(pg, "RUNS", runs)
    rc = pg.main()
    assert rc == 0
    out = json.loads((runs / "q_promotion_gate.json").read_text(encoding="utf-8"))
    assert out["ok"] is False
    assert len(out["reasons"]) >= 1


def test_q_promotion_gate_cost_stress_required_pass(tmp_path: Path, monkeypatch):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics_oos_net": {
            "sharpe": 1.2,
            "hit_rate": 0.51,
            "max_drawdown": -0.04,
            "n": 300,
        }
    }
    stress = {
        "ok": True,
        "worst_case_robust": {"sharpe": 1.0, "hit_rate": 0.49, "max_drawdown": -0.05},
        "thresholds": {"min_robust_sharpe": 0.9},
        "reasons": [],
    }
    (runs / "strict_oos_validation.json").write_text(json.dumps(payload), encoding="utf-8")
    (runs / "cost_stress_validation.json").write_text(json.dumps(stress), encoding="utf-8")
    monkeypatch.setattr(pg, "ROOT", tmp_path)
    monkeypatch.setattr(pg, "RUNS", runs)
    monkeypatch.setenv("Q_PROMOTION_REQUIRE_COST_STRESS", "1")
    rc = pg.main()
    assert rc == 0
    out = json.loads((runs / "q_promotion_gate.json").read_text(encoding="utf-8"))
    assert out["ok"] is True


def test_q_promotion_gate_cost_stress_required_fail(tmp_path: Path, monkeypatch):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics_oos_net": {
            "sharpe": 1.2,
            "hit_rate": 0.51,
            "max_drawdown": -0.04,
            "n": 300,
        }
    }
    stress = {
        "ok": False,
        "worst_case_robust": {"sharpe": 0.6, "hit_rate": 0.45, "max_drawdown": -0.20},
        "reasons": ["cost_stress_robust_sharpe<0.90 (0.600)"],
    }
    (runs / "strict_oos_validation.json").write_text(json.dumps(payload), encoding="utf-8")
    (runs / "cost_stress_validation.json").write_text(json.dumps(stress), encoding="utf-8")
    monkeypatch.setattr(pg, "ROOT", tmp_path)
    monkeypatch.setattr(pg, "RUNS", runs)
    monkeypatch.setenv("Q_PROMOTION_REQUIRE_COST_STRESS", "1")
    rc = pg.main()
    assert rc == 0
    out = json.loads((runs / "q_promotion_gate.json").read_text(encoding="utf-8"))
    assert out["ok"] is False
    assert "cost_stress_fail" in out["reasons"]


def test_robust_oos_aggregate_shapes():
    ms = [
        {"sharpe": 1.0, "hit_rate": 0.49, "max_drawdown": -0.05, "n": 300},
        {"sharpe": 1.4, "hit_rate": 0.51, "max_drawdown": -0.03, "n": 280},
        {"sharpe": 0.8, "hit_rate": 0.47, "max_drawdown": -0.06, "n": 260},
    ]
    out = so._aggregate_robust(ms)
    assert out["num_splits"] == 3
    assert out["n"] == 260
    assert out["sharpe"] > 0.0
    assert out["hit_rate"] > 0.0


def test_strict_oos_latest_window_metric_present(tmp_path: Path, monkeypatch):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    r = np.linspace(-0.001, 0.002, 800, dtype=float)
    np.savetxt(runs / "daily_returns.csv", r, delimiter=",")
    monkeypatch.setattr(so, "ROOT", tmp_path)
    monkeypatch.setattr(so, "RUNS", runs)
    monkeypatch.setenv("Q_DISABLE_REPORT_CARDS", "1")
    monkeypatch.setenv("Q_STRICT_OOS_LATEST_HOLDOUT_DAYS", "200")
    monkeypatch.setenv("Q_STRICT_OOS_LATEST_HOLDOUT_MIN", "120")
    rc = so.main()
    assert rc == 0
    out = json.loads((runs / "strict_oos_validation.json").read_text(encoding="utf-8"))
    latest = out.get("metrics_oos_latest", {})
    assert int(latest.get("n", 0)) == 200
    assert "latest_holdout_days_used" in out


def test_q_promotion_gate_latest_holdout_required_fail(tmp_path: Path, monkeypatch):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics_oos_net": {"sharpe": 1.3, "hit_rate": 0.51, "max_drawdown": -0.04, "n": 300},
        "metrics_oos_latest": {"sharpe": 0.4, "hit_rate": 0.45, "max_drawdown": -0.20, "n": 80},
    }
    (runs / "strict_oos_validation.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(pg, "ROOT", tmp_path)
    monkeypatch.setattr(pg, "RUNS", runs)
    monkeypatch.setenv("Q_PROMOTION_REQUIRE_LATEST_HOLDOUT", "1")
    monkeypatch.setenv("Q_PROMOTION_MIN_LATEST_OOS_SHARPE", "0.9")
    monkeypatch.setenv("Q_PROMOTION_MIN_LATEST_OOS_HIT", "0.48")
    monkeypatch.setenv("Q_PROMOTION_MAX_LATEST_ABS_MDD", "0.12")
    monkeypatch.setenv("Q_PROMOTION_MIN_LATEST_OOS_SAMPLES", "126")
    rc = pg.main()
    assert rc == 0
    out = json.loads((runs / "q_promotion_gate.json").read_text(encoding="utf-8"))
    assert out["ok"] is False
    assert any("latest_oos" in str(x) for x in out.get("reasons", []))
