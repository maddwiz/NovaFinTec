import json
from pathlib import Path

import tools.run_runtime_combo_search as rcs


def test_runtime_combo_search_selects_best_valid(monkeypatch, tmp_path: Path):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rcs, "ROOT", tmp_path)
    monkeypatch.setattr(rcs, "RUNS", runs)

    monkeypatch.setenv("Q_RUNTIME_SEARCH_FLOORS", "0.18,0.22")
    monkeypatch.setenv("Q_RUNTIME_SEARCH_FLAGS", "a,b")

    monkeypatch.setattr(rcs, "_base_runtime_env", lambda: {})

    def _fake_eval(env):
        floor = float(env.get("Q_RUNTIME_TOTAL_FLOOR", "0.18"))
        disabled = {x for x in env.get("Q_DISABLE_GOVERNORS", "").split(",") if x}
        # Best valid case by score: floor 0.18 with only b disabled.
        sharpe = 1.0
        hit = 0.50
        mdd = -0.04
        if floor == 0.18 and disabled == {"b"}:
            sharpe, hit, mdd = 1.6, 0.50, -0.04
        elif floor == 0.22 and disabled == {"a"}:
            sharpe, hit, mdd = 1.55, 0.50, -0.08
        return {
            "robust_sharpe": sharpe,
            "robust_hit_rate": hit,
            "robust_max_drawdown": mdd,
            "promotion_ok": True,
            "cost_stress_ok": True,
            "health_ok": True,
            "health_alerts_hard": 0,
            "rc": [{"step": "x", "code": 0}],
        }

    monkeypatch.setattr(rcs, "_eval_combo", _fake_eval)
    rc = rcs.main()
    assert rc == 0

    sel = json.loads((runs / "runtime_profile_selected.json").read_text(encoding="utf-8"))
    assert abs(float(sel["runtime_total_floor"]) - 0.18) < 1e-9
    assert sel["disable_governors"] == ["b"]
    assert (runs / "runtime_combo_search_progress.json").exists()
    assert (runs / "runtime_profile_stable.json").exists()
    assert (runs / "runtime_profile_active.json").exists()


def test_runtime_combo_search_canary_promotes_after_required_passes(monkeypatch, tmp_path: Path):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rcs, "ROOT", tmp_path)
    monkeypatch.setattr(rcs, "RUNS", runs)
    monkeypatch.setattr(rcs, "_base_runtime_env", lambda: {})
    monkeypatch.setenv("Q_RUNTIME_CANARY_REQUIRED_PASSES", "2")
    monkeypatch.setenv("Q_RUNTIME_CANARY_ROLLBACK_FAILS", "2")
    monkeypatch.setenv("Q_RUNTIME_SEARCH_FLOORS", "0.18")
    monkeypatch.setenv("Q_RUNTIME_SEARCH_FLAGS", "a")

    call = {"n": 0}

    def _fake_eval(env):
        call["n"] += 1
        dis = str(env.get("Q_DISABLE_GOVERNORS", "")).strip()
        # First script run picks baseline stable.
        if call["n"] <= 2:
            return {
                "robust_sharpe": 1.2 if dis == "" else 1.1,
                "robust_hit_rate": 0.50,
                "robust_max_drawdown": -0.04,
                "promotion_ok": True,
                "cost_stress_ok": True,
                "health_ok": True,
                "health_alerts_hard": 0,
                "rc": [{"step": "x", "code": 0}],
            }
        # Second+ script run candidate improves enough to be canary/promoted.
        return {
            "robust_sharpe": 1.2 if dis == "" else 1.4,
            "robust_hit_rate": 0.50,
            "robust_max_drawdown": -0.04,
            "promotion_ok": True,
            "cost_stress_ok": True,
            "health_ok": True,
            "health_alerts_hard": 0,
            "rc": [{"step": "x", "code": 0}],
        }

    monkeypatch.setattr(rcs, "_eval_combo", _fake_eval)

    # Run 1: bootstrap stable.
    assert rcs.main() == 0
    st = json.loads((runs / "runtime_profile_stable.json").read_text(encoding="utf-8"))
    assert abs(float(st["robust_sharpe"]) - 1.2) < 1e-9

    # Run 2: canary armed (not yet promoted).
    assert rcs.main() == 0
    status2 = json.loads((runs / "runtime_profile_promotion_status.json").read_text(encoding="utf-8"))
    assert status2["action"] in {"canary_armed", "promoted"}

    # Run 3: should promote due required_passes=2.
    assert rcs.main() == 0
    st3 = json.loads((runs / "runtime_profile_stable.json").read_text(encoding="utf-8"))
    assert abs(float(st3["robust_sharpe"]) - 1.4) < 1e-9


def test_score_row_penalizes_cost_and_turnover(monkeypatch):
    base = {
        "robust_sharpe": 1.0,
        "robust_hit_rate": 0.50,
        "robust_max_drawdown": -0.04,
        "ann_cost_estimate": 0.01,
        "mean_turnover": 0.04,
    }
    stressed = {
        "robust_sharpe": 1.0,
        "robust_hit_rate": 0.50,
        "robust_max_drawdown": -0.04,
        "ann_cost_estimate": 0.08,
        "mean_turnover": 0.20,
    }
    monkeypatch.setenv("Q_RUNTIME_SEARCH_COST_REF_ANNUAL", "0.02")
    monkeypatch.setenv("Q_RUNTIME_SEARCH_COST_PENALTY", "4.0")
    monkeypatch.setenv("Q_RUNTIME_SEARCH_TURNOVER_REF_DAILY", "0.06")
    monkeypatch.setenv("Q_RUNTIME_SEARCH_TURNOVER_PENALTY", "2.0")
    assert float(rcs._score_row(base)) > float(rcs._score_row(stressed))


def test_base_runtime_env_uses_friction_calibration(monkeypatch, tmp_path: Path):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rcs, "RUNS", runs)
    (runs / "friction_calibration.json").write_text(
        json.dumps(
            {
                "ok": True,
                "recommendation": {
                    "recommended_cost_base_bps": 9.1,
                    "recommended_cost_vol_scaled_bps": 1.7,
                },
            }
        ),
        encoding="utf-8",
    )
    env = rcs._base_runtime_env()
    assert env.get("Q_COST_BASE_BPS") == "9.1"
    assert env.get("Q_COST_VOL_SCALED_BPS") == "1.7"


def test_canary_qualifies_on_score_delta_when_sharpe_delta_small(monkeypatch):
    monkeypatch.setenv("Q_RUNTIME_CANARY_MIN_SHARPE_DELTA", "0.02")
    monkeypatch.setenv("Q_RUNTIME_CANARY_MIN_SCORE_DELTA", "0.01")
    stable = {
        "robust_sharpe": 1.50,
        "robust_hit_rate": 0.49,
        "robust_max_drawdown": -0.040,
        "score": 1.55,
    }
    candidate = {
        "robust_sharpe": 1.515,  # below sharpe delta gate
        "robust_hit_rate": 0.49,
        "robust_max_drawdown": -0.040,
        "score": 1.565,  # meets score delta gate
        "promotion_ok": True,
        "cost_stress_ok": True,
        "health_ok": True,
    }
    ok, reasons = rcs._canary_qualifies(stable, candidate)
    assert ok is True
    assert reasons == []


def test_canary_rejects_when_both_sharpe_and_score_deltas_low(monkeypatch):
    monkeypatch.setenv("Q_RUNTIME_CANARY_MIN_SHARPE_DELTA", "0.02")
    monkeypatch.setenv("Q_RUNTIME_CANARY_MIN_SCORE_DELTA", "0.01")
    stable = {
        "robust_sharpe": 1.50,
        "robust_hit_rate": 0.49,
        "robust_max_drawdown": -0.040,
        "score": 1.55,
    }
    candidate = {
        "robust_sharpe": 1.51,
        "robust_hit_rate": 0.49,
        "robust_max_drawdown": -0.040,
        "score": 1.556,
        "promotion_ok": True,
        "cost_stress_ok": True,
        "health_ok": True,
    }
    ok, reasons = rcs._canary_qualifies(stable, candidate)
    assert ok is False
    assert any("delta_below_thresholds" in str(r) for r in reasons)
