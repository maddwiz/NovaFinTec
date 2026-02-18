import json
from pathlib import Path

import numpy as np

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


def test_score_row_penalizes_complexity(monkeypatch):
    simple = {
        "robust_sharpe": 1.0,
        "robust_hit_rate": 0.50,
        "robust_max_drawdown": -0.04,
        "disable_governors": ["a", "b", "c", "d"],
        "search_flag_count": 4,
        "enable_asset_class_diversification": False,
        "macro_proxy_strength": 0.0,
        "capacity_impact_strength": 0.0,
        "uncertainty_macro_shock_blend": 0.0,
    }
    complex_row = {
        "robust_sharpe": 1.0,
        "robust_hit_rate": 0.50,
        "robust_max_drawdown": -0.04,
        "disable_governors": [],
        "search_flag_count": 4,
        "enable_asset_class_diversification": True,
        "macro_proxy_strength": 0.3,
        "capacity_impact_strength": 0.2,
        "uncertainty_macro_shock_blend": 0.2,
    }
    monkeypatch.setenv("Q_RUNTIME_SEARCH_COMPLEXITY_PENALTY", "0.2")
    monkeypatch.setenv("Q_RUNTIME_SEARCH_COMPLEXITY_STRENGTH_PENALTY", "0.2")
    assert float(rcs._score_row(simple)) > float(rcs._score_row(complex_row))


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


def test_profile_payload_carries_cost_fields():
    row = {
        "runtime_total_floor": 0.16,
        "disable_governors": ["x"],
        "robust_sharpe": 1.2,
        "robust_hit_rate": 0.5,
        "robust_max_drawdown": -0.03,
        "ann_cost_estimate": 0.011,
        "mean_turnover": 0.04,
        "mean_effective_cost_bps": 10.3,
        "score": 1.3,
        "promotion_ok": True,
        "cost_stress_ok": True,
        "health_ok": True,
    }
    out = rcs._profile_payload(row)
    assert abs(float(out["ann_cost_estimate"]) - 0.011) < 1e-12
    assert abs(float(out["mean_turnover"]) - 0.04) < 1e-12
    assert abs(float(out["mean_effective_cost_bps"]) - 10.3) < 1e-12


def test_profile_payload_carries_latest_oos_fields():
    row = {
        "runtime_total_floor": 0.16,
        "disable_governors": [],
        "latest_oos_sharpe": 1.2,
        "latest_oos_hit_rate": 0.49,
        "latest_oos_max_drawdown": -0.03,
        "latest_oos_n": 252,
    }
    out = rcs._profile_payload(row)
    assert abs(float(out["latest_oos_sharpe"]) - 1.2) < 1e-12
    assert abs(float(out["latest_oos_hit_rate"]) - 0.49) < 1e-12
    assert abs(float(out["latest_oos_max_drawdown"]) + 0.03) < 1e-12
    assert int(out["latest_oos_n"]) == 252


def test_score_row_penalizes_weak_latest_oos(monkeypatch):
    strong = {
        "robust_sharpe": 1.2,
        "robust_hit_rate": 0.50,
        "robust_max_drawdown": -0.04,
        "ann_cost_estimate": 0.01,
        "mean_turnover": 0.04,
        "latest_oos_sharpe": 1.1,
        "latest_oos_hit_rate": 0.50,
        "latest_oos_max_drawdown": -0.04,
        "latest_oos_n": 252,
    }
    weak = {
        **strong,
        "latest_oos_sharpe": 0.20,
        "latest_oos_hit_rate": 0.42,
        "latest_oos_max_drawdown": -0.20,
        "latest_oos_n": 80,
    }
    monkeypatch.setenv("Q_RUNTIME_SEARCH_LATEST_SHARPE_WEIGHT", "0.50")
    monkeypatch.setenv("Q_RUNTIME_SEARCH_LATEST_HIT_WEIGHT", "0.50")
    monkeypatch.setenv("Q_RUNTIME_SEARCH_LATEST_MDD_PENALTY", "2.0")
    monkeypatch.setenv("Q_RUNTIME_SEARCH_LATEST_N_PENALTY", "1.0")
    assert float(rcs._score_row(strong)) > float(rcs._score_row(weak))


def test_default_class_enable_grid_detects_diversified_universe(monkeypatch, tmp_path: Path):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "asset_names.csv").write_text("symbol\nSPY\nTLT\nGLD\nEURUSD\n", encoding="utf-8")
    monkeypatch.setattr(rcs, "ROOT", tmp_path)
    monkeypatch.setattr(rcs, "RUNS", runs)
    monkeypatch.delenv("Q_RUNTIME_SEARCH_CLASS_ENABLES", raising=False)
    monkeypatch.setenv("Q_RUNTIME_SEARCH_MIN_CLASSES_FOR_DIVERSIFICATION", "3")
    assert rcs._default_class_enable_grid() == [0, 1]


def test_default_class_enable_grid_respects_env_override(monkeypatch, tmp_path: Path):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rcs, "ROOT", tmp_path)
    monkeypatch.setattr(rcs, "RUNS", runs)
    monkeypatch.setenv("Q_RUNTIME_SEARCH_CLASS_ENABLES", "1")
    assert rcs._default_class_enable_grid() == [1]


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


def test_canary_rejects_large_cost_worsening(monkeypatch):
    monkeypatch.setenv("Q_RUNTIME_CANARY_MIN_SHARPE_DELTA", "0.0")
    monkeypatch.setenv("Q_RUNTIME_CANARY_MIN_SCORE_DELTA", "0.0")
    monkeypatch.setenv("Q_RUNTIME_CANARY_MAX_ANN_COST_WORSEN", "0.001")
    monkeypatch.setenv("Q_RUNTIME_CANARY_MAX_TURNOVER_WORSEN", "0.02")
    stable = {
        "robust_sharpe": 1.40,
        "robust_hit_rate": 0.49,
        "robust_max_drawdown": -0.04,
        "score": 1.41,
        "ann_cost_estimate": 0.010,
        "mean_turnover": 0.05,
    }
    candidate = {
        "robust_sharpe": 1.45,
        "robust_hit_rate": 0.49,
        "robust_max_drawdown": -0.04,
        "score": 1.46,
        "ann_cost_estimate": 0.015,  # +0.005 > max 0.001
        "mean_turnover": 0.055,
        "promotion_ok": True,
        "cost_stress_ok": True,
        "health_ok": True,
    }
    ok, reasons = rcs._canary_qualifies(stable, candidate)
    assert ok is False
    assert any("ann_cost_worsen" in str(r) for r in reasons)


def test_canary_ignores_cost_worsen_when_stable_missing_cost_fields(monkeypatch):
    monkeypatch.setenv("Q_RUNTIME_CANARY_MIN_SHARPE_DELTA", "0.01")
    monkeypatch.setenv("Q_RUNTIME_CANARY_MIN_SCORE_DELTA", "0.01")
    stable = {
        "robust_sharpe": 1.50,
        "robust_hit_rate": 0.49,
        "robust_max_drawdown": -0.04,
        "score": 1.51,
        # intentionally missing ann_cost_estimate / mean_turnover
    }
    candidate = {
        "robust_sharpe": 1.55,
        "robust_hit_rate": 0.49,
        "robust_max_drawdown": -0.04,
        "score": 1.56,
        "ann_cost_estimate": 0.02,
        "mean_turnover": 0.08,
        "promotion_ok": True,
        "cost_stress_ok": True,
        "health_ok": True,
    }
    ok, reasons = rcs._canary_qualifies(stable, candidate)
    assert ok is True
    assert reasons == []


def test_canary_rejects_latest_oos_degradation(monkeypatch):
    monkeypatch.setenv("Q_RUNTIME_CANARY_MIN_SHARPE_DELTA", "0.0")
    monkeypatch.setenv("Q_RUNTIME_CANARY_MIN_SCORE_DELTA", "0.0")
    monkeypatch.setenv("Q_RUNTIME_CANARY_MAX_LATEST_SHARPE_DROP", "0.03")
    monkeypatch.setenv("Q_RUNTIME_CANARY_MAX_LATEST_HIT_DROP", "0.005")
    monkeypatch.setenv("Q_RUNTIME_CANARY_MAX_LATEST_ABS_MDD_WORSEN", "0.01")
    monkeypatch.setenv("Q_RUNTIME_CANARY_LATEST_MIN_N", "126")
    stable = {
        "robust_sharpe": 1.30,
        "robust_hit_rate": 0.50,
        "robust_max_drawdown": -0.040,
        "score": 1.30,
        "latest_oos_sharpe": 1.25,
        "latest_oos_hit_rate": 0.51,
        "latest_oos_max_drawdown": -0.03,
        "latest_oos_n": 252,
    }
    candidate = {
        "robust_sharpe": 1.36,
        "robust_hit_rate": 0.50,
        "robust_max_drawdown": -0.040,
        "score": 1.37,
        "latest_oos_sharpe": 1.15,
        "latest_oos_hit_rate": 0.49,
        "latest_oos_max_drawdown": -0.05,
        "latest_oos_n": 252,
        "promotion_ok": True,
        "cost_stress_ok": True,
        "health_ok": True,
    }
    ok, reasons = rcs._canary_qualifies(stable, candidate)
    assert ok is False
    assert any("latest_sharpe_drop" in str(r) for r in reasons)
    assert any("latest_mdd_worsen" in str(r) for r in reasons)


def test_canary_skips_latest_checks_when_latest_missing(monkeypatch):
    monkeypatch.setenv("Q_RUNTIME_CANARY_MIN_SHARPE_DELTA", "0.0")
    monkeypatch.setenv("Q_RUNTIME_CANARY_MIN_SCORE_DELTA", "0.0")
    monkeypatch.setenv("Q_RUNTIME_CANARY_MAX_LATEST_SHARPE_DROP", "0.01")
    stable = {
        "robust_sharpe": 1.20,
        "robust_hit_rate": 0.50,
        "robust_max_drawdown": -0.040,
        "score": 1.20,
    }
    candidate = {
        "robust_sharpe": 1.25,
        "robust_hit_rate": 0.50,
        "robust_max_drawdown": -0.040,
        "score": 1.25,
        "latest_oos_sharpe": 0.10,
        "latest_oos_hit_rate": 0.40,
        "latest_oos_max_drawdown": -0.20,
        "latest_oos_n": 252,
        "promotion_ok": True,
        "cost_stress_ok": True,
        "health_ok": True,
    }
    ok, reasons = rcs._canary_qualifies(stable, candidate)
    assert ok is True
    assert reasons == []


def test_runtime_combo_search_writes_relaxed_candidate_when_no_promotable(monkeypatch, tmp_path: Path):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rcs, "ROOT", tmp_path)
    monkeypatch.setattr(rcs, "RUNS", runs)
    monkeypatch.setattr(rcs, "_base_runtime_env", lambda: {})
    monkeypatch.setenv("Q_RUNTIME_SEARCH_FLOORS", "0.18")
    monkeypatch.setenv("Q_RUNTIME_SEARCH_FLAGS", "a")

    stable = {
        "runtime_total_floor": 0.14,
        "disable_governors": [],
        "robust_sharpe": 1.20,
        "robust_hit_rate": 0.50,
        "robust_max_drawdown": -0.04,
        "score": 1.20,
        "promotion_ok": True,
        "cost_stress_ok": True,
        "health_ok": True,
    }
    (runs / "runtime_profile_stable.json").write_text(json.dumps(stable), encoding="utf-8")
    (runs / "runtime_profile_active.json").write_text(json.dumps(stable), encoding="utf-8")

    def _fake_eval(_env):
        return {
            "robust_sharpe": 1.30,
            "robust_hit_rate": 0.50,
            "robust_max_drawdown": -0.04,
            "promotion_ok": False,
            "cost_stress_ok": True,
            "health_ok": True,
            "health_alerts_hard": 0,
            "rc": [{"step": "x", "code": 0}],
        }

    monkeypatch.setattr(rcs, "_eval_combo", _fake_eval)
    assert rcs.main() == 0

    out = json.loads((runs / "runtime_combo_search.json").read_text(encoding="utf-8"))
    assert int(out["rows_valid"]) == 0
    assert int(out["rows_relaxed_valid"]) > 0
    assert out["selected"] is None
    assert isinstance(out["selected_relaxed"], dict)
    assert (runs / "runtime_profile_selected_relaxed.json").exists()
    status = json.loads((runs / "runtime_profile_promotion_status.json").read_text(encoding="utf-8"))
    assert status["action"] == "no_promotable_profile"


def test_governor_train_validation_split_respects_holdout_min():
    train, val = rcs._governor_train_validation_split(total_rows=800, train_frac=0.75, holdout_min=252)
    assert train == 548
    assert val == 252

    train2, val2 = rcs._governor_train_validation_split(total_rows=2000, train_frac=0.75, holdout_min=252)
    assert train2 == 1500
    assert val2 == 500


def test_eval_combo_scores_using_governor_validation(monkeypatch, tmp_path: Path):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rcs, "ROOT", tmp_path)
    monkeypatch.setattr(rcs, "RUNS", runs)
    monkeypatch.setattr(rcs, "_run", lambda *_args, **_kwargs: 0)

    def _fake_load_json(path: Path):
        name = path.name
        if name == "strict_oos_validation.json":
            return {
                "metrics_oos_robust": {"sharpe": 9.0, "hit_rate": 0.95, "max_drawdown": -0.50},
                "metrics_oos_latest": {"sharpe": 1.0, "hit_rate": 0.5, "max_drawdown": -0.1, "n": 252},
            }
        if name in {"q_promotion_gate.json", "cost_stress_validation.json", "health_alerts.json"}:
            return {"ok": True, "alerts_hard": 0}
        return {}

    monkeypatch.setattr(rcs, "_load_json", _fake_load_json)
    r = np.r_[np.full(300, 0.01), np.full(100, -0.01)]
    monkeypatch.setattr(rcs, "_load_daily_returns_for_governor_eval", lambda: r)
    monkeypatch.setenv("Q_RUNTIME_SEARCH_GOVERNOR_TRAIN_FRAC", "0.75")
    monkeypatch.setenv("Q_RUNTIME_SEARCH_GOVERNOR_HOLDOUT_MIN", "80")

    out = rcs._eval_combo({})
    assert int(out["governor_train_rows"]) == 300
    assert int(out["governor_validation_rows"]) == 100
    # Validation slice is negative-only, so hit rate should be near zero and
    # robust metrics must come from this validation segment, not strict robust.
    assert float(out["governor_validation_hit_rate"]) < 0.05
    assert float(out["robust_hit_rate"]) == float(out["governor_validation_hit_rate"])
    assert float(out["strict_robust_sharpe"]) == 9.0
    assert float(out["robust_sharpe"]) != float(out["strict_robust_sharpe"])


def test_eval_combo_marks_governor_validation_not_ok_when_below_min_rows(monkeypatch, tmp_path: Path):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rcs, "ROOT", tmp_path)
    monkeypatch.setattr(rcs, "RUNS", runs)
    monkeypatch.setattr(rcs, "_run", lambda *_args, **_kwargs: 0)

    def _fake_load_json(path: Path):
        name = path.name
        if name == "strict_oos_validation.json":
            return {"metrics_oos_robust": {"sharpe": 1.2, "hit_rate": 0.5, "max_drawdown": -0.05}}
        if name in {"q_promotion_gate.json", "cost_stress_validation.json", "health_alerts.json"}:
            return {"ok": True, "alerts_hard": 0}
        return {}

    monkeypatch.setattr(rcs, "_load_json", _fake_load_json)
    # Split -> train=300, val=100 with frac=0.75 and holdout_min=80.
    r = np.r_[np.full(300, 0.01), np.full(100, -0.01)]
    monkeypatch.setattr(rcs, "_load_daily_returns_for_governor_eval", lambda: r)
    monkeypatch.setenv("Q_RUNTIME_SEARCH_GOVERNOR_TRAIN_FRAC", "0.75")
    monkeypatch.setenv("Q_RUNTIME_SEARCH_GOVERNOR_HOLDOUT_MIN", "80")
    monkeypatch.setenv("Q_RUNTIME_SEARCH_GOVERNOR_VALIDATION_MIN_ROWS", "120")

    out = rcs._eval_combo({})
    assert int(out["governor_validation_rows"]) == 100
    assert int(out["governor_validation_min_rows"]) == 120
    assert bool(out["governor_validation_ok"]) is False


def test_runtime_combo_search_filters_candidates_by_governor_validation_ok(monkeypatch, tmp_path: Path):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rcs, "ROOT", tmp_path)
    monkeypatch.setattr(rcs, "RUNS", runs)
    monkeypatch.setattr(rcs, "_base_runtime_env", lambda: {})
    monkeypatch.setenv("Q_RUNTIME_SEARCH_FLOORS", "0.18,0.22")
    monkeypatch.setenv("Q_RUNTIME_SEARCH_FLAGS", "a")
    monkeypatch.setenv("Q_RUNTIME_SEARCH_REQUIRE_GOVERNOR_VALIDATION", "1")

    def _fake_eval(env):
        floor = float(env.get("Q_RUNTIME_TOTAL_FLOOR", "0.18"))
        disabled = {x for x in env.get("Q_DISABLE_GOVERNORS", "").split(",") if x}
        # Highest raw score candidate fails governor-validation quality gate.
        if floor == 0.22 and disabled == set():
            sh = 2.0
            gv_ok = False
        elif floor == 0.18 and disabled == set():
            sh = 1.2
            gv_ok = True
        else:
            sh = 1.0
            gv_ok = True
        return {
            "robust_sharpe": sh,
            "robust_hit_rate": 0.50,
            "robust_max_drawdown": -0.04,
            "governor_validation_ok": gv_ok,
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
    # The non-compliant (gv_ok=False) high-sharpe row must be excluded.
    assert abs(float(sel["runtime_total_floor"]) - 0.18) < 1e-9
