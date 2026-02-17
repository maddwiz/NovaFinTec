import numpy as np

import tools.run_execution_constraints as rec


def test_step_turnover_basic():
    w = np.array(
        [
            [0.0, 0.0],
            [1.0, -1.0],
            [2.0, -2.0],
        ],
        dtype=float,
    )
    t = rec._step_turnover(w)
    assert np.allclose(t, np.array([2.0, 2.0], dtype=float))


def test_apply_asset_delta_cap_limits_each_asset_step():
    w = np.array(
        [
            [0.0, 0.0],
            [0.30, -0.40],
            [0.10, -0.10],
        ],
        dtype=float,
    )
    out = rec._apply_asset_delta_cap(w, cap=0.15)
    d = np.diff(out, axis=0)
    assert float(np.max(np.abs(d))) <= 0.15 + 1e-12
    assert np.allclose(out[1], np.array([0.15, -0.15], dtype=float))
    assert np.allclose(out[2], np.array([0.10, -0.10], dtype=float))


def test_apply_asset_delta_cap_supports_per_step_caps():
    w = np.array(
        [
            [0.0, 0.0],
            [0.50, -0.50],
            [0.90, -0.90],
        ],
        dtype=float,
    )
    out = rec._apply_asset_delta_cap(w, cap=np.array([0.20, 0.05], dtype=float))
    d = np.diff(out, axis=0)
    assert np.allclose(np.abs(d[0]), np.array([0.20, 0.20], dtype=float))
    assert np.allclose(np.abs(d[1]), np.array([0.05, 0.05], dtype=float))


def test_apply_turnover_caps_step_limit():
    w = np.array(
        [
            [0.0, 0.0],
            [1.0, -1.0],
            [2.0, -2.0],
        ],
        dtype=float,
    )
    out, before, after = rec._apply_turnover_caps(w, max_step_turnover=1.0)

    assert np.allclose(before, np.array([2.0, 2.0], dtype=float))
    assert np.all(after <= 1.0 + 1e-12)
    assert np.allclose(after, np.array([1.0, 1.0], dtype=float))
    assert np.allclose(out[-1], np.array([1.0, -1.0], dtype=float))


def test_apply_turnover_caps_step_limit_array():
    w = np.array(
        [
            [0.0, 0.0],
            [1.0, -1.0],
            [2.0, -2.0],
        ],
        dtype=float,
    )
    out, _before, after = rec._apply_turnover_caps(w, max_step_turnover=np.array([2.0, 0.50], dtype=float))
    assert np.allclose(after, np.array([2.0, 0.5], dtype=float))
    assert np.allclose(out[-1], np.array([1.25, -1.25], dtype=float))


def test_apply_turnover_caps_rolling_budget():
    w = np.array(
        [
            [0.0, 0.0],
            [1.0, -1.0],
            [2.0, -2.0],
        ],
        dtype=float,
    )
    out, _before, after = rec._apply_turnover_caps(
        w,
        max_step_turnover=None,
        rolling_window=2,
        rolling_limit=1.5,
    )

    assert np.allclose(after, np.array([1.5, 0.0], dtype=float))
    assert np.allclose(out[1], np.array([0.75, -0.75], dtype=float))
    assert np.allclose(out[2], out[1])


def test_load_config_reads_turnover_env_fallbacks(monkeypatch, tmp_path):
    monkeypatch.setattr(rec, "CFG", tmp_path / "missing.json")
    monkeypatch.setenv("TURNOVER_MAX_STEP", "0.33")
    monkeypatch.setenv("TURNOVER_BUDGET_WINDOW", "7")
    monkeypatch.setenv("TURNOVER_BUDGET_LIMIT", "1.20")
    cfg = rec._load_config()

    assert cfg["max_step_turnover"] == 0.33
    assert cfg["rolling_turnover_window"] == 7
    assert cfg["rolling_turnover_limit"] == 1.2


def test_load_config_q_exec_env_overrides_file(monkeypatch, tmp_path):
    p = tmp_path / "execution_constraints.json"
    p.write_text(
        '{"allow_shorts": true, "max_abs_weight": 0.20, "max_step_turnover": 0.50, "renormalize_to_gross": true}'
    )
    monkeypatch.setattr(rec, "CFG", p)
    monkeypatch.setenv("Q_EXEC_ALLOW_SHORTS", "false")
    monkeypatch.setenv("Q_EXEC_MAX_ABS_WEIGHT", "0.11")
    monkeypatch.setenv("Q_EXEC_MAX_STEP_TURNOVER", "0.25")
    monkeypatch.setenv("Q_EXEC_RENORMALIZE_TO_GROSS", "0")

    cfg = rec._load_config()
    assert cfg["allow_shorts"] is False
    assert cfg["max_abs_weight"] == 0.11
    assert cfg["max_step_turnover"] == 0.25
    assert cfg["renormalize_to_gross"] is False


def test_apply_session_cap_scales_value_by_session():
    scales = {"regular": 1.0, "after_hours": 0.5, "closed": 0.0}
    assert rec._apply_session_cap(0.4, "regular", scales) == 0.4
    assert rec._apply_session_cap(0.4, "after_hours", scales) == 0.2
    assert rec._apply_session_cap(0.4, "closed", scales) == 0.0


def test_apply_session_cap_returns_none_for_missing_base():
    scales = {"regular": 1.0, "after_hours": 0.6}
    assert rec._apply_session_cap(None, "after_hours", scales) is None


def test_apply_symbol_caps_supports_side_specific_limits():
    w = np.array(
        [
            [0.30, -0.30],
            [0.10, -0.10],
        ],
        dtype=float,
    )
    idx = {"AAPL": 0, "TSLA": 1}
    out = rec._apply_symbol_caps(
        w,
        idx,
        symbol_caps={"AAPL": 0.25},
        symbol_long_caps={"AAPL": 0.12},
        symbol_short_caps={"TSLA": 0.08},
    )
    # AAPL symmetric cap first (0.25), then long cap (0.12)
    assert np.allclose(out[:, 0], np.array([0.12, 0.10], dtype=float))
    # TSLA short cap prevents values below -0.08
    assert np.allclose(out[:, 1], np.array([-0.08, -0.08], dtype=float))


def test_adaptive_risk_scale_tightens_on_fracture_alert_and_low_quality():
    cfg = {
        "adaptive_risk_enabled": True,
        "fracture_state_scales": {"calm": 1.0, "fracture_warn": 0.74, "fracture_alert": 0.56},
        "quality_scale_floor": 0.60,
        "quality_scale_ceiling": 1.00,
    }
    fracture = {"state": "fracture_alert", "latest_score": 0.90}
    quality = {"quality_score": 0.25}
    scale, detail = rec._adaptive_risk_scale(cfg, fracture, quality)
    assert scale < 0.50
    assert detail["fracture_state"] == "fracture_alert"
    assert detail["quality_score"] == 0.25


def test_adaptive_risk_scale_disabled_returns_one():
    scale, detail = rec._adaptive_risk_scale({"adaptive_risk_enabled": False}, {}, {})
    assert scale == 1.0
    assert detail["enabled"] is False


def test_build_dynamic_turnover_scale_blends_capacity_and_macro(monkeypatch, tmp_path):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    np.savetxt(runs / "capacity_impact_scalar.csv", np.array([1.0, 0.8, 0.6, 0.9]), delimiter=",")
    np.savetxt(runs / "macro_risk_scalar.csv", np.array([1.0, 0.7, 0.7, 0.95]), delimiter=",")
    monkeypatch.setattr(rec, "RUNS", runs)
    cfg = {
        "dynamic_turnover_enabled": True,
        "dynamic_turnover_floor": 0.50,
        "dynamic_turnover_capacity_weight": 0.7,
        "dynamic_turnover_macro_weight": 0.3,
        "dynamic_turnover_smooth_alpha": 1.0,
    }
    s, d = rec._build_dynamic_turnover_scale(cfg, rows=4)
    assert len(s) == 4
    assert d["enabled"] is True
    assert float(np.min(s)) >= 0.50
    assert s[2] < s[0]
