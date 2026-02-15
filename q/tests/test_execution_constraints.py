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
