import tools.build_final_portfolio as bfp


def test_runtime_floor_uses_param_profile_when_env_missing(monkeypatch):
    monkeypatch.delenv("Q_RUNTIME_TOTAL_FLOOR", raising=False)
    monkeypatch.setattr(
        bfp,
        "_GOV_PARAM_PROFILE",
        {
            "parameters": {
                "runtime_total_floor": 0.07,
            }
        },
    )
    monkeypatch.setattr(bfp, "_GOV_PROFILE", {"runtime_total_floor": 0.25})
    out = bfp._runtime_total_floor_default()
    assert abs(float(out) - 0.07) < 1e-9


def test_runtime_floor_env_overrides_param_profile(monkeypatch):
    monkeypatch.setenv("Q_RUNTIME_TOTAL_FLOOR", "0.13")
    monkeypatch.setattr(
        bfp,
        "_GOV_PARAM_PROFILE",
        {
            "parameters": {
                "runtime_total_floor": 0.07,
            }
        },
    )
    monkeypatch.setattr(bfp, "_GOV_PROFILE", {"runtime_total_floor": 0.25})
    out = bfp._runtime_total_floor_default()
    assert abs(float(out) - 0.13) < 1e-9


def test_env_or_profile_helpers_parse_profile_values(monkeypatch):
    monkeypatch.delenv("Q_TEST_FLOAT", raising=False)
    monkeypatch.delenv("Q_TEST_INT", raising=False)
    monkeypatch.delenv("Q_TEST_BOOL", raising=False)
    monkeypatch.setattr(
        bfp,
        "_GOV_PARAM_PROFILE",
        {
            "parameters": {
                "test_float": 0.42,
                "test_int": 8,
                "test_bool": True,
            }
        },
    )
    fv = bfp._env_or_profile_float("Q_TEST_FLOAT", "test_float", 0.1, 0.0, 1.0)
    iv = bfp._env_or_profile_int("Q_TEST_INT", "test_int", 3, 1, 20)
    bv = bfp._env_or_profile_bool("Q_TEST_BOOL", "test_bool", False)
    assert abs(float(fv) - 0.42) < 1e-9
    assert int(iv) == 8
    assert bv is True


def test_apply_governor_strength_blends_with_identity():
    vec = [0.7, 1.0, 1.2]
    no_effect = bfp._apply_governor_strength(vec, 0.0)
    full_effect = bfp._apply_governor_strength(vec, 1.0)
    stronger = bfp._apply_governor_strength(vec, 1.5)
    assert list(no_effect) == [1.0, 1.0, 1.0]
    assert list(full_effect) == [0.7, 1.0, 1.2]
    assert stronger[0] < full_effect[0]
    assert stronger[2] > full_effect[2]
