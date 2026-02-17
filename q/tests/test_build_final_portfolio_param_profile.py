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


def test_credit_leadlag_strength_uses_profile_value(monkeypatch):
    monkeypatch.delenv("Q_CREDIT_LEADLAG_STRENGTH", raising=False)
    monkeypatch.setattr(
        bfp,
        "_GOV_PARAM_PROFILE",
        {
            "parameters": {
                "credit_leadlag_strength": 0.44,
            }
        },
    )
    out = bfp._env_or_profile_float(
        "Q_CREDIT_LEADLAG_STRENGTH",
        "credit_leadlag_strength",
        0.35,
        0.0,
        2.0,
    )
    assert abs(float(out) - 0.44) < 1e-9


def test_microstructure_strength_uses_profile_value(monkeypatch):
    monkeypatch.delenv("Q_MICROSTRUCTURE_STRENGTH", raising=False)
    monkeypatch.setattr(
        bfp,
        "_GOV_PARAM_PROFILE",
        {
            "parameters": {
                "microstructure_strength": 0.27,
            }
        },
    )
    out = bfp._env_or_profile_float(
        "Q_MICROSTRUCTURE_STRENGTH",
        "microstructure_strength",
        0.20,
        0.0,
        2.0,
    )
    assert abs(float(out) - 0.27) < 1e-9


def test_calendar_event_strength_uses_profile_value(monkeypatch):
    monkeypatch.delenv("Q_CALENDAR_EVENT_STRENGTH", raising=False)
    monkeypatch.setattr(
        bfp,
        "_GOV_PARAM_PROFILE",
        {
            "parameters": {
                "calendar_event_strength": 0.18,
            }
        },
    )
    out = bfp._env_or_profile_float(
        "Q_CALENDAR_EVENT_STRENGTH",
        "calendar_event_strength",
        0.12,
        0.0,
        2.0,
    )
    assert abs(float(out) - 0.18) < 1e-9


def test_apply_governor_strength_blends_with_identity():
    vec = [0.7, 1.0, 1.2]
    no_effect = bfp._apply_governor_strength(vec, 0.0)
    full_effect = bfp._apply_governor_strength(vec, 1.0)
    stronger = bfp._apply_governor_strength(vec, 1.5)
    assert list(no_effect) == [1.0, 1.0, 1.0]
    assert list(full_effect) == [0.7, 1.0, 1.2]
    assert stronger[0] < full_effect[0]
    assert stronger[2] > full_effect[2]


def test_disabled_governors_merges_profile_and_env(monkeypatch):
    monkeypatch.setattr(
        bfp,
        "_GOV_PROFILE",
        {
            "disable_governors": ["dream_coherence", "symbolic_governor"],
        },
    )
    monkeypatch.setenv("Q_DISABLE_GOVERNORS", "uncertainty_sizing,symbolic_governor")
    out = bfp._disabled_governors()
    assert "dream_coherence" in out
    assert "symbolic_governor" in out
    assert "uncertainty_sizing" in out
