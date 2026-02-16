import pandas as pd

import tools.run_governor_profile as rgp


def test_build_governor_profile_handles_missing_trace():
    out = rgp.build_governor_profile(pd.DataFrame())
    assert out["ok"] is False
    assert out["disable_governors"] == []


def test_build_governor_profile_marks_neutral_and_protected_governors():
    df = pd.DataFrame(
        {
            "global_governor": [0.72, 0.74, 0.76, 0.71],
            "quality_governor": [0.65, 0.66, 0.64, 0.67],
            "meta_mix_reliability": [1.0, 1.0, 1.0, 1.0],  # neutral
            "symbolic_governor": [1.000, 1.000, 1.001, 0.999],  # effectively neutral
            "heartbeat_scaler": [0.85, 0.90, 0.88, 0.92],  # active
            "runtime_floor": [1.0, 1.1, 1.0, 1.0],
            "runtime_total_scalar": [0.30, 0.26, 0.28, 0.25],
        }
    )
    out = rgp.build_governor_profile(
        df,
        protected={"global_governor", "quality_governor", "runtime_floor"},
    )
    assert out["ok"] is True
    disable = set(out["disable_governors"])
    assert "meta_mix_reliability" in disable
    assert "symbolic_governor" in disable
    assert "global_governor" not in disable
    assert "quality_governor" not in disable
    assert "runtime_floor" not in disable
    floor = float(out["runtime_total_floor"])
    assert 0.20 <= floor <= 0.35
    assert floor >= 0.22
