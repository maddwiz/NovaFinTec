import json

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
    assert 0.05 <= floor <= 0.15


def test_build_governor_profile_marks_ablation_harmful_governors():
    df = pd.DataFrame(
        {
            "global_governor": [0.75, 0.73, 0.76, 0.74],
            "heartbeat_scaler": [0.92, 0.93, 0.91, 0.94],
            "runtime_total_scalar": [0.30, 0.28, 0.29, 0.27],
        }
    )
    out = rgp.build_governor_profile(
        df,
        protected={"global_governor"},
        ablation_impacts={
            "heartbeat_scaler": {
                "delta_sharpe": 0.03,
                "delta_hit": -0.001,
                "delta_maxdd": 0.001,
            }
        },
        ablation_min_sharpe_gain=0.02,
        ablation_max_hit_drop=0.003,
        ablation_max_mdd_worsen=0.002,
    )
    disable = set(out["disable_governors"])
    assert "heartbeat_scaler" in disable
    rows = {str(r["name"]): r for r in out["governors"]}
    assert rows["heartbeat_scaler"]["status"] == "ablation_harmful"


def test_build_governor_profile_does_not_mark_ablation_harmful_when_risk_worsens():
    df = pd.DataFrame(
        {
            "heartbeat_scaler": [0.92, 0.93, 0.91, 0.94],
            "runtime_total_scalar": [0.30, 0.28, 0.29, 0.27],
        }
    )
    out = rgp.build_governor_profile(
        df,
        ablation_impacts={
            "heartbeat_scaler": {
                "delta_sharpe": 0.03,
                "delta_hit": -0.02,  # too much hit degradation
                "delta_maxdd": 0.0005,
            }
        },
        ablation_min_sharpe_gain=0.02,
        ablation_max_hit_drop=0.003,
        ablation_max_mdd_worsen=0.002,
    )
    disable = set(out["disable_governors"])
    assert "heartbeat_scaler" not in disable


def test_ablation_map_from_summary_parses_drop_rows(tmp_path):
    p = tmp_path / "governor_ablation.json"
    p.write_text(
        json.dumps(
            {
                "rows": [
                    {"scenario": "baseline"},
                    {
                        "scenario": "drop_heartbeat_scaler",
                        "delta_sharpe_vs_baseline": 0.02,
                        "delta_hit_vs_baseline": -0.001,
                        "delta_maxdd_vs_baseline": 0.0003,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    out = rgp._ablation_map_from_summary(p)
    assert "heartbeat_scaler" in out
    assert abs(float(out["heartbeat_scaler"]["delta_sharpe"]) - 0.02) < 1e-12
