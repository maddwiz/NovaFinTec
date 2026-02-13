import json

import numpy as np
import pandas as pd

import tools.run_cross_hive as rch


def test_novaspine_hive_multipliers_default_when_missing(tmp_path):
    old_runs = rch.RUNS
    try:
        rch.RUNS = tmp_path
        out = rch.novaspine_hive_multipliers(["EQ", "FX"])
    finally:
        rch.RUNS = old_runs

    assert out == {"EQ": 1.0, "FX": 1.0}


def test_novaspine_hive_multipliers_reads_and_clips(tmp_path):
    payload = {
        "per_hive": {
            "EQ": {"boost": 1.5},
            "FX": {"boost": 0.6},
            "RATES": {"boost": 1.03},
        }
    }
    (tmp_path / "novaspine_hive_feedback.json").write_text(json.dumps(payload), encoding="utf-8")

    old_runs = rch.RUNS
    try:
        rch.RUNS = tmp_path
        out = rch.novaspine_hive_multipliers(["EQ", "FX", "RATES", "COMMOD"])
    finally:
        rch.RUNS = old_runs

    assert out["EQ"] == 1.2
    assert out["FX"] == 0.8
    assert out["RATES"] == 1.03
    assert out["COMMOD"] == 1.0


def test_adaptive_arb_schedules_respond_to_disagreement():
    idx = pd.date_range("2025-01-01", periods=3, freq="D")
    stab = pd.DataFrame(
        {
            "EQ": [1.0, 0.5, 0.1],
            "FX": [1.0, 0.5, 0.1],
        },
        index=idx,
    )
    alpha_t, inertia_t, diag = rch.adaptive_arb_schedules(2.2, 0.8, stab)
    assert len(alpha_t) == 3
    assert len(inertia_t) == 3
    assert alpha_t[0] > alpha_t[-1]
    assert inertia_t[0] < inertia_t[-1]
    assert diag["alpha_min"] <= diag["alpha_max"]


def test_dynamic_quality_multipliers_reward_stronger_hive(tmp_path):
    idx = pd.date_range("2025-01-01", periods=140, freq="D")
    strong = pd.Series(0.003 + 0.001 * np.sin(np.linspace(0, 8, len(idx))), index=idx)
    weak = pd.Series(-0.001 + 0.001 * np.cos(np.linspace(0, 8, len(idx))), index=idx)

    rows = []
    for d, a, b in zip(idx, strong.values, weak.values):
        rows.append({"DATE": d.strftime("%Y-%m-%d"), "HIVE": "EQ", "hive_oos_ret": float(a)})
        rows.append({"DATE": d.strftime("%Y-%m-%d"), "HIVE": "FX", "hive_oos_ret": float(b)})
    pd.DataFrame(rows).to_csv(tmp_path / "hive_wf_oos_returns.csv", index=False)

    old_runs = rch.RUNS
    try:
        rch.RUNS = tmp_path
        mult = rch.dynamic_quality_multipliers(idx, ["EQ", "FX"])
    finally:
        rch.RUNS = old_runs

    assert set(mult.columns) == {"EQ", "FX"}
    assert len(mult) == len(idx)
    assert float(mult["EQ"].mean()) > float(mult["FX"].mean())
    assert float(mult.min().min()) >= 0.60 - 1e-9
    assert float(mult.max().max()) <= 1.45 + 1e-9
