import tools.run_quality_governor as rqg
import numpy as np


def test_execution_constraint_quality_high_for_reasonable_retention():
    q, detail = rqg._execution_constraint_quality(
        {
            "gross_before_mean": 0.80,
            "gross_after_mean": 0.62,
            "turnover_before_mean": 0.20,
            "turnover_after_mean": 0.13,
            "turnover_after_max": 0.30,
            "max_step_turnover": 0.35,
        },
        [],
    )
    assert q is not None
    assert 0.60 <= float(q) <= 1.0
    assert float(detail["gross_retention"]) > 0.7


def test_execution_constraint_quality_drops_when_over_throttled():
    q, _detail = rqg._execution_constraint_quality(
        {
            "gross_before_mean": 0.40,
            "gross_after_mean": 0.01,
            "turnover_before_mean": 0.22,
            "turnover_after_mean": 0.001,
        },
        ["execution constraints may be over-throttling turnover"],
    )
    assert q is not None
    assert float(q) < 0.20


def test_execution_constraint_quality_drops_when_turnover_increases():
    q, detail = rqg._execution_constraint_quality(
        {
            "gross_before_mean": 0.30,
            "gross_after_mean": 0.28,
            "turnover_before_mean": 0.10,
            "turnover_after_mean": 0.18,
            "turnover_after_max": 0.40,
            "max_step_turnover": 0.30,
        },
        [],
    )
    assert q is not None
    assert float(detail["turnover_retention"]) > 1.0
    assert float(q) < 0.60


def test_cap_step_change_limits_adjacent_moves():
    x = np.array([0.60, 0.90, 0.30, 1.10], dtype=float)
    y = rqg._cap_step_change(x, max_step=0.10, lo=0.55, hi=1.15)
    d = np.diff(y)
    assert float(np.max(np.abs(d))) <= 0.10 + 1e-12
    assert float(np.min(y)) >= 0.55 - 1e-12
    assert float(np.max(y)) <= 1.15 + 1e-12


def test_nested_leakage_quality_high_when_utilization_is_healthy():
    q, detail = rqg._nested_leakage_quality(
        {
            "assets": 20,
            "avg_outer_fold_utilization": 0.82,
            "low_utilization_assets": 1,
            "avg_train_ratio_mean": 0.88,
            "params": {"purge_embargo_ratio": 0.22},
        }
    )
    assert q is not None
    assert float(q) > 0.75
    assert float(detail["avg_outer_fold_utilization"]) == 0.82


def test_nested_leakage_quality_drops_with_low_utilization_and_heavy_purge():
    q, detail = rqg._nested_leakage_quality(
        {
            "assets": 16,
            "avg_outer_fold_utilization": 0.38,
            "low_utilization_assets": 10,
            "avg_train_ratio_mean": 0.50,
            "params": {"purge_embargo_ratio": 0.95},
        }
    )
    assert q is not None
    assert float(q) < 0.45
    assert float(detail["purge_embargo_ratio"]) == 0.95
