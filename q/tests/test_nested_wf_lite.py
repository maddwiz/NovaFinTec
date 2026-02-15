import numpy as np

import tools.nested_wf_lite as nwl


def test_inner_score_multi_fold_returns_finite(monkeypatch):
    rng = np.random.default_rng(5)
    n = 1400
    ret = rng.normal(0.0002, 0.01, n)
    pos = np.clip(rng.normal(0.0, 0.5, n), -1.0, 1.0)
    cfg = {"cap": 0.8, "deadband": 0.05, "span": 3}

    monkeypatch.setattr(nwl, "INNER_FOLDS", 5)
    score = nwl._inner_score(ret, pos, cfg)
    assert np.isfinite(score)


def test_inner_score_respects_minimum_rows(monkeypatch):
    ret = np.zeros(100)
    pos = np.zeros(100)
    cfg = {"cap": 0.8, "deadband": 0.05, "span": 3}

    monkeypatch.setattr(nwl, "INNER_FOLDS", 4)
    score = nwl._inner_score(ret, pos, cfg)
    assert score < -1e8


def test_inner_score_with_large_purge_can_invalidate_folds(monkeypatch):
    rng = np.random.default_rng(12)
    n = 900
    ret = rng.normal(0.0001, 0.01, n)
    pos = np.clip(rng.normal(0.0, 0.4, n), -1.0, 1.0)
    cfg = {"cap": 0.8, "deadband": 0.05, "span": 3}

    monkeypatch.setattr(nwl, "INNER_FOLDS", 4)
    monkeypatch.setattr(nwl, "EMBARGO", 5)
    monkeypatch.setattr(nwl, "PURGE", 260)
    monkeypatch.setattr(nwl, "INNER_MIN", 700)
    score = nwl._inner_score(ret, pos, cfg)
    assert score < -1e8


def test_run_asset_nested_emits_fold_utilization_stats(monkeypatch):
    rng = np.random.default_rng(17)
    n = 1400
    ret = rng.normal(0.0001, 0.01, n)
    pos = np.clip(rng.normal(0.0, 0.5, n), -1.0, 1.0)

    monkeypatch.setattr(nwl, "TRAIN_MIN", 500)
    monkeypatch.setattr(nwl, "OUTER_STEP", 63)
    monkeypatch.setattr(nwl, "INNER_MIN", 252)
    monkeypatch.setattr(nwl, "INNER_STEP", 21)
    monkeypatch.setattr(nwl, "INNER_FOLDS", 3)
    monkeypatch.setattr(nwl, "EMBARGO", 5)
    monkeypatch.setattr(nwl, "PURGE", 3)
    monkeypatch.setattr(nwl, "MIN_ROWS", 200)
    monkeypatch.setattr(nwl, "MAX_ROWS", 0)

    stats, chosen = nwl.run_asset_nested(ret, pos)
    assert stats
    assert int(stats["outer_folds_used"]) == len(chosen)
    assert int(stats["outer_folds_attempted"]) >= int(stats["outer_folds_used"])
    assert 0.0 <= float(stats["outer_fold_utilization"]) <= 1.0
