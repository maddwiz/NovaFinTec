import tools.run_hit_rate_recovery_tune as hrt


def test_local_grid_is_sorted_and_bounded():
    g = hrt._local_grid(0.05, 0.0, 0.2, 0.02)
    assert g == sorted(g)
    assert g[0] >= 0.0
    assert g[-1] <= 0.2


def test_score_candidate_penalizes_low_hit():
    good = {"sharpe": 1.3, "hit_rate": 0.50, "max_drawdown": -0.05, "n": 300}
    bad = {"sharpe": 1.5, "hit_rate": 0.42, "max_drawdown": -0.05, "n": 300}
    s_good, d_good = hrt._score_candidate(good)
    s_bad, d_bad = hrt._score_candidate(bad)
    assert d_bad["hit_gap"] > d_good["hit_gap"]
    assert s_good > s_bad
