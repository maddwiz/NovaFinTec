import tools.run_system_health as rsh


def test_analyze_execution_constraints_no_issues_on_reasonable_profile():
    metrics, issues = rsh._analyze_execution_constraints(
        {
            "gross_before_mean": 0.90,
            "gross_after_mean": 0.65,
            "turnover_before_mean": 0.35,
            "turnover_after_mean": 0.22,
            "turnover_after_max": 0.30,
            "max_step_turnover": 0.30,
        }
    )
    assert "exec_turnover_after_mean" in metrics
    assert not issues


def test_analyze_execution_constraints_flags_turnover_increase():
    _metrics, issues = rsh._analyze_execution_constraints(
        {
            "turnover_before_mean": 0.20,
            "turnover_after_mean": 0.25,
        }
    )
    assert any("increased mean turnover" in x for x in issues)


def test_analyze_execution_constraints_flags_over_throttle():
    _metrics, issues = rsh._analyze_execution_constraints(
        {
            "turnover_before_mean": 0.30,
            "turnover_after_mean": 0.00,
        }
    )
    assert any("over-throttling turnover" in x for x in issues)


def test_analyze_execution_constraints_flags_step_violation():
    _metrics, issues = rsh._analyze_execution_constraints(
        {
            "max_step_turnover": 0.30,
            "turnover_after_max": 0.32,
        }
    )
    assert any("exceeds configured max_step_turnover" in x for x in issues)


def test_staleness_issues_flags_old_required_and_optional():
    stats, issues = rsh._staleness_issues(
        [
            {"file": "a.csv", "required": True, "exists": True, "hours_since_update": 30.0},
            {"file": "b.csv", "required": False, "exists": True, "hours_since_update": 90.0},
            {"file": "c.csv", "required": True, "exists": False, "hours_since_update": None},
        ],
        max_required_hours=24.0,
        max_optional_hours=72.0,
    )
    assert stats["stale_required_count"] == 1
    assert stats["stale_optional_count"] == 1
    assert any("stale_required_file>" in x for x in issues)
    assert any("stale_optional_file>" in x for x in issues)


def test_staleness_issues_ignores_fresh_files():
    stats, issues = rsh._staleness_issues(
        [
            {"file": "a.csv", "required": True, "exists": True, "hours_since_update": 2.0},
            {"file": "b.csv", "required": False, "exists": True, "hours_since_update": 6.0},
        ],
        max_required_hours=24.0,
        max_optional_hours=72.0,
    )
    assert stats["stale_required_count"] == 0
    assert stats["stale_optional_count"] == 0
    assert issues == []
