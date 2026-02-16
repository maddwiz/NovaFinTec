import tools.run_system_health as rsh
from pathlib import Path


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


def test_load_named_series_reads_requested_column(tmp_path: Path):
    p = tmp_path / "named.csv"
    p.write_text(
        "\n".join(
            [
                "DATE,entropy_target,entropy_strength",
                "2026-02-01,0.71,0.66",
                "2026-02-02,0.73,0.69",
            ]
        ),
        encoding="utf-8",
    )
    s = rsh._load_named_series(p, "entropy_strength")
    assert s is not None
    assert len(s) == 2
    assert float(s[-1]) == 0.69


def test_load_row_mean_series_ignores_date_columns(tmp_path: Path):
    p = tmp_path / "crowding.csv"
    p.write_text(
        "\n".join(
            [
                "DATE,EQ,FX,RATES",
                "2026-02-01,0.60,0.40,0.50",
                "2026-02-02,0.70,0.30,0.50",
            ]
        ),
        encoding="utf-8",
    )
    s = rsh._load_row_mean_series(p)
    assert s is not None
    assert len(s) == 2
    assert float(s[0]) == 0.5
    assert float(s[1]) == 0.5


def test_overlay_aion_feedback_metrics_extracts_and_flags_alert():
    metrics, issues = rsh._overlay_aion_feedback_metrics(
        {
            "runtime_context": {
                "aion_feedback": {
                    "active": True,
                    "status": "alert",
                    "risk_scale": 0.72,
                    "closed_trades": 12,
                    "hit_rate": 0.34,
                    "profit_factor": 0.70,
                    "expectancy": -2.1,
                    "drawdown_norm": 2.8,
                }
            }
        }
    )
    assert metrics.get("aion_feedback_active") is True
    assert metrics.get("aion_feedback_status") == "alert"
    assert int(metrics.get("aion_feedback_closed_trades")) == 12
    assert any("aion_feedback_status=alert" in x for x in issues)
    assert any("aion_feedback_risk_scale_low" in x for x in issues)


def test_overlay_aion_feedback_metrics_flags_stale_feedback(monkeypatch):
    monkeypatch.setenv("Q_MAX_AION_FEEDBACK_AGE_HOURS", "24")
    metrics, issues = rsh._overlay_aion_feedback_metrics(
        {
            "runtime_context": {
                "aion_feedback": {
                    "active": True,
                    "status": "ok",
                    "risk_scale": 0.98,
                    "closed_trades": 20,
                    "age_hours": 72.0,
                }
            }
        }
    )
    assert metrics.get("aion_feedback_stale") is True
    assert any("aion_feedback_stale" in x for x in issues)


def test_overlay_aion_feedback_metrics_stale_suppresses_status_alert():
    metrics, issues = rsh._overlay_aion_feedback_metrics(
        {
            "runtime_context": {
                "aion_feedback": {
                    "active": True,
                    "status": "alert",
                    "risk_scale": 0.70,
                    "closed_trades": 20,
                    "stale": True,
                }
            }
        }
    )
    assert metrics.get("aion_feedback_stale") is True
    assert any("aion_feedback_stale" in x for x in issues)
    assert not any("aion_feedback_status=alert" in x for x in issues)
