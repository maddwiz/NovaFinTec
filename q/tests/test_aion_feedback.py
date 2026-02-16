from pathlib import Path

from qmods import aion_feedback as af


def test_resolve_shadow_trades_path_uses_env_override(monkeypatch, tmp_path: Path):
    shadow = tmp_path / "shadow.csv"
    monkeypatch.setenv("Q_AION_SHADOW_TRADES", str(shadow))
    assert af.resolve_shadow_trades_path(root=tmp_path) == shadow


def test_load_outcome_feedback_missing_shadow_trades(monkeypatch, tmp_path: Path):
    missing = tmp_path / "missing_shadow_trades.csv"
    monkeypatch.setenv("Q_AION_SHADOW_TRADES", str(missing))
    out = af.load_outcome_feedback(root=tmp_path)
    assert out["active"] is False
    assert out["status"] == "missing"
    assert out["risk_scale"] == 1.0
    assert out["path"] == str(missing)


def test_load_outcome_feedback_computes_stale_alert(monkeypatch, tmp_path: Path):
    shadow = tmp_path / "shadow_trades.csv"
    shadow.write_text(
        "\n".join(
            [
                "timestamp,symbol,side,pnl",
                "2025-01-01 10:00:00,AAPL,EXIT_BUY,-8.0",
                "2025-01-01 10:05:00,MSFT,EXIT_SELL,-7.0",
                "2025-01-01 10:10:00,NVDA,EXIT_BUY,-9.0",
                "2025-01-01 10:15:00,TSLA,EXIT_SELL,-6.0",
                "2025-01-01 10:20:00,AMZN,EXIT_BUY,-8.0",
                "2025-01-01 10:25:00,META,EXIT_SELL,-6.5",
                "2025-01-01 10:30:00,GOOG,EXIT_BUY,-7.0",
                "2025-01-01 10:35:00,AMD,EXIT_SELL,-5.5",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("Q_AION_SHADOW_TRADES", str(shadow))
    out = af.load_outcome_feedback(root=tmp_path, min_trades=8, max_age_hours=1.0)

    assert out["active"] is True
    assert int(out["closed_trades"]) == 8
    assert float(out["risk_scale"]) < 1.0
    assert out["stale"] is True
    assert out["last_closed_ts"] is not None
    assert "stale_feedback" in out["reasons"]


def test_load_outcome_feedback_can_skip_stale_reason(monkeypatch, tmp_path: Path):
    shadow = tmp_path / "shadow_trades.csv"
    shadow.write_text(
        "\n".join(
            [
                "timestamp,symbol,side,pnl",
                "2025-01-01 10:00:00,AAPL,EXIT_BUY,8.0",
                "2025-01-01 10:05:00,MSFT,EXIT_SELL,-3.0",
                "2025-01-01 10:10:00,NVDA,EXIT_BUY,7.0",
                "2025-01-01 10:15:00,TSLA,EXIT_SELL,-2.0",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("Q_AION_SHADOW_TRADES", str(shadow))
    out = af.load_outcome_feedback(root=tmp_path, min_trades=3, max_age_hours=1.0, mark_stale_reason=False)

    assert out["active"] is True
    assert out["stale"] is True
    assert "stale_feedback" not in out["reasons"]


def test_choose_feedback_source_prefers_fresh_shadow_when_overlay_stale():
    selected, source = af.choose_feedback_source(
        {
            "active": True,
            "status": "alert",
            "risk_scale": 0.70,
            "closed_trades": 20,
            "age_hours": 96.0,
            "max_age_hours": 24.0,
            "stale": True,
        },
        {
            "active": True,
            "status": "ok",
            "risk_scale": 0.96,
            "closed_trades": 20,
            "age_hours": 2.0,
            "max_age_hours": 24.0,
            "stale": False,
        },
        source_pref="auto",
    )
    assert source == "shadow_trades"
    assert selected.get("status") == "ok"


def test_choose_feedback_source_honors_overlay_preference():
    selected, source = af.choose_feedback_source(
        {"active": True, "status": "ok", "closed_trades": 10},
        {"active": True, "status": "warn", "closed_trades": 10},
        source_pref="overlay",
    )
    assert source == "overlay"
    assert selected.get("status") == "ok"


def test_choose_feedback_source_can_prefer_overlay_when_fresh():
    selected, source = af.choose_feedback_source(
        {"active": True, "status": "ok", "closed_trades": 10, "stale": False},
        {"active": True, "status": "warn", "closed_trades": 10, "stale": False},
        source_pref="auto",
        prefer_overlay_when_fresh=True,
    )
    assert source == "overlay"
    assert selected.get("status") == "ok"
