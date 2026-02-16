import json
import os
from datetime import datetime, timezone
from pathlib import Path

from aion.exec.runtime_health import (
    aion_feedback_runtime_info,
    overlay_runtime_status,
    runtime_controls_stale_info,
    runtime_controls_stale_threshold_sec,
)


def test_runtime_controls_stale_threshold_uses_watchlist_and_loop_seconds():
    p = {"loop_seconds": 30, "watchlist_size": 180}
    thr = runtime_controls_stale_threshold_sec(
        p,
        default_loop_seconds=30,
        base_stale_seconds=120,
    )
    assert thr >= 600.0


def test_runtime_controls_stale_info_reads_payload_and_age(tmp_path: Path):
    rc = tmp_path / "runtime_controls.json"
    rc.write_text('{"loop_seconds": 20, "watchlist_size": 24}', encoding="utf-8")
    info = runtime_controls_stale_info(
        rc,
        default_loop_seconds=30,
        base_stale_seconds=120,
    )
    assert isinstance(info["payload"], dict)
    assert info["payload"]["loop_seconds"] == 20
    assert info["age_sec"] is not None
    assert info["threshold_sec"] >= 120


def test_aion_feedback_runtime_info_prefers_runtime_controls_and_flags_stale():
    out = aion_feedback_runtime_info(
        {
            "aion_feedback_active": True,
            "aion_feedback_status": "warn",
            "aion_feedback_risk_scale": 0.88,
            "aion_feedback_closed_trades": 14,
            "aion_feedback_age_hours": 96.0,
            "aion_feedback_max_age_hours": 72.0,
            "aion_feedback_stale": False,
            "aion_feedback_last_closed_ts": "2026-02-16T15:35:00Z",
        },
        {"runtime_context": {"aion_feedback": {"active": True, "status": "ok", "risk_scale": 1.0}}},
    )
    assert out["source"] == "runtime_controls"
    assert out["present"] is True
    assert out["stale"] is True
    assert out["state"] == "stale"
    assert out["last_closed_ts"] == "2026-02-16T15:35:00Z"


def test_aion_feedback_runtime_info_falls_back_to_overlay_context():
    out = aion_feedback_runtime_info(
        {},
        {
            "runtime_context": {
                "aion_feedback": {
                    "active": True,
                    "status": "alert",
                    "risk_scale": 0.76,
                    "closed_trades": 20,
                    "last_closed_ts": "2026-02-16T15:35:00Z",
                    "stale": False,
                }
            }
        },
    )
    assert out["source"] == "overlay_runtime_context"
    assert out["state"] == "alert"
    assert out["present"] is True
    assert out["last_closed_ts"] == "2026-02-16T15:35:00Z"


def test_overlay_runtime_status_prefers_payload_timestamp_over_mtime(tmp_path: Path):
    p = tmp_path / "overlay.json"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    p.write_text(
        json.dumps(
            {
                "generated_at_utc": ts,
                "runtime_context": {"runtime_multiplier": 0.9, "risk_flags": ["drift_warn"]},
            }
        ),
        encoding="utf-8",
    )
    old_ts = 946684800
    p.touch()
    p.chmod(0o644)
    # Keep old mtime so payload timestamp should still win.
    os.utime(p, (old_ts, old_ts))
    out = overlay_runtime_status(p, max_age_hours=1.0)
    assert out["exists"] is True
    assert out["age_source"] == "payload"
    assert out["stale"] is False
    assert "drift_warn" in out["risk_flags"]
