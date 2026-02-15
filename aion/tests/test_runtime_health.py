from pathlib import Path

from aion.exec.runtime_health import runtime_controls_stale_info, runtime_controls_stale_threshold_sec


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
