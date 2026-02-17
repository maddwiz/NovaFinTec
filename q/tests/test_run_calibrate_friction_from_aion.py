import json
from pathlib import Path

import pandas as pd

import tools.run_calibrate_friction_from_aion as cal


def _write_shadow(path: Path, rows: int) -> None:
    now = pd.Timestamp.now(tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": [(now - pd.Timedelta(minutes=5 * i)).isoformat() for i in range(rows)][::-1],
            "symbol": ["AAPL"] * rows,
            "side": ["ENTRY_BUY"] * rows,
            "qty": [10 + (i % 5) for i in range(rows)],
            "entry": [100.0] * rows,
            "exit": [0.0] * rows,
            "pnl": [0.0] * rows,
            "reason": ["test"] * rows,
            "confidence": [0.8] * rows,
            "regime": ["trending"] * rows,
            "stop": [99.0] * rows,
            "target": [102.0] * rows,
            "trail": [99.0] * rows,
            "fill_ratio": [0.92] * rows,
            "slippage_bps": [4.0 + 0.1 * i for i in range(rows)],
        }
    )
    df.to_csv(path, index=False)


def test_calibrate_friction_writes_recommendation(monkeypatch, tmp_path: Path):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cal, "ROOT", tmp_path)
    monkeypatch.setattr(cal, "RUNS", runs)

    shadow = tmp_path / "shadow_trades.csv"
    monitor = tmp_path / "runtime_monitor.json"
    _write_shadow(shadow, rows=36)
    monitor.write_text(
        json.dumps(
            {
                "slippage_points": [5.2, 5.7, 4.9, 6.1, 5.4],
            }
        ),
        encoding="utf-8",
    )
    (runs / "daily_costs_info.json").write_text(
        json.dumps({"cost_base_bps": 8.0}),
        encoding="utf-8",
    )

    monkeypatch.setenv("Q_AION_SHADOW_TRADES", str(shadow))
    monkeypatch.setenv("Q_AION_RUNTIME_MONITOR", str(monitor))
    monkeypatch.setenv("Q_FRICTION_CALIB_MIN_SAMPLES", "10")
    monkeypatch.setenv("Q_DISABLE_REPORT_CARDS", "1")

    rc = cal.main()
    assert rc == 0
    out = json.loads((runs / "friction_calibration.json").read_text(encoding="utf-8"))
    assert out["ok"] is True
    rec = out["recommendation"]
    assert float(rec["recommended_cost_base_bps"]) >= 8.0
    assert float(rec["recommended_cost_vol_scaled_bps"]) >= 0.0
    assert int(rec["stats"]["samples_total"]) >= 10


def test_calibrate_friction_marks_insufficient_samples(monkeypatch, tmp_path: Path):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cal, "ROOT", tmp_path)
    monkeypatch.setattr(cal, "RUNS", runs)

    shadow = tmp_path / "shadow_trades.csv"
    _write_shadow(shadow, rows=3)

    monkeypatch.setenv("Q_AION_SHADOW_TRADES", str(shadow))
    monkeypatch.setenv("Q_AION_RUNTIME_MONITOR", str(tmp_path / "missing_runtime_monitor.json"))
    monkeypatch.setenv("Q_FRICTION_CALIB_MIN_SAMPLES", "20")
    monkeypatch.setenv("Q_DISABLE_REPORT_CARDS", "1")

    rc = cal.main()
    assert rc == 0
    out = json.loads((runs / "friction_calibration.json").read_text(encoding="utf-8"))
    assert out["ok"] is False
    reasons = out["recommendation"].get("reasons", [])
    assert any("insufficient_samples" in str(r) for r in reasons)
