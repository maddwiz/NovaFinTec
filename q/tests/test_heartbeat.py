from pathlib import Path

import pandas as pd

from qmods.heartbeat import compute_heartbeat_from_returns


def test_compute_heartbeat_from_returns_writes_outputs(tmp_path: Path):
    rets = pd.Series([0.001, -0.002, 0.0005, 0.003, -0.001] * 30)
    out_json = tmp_path / "heartbeat.json"
    out_png = tmp_path / "heartbeat.png"
    out_bpm_csv = tmp_path / "heartbeat_bpm.csv"
    out_scaler_csv = tmp_path / "heartbeat_exposure_scaler.csv"

    compute_heartbeat_from_returns(
        rets,
        out_json=str(out_json),
        out_png=str(out_png),
        out_bpm_csv=str(out_bpm_csv),
        out_scaler_csv=str(out_scaler_csv),
    )

    assert out_json.exists()
    assert out_bpm_csv.exists()
    assert out_scaler_csv.exists()
    assert out_png.exists()
