from pathlib import Path

import pandas as pd

from qmods.heartbeat import HBConfig, bpm_to_exposure_scaler, compute_heartbeat_from_returns, map_vol_to_bpm


def test_compute_heartbeat_from_returns_writes_outputs(tmp_path: Path):
    rets = pd.Series([0.001, -0.002, 0.0005, 0.003, -0.001] * 30)
    out_json = tmp_path / "heartbeat.json"
    out_png = tmp_path / "heartbeat.png"
    out_bpm_csv = tmp_path / "heartbeat_bpm.csv"
    out_scaler_csv = tmp_path / "heartbeat_exposure_scaler.csv"
    out_stress_csv = tmp_path / "heartbeat_stress.csv"

    compute_heartbeat_from_returns(
        rets,
        out_json=str(out_json),
        out_png=str(out_png),
        out_bpm_csv=str(out_bpm_csv),
        out_scaler_csv=str(out_scaler_csv),
        out_stress_csv=str(out_stress_csv),
    )

    assert out_json.exists()
    assert out_bpm_csv.exists()
    assert out_scaler_csv.exists()
    assert out_stress_csv.exists()
    assert out_png.exists()


def test_map_vol_to_bpm_adaptive_tracks_regime_shift():
    cfg = HBConfig()
    vol = pd.Series([0.008] * 120 + [0.020] * 120 + [0.045] * 120)
    bpm = map_vol_to_bpm(vol, cfg)
    scaler = bpm_to_exposure_scaler(bpm, cfg)

    lo = float(bpm.iloc[:120].mean())
    mid = float(bpm.iloc[120:240].mean())
    hi = float(bpm.iloc[240:].mean())
    assert hi > mid > lo

    slo = float(scaler.iloc[:120].mean())
    smid = float(scaler.iloc[120:240].mean())
    shi = float(scaler.iloc[240:].mean())
    assert slo > smid
    assert slo > shi
    assert min(smid, shi) <= (slo - 0.05)
