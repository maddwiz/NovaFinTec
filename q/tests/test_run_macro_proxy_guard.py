import json
from pathlib import Path

import numpy as np
import pandas as pd

import tools.run_macro_proxy_guard as mpg


def _write_series_csv(path: Path, values: np.ndarray) -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=len(values), freq="B"),
            "Close": values.astype(float),
        }
    )
    df.to_csv(path, index=False)


def test_run_macro_proxy_guard_writes_outputs(tmp_path: Path, monkeypatch):
    root = tmp_path
    runs = root / "runs_plus"
    data = root / "data"
    runs.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)

    t = 300
    x = np.linspace(0.0, 25.0, t)
    _write_series_csv(data / "VIX9D.csv", 18.0 + 2.5 * np.sin(x))
    _write_series_csv(data / "VIX3M.csv", 19.0 + 1.8 * np.cos(0.8 * x))
    _write_series_csv(data / "VIXCLS.csv", 17.0 + 2.2 * np.sin(0.9 * x))
    _write_series_csv(data / "DGS10.csv", 3.0 + 0.4 * np.sin(0.2 * x))
    _write_series_csv(data / "DGS2.csv", 3.2 + 0.5 * np.cos(0.2 * x))
    _write_series_csv(data / "DGS3MO.csv", 2.8 + 0.35 * np.sin(0.3 * x))
    _write_series_csv(data / "LQD.csv", 110.0 + np.cumsum(0.02 * np.sin(0.4 * x)))
    _write_series_csv(data / "HYG.csv", 90.0 + np.cumsum(0.03 * np.cos(0.4 * x)))

    # Target length for alignment.
    np.savetxt(runs / "asset_returns.csv", np.zeros((220, 4), dtype=float), delimiter=",")

    monkeypatch.setattr(mpg, "ROOT", root)
    monkeypatch.setattr(mpg, "RUNS", runs)
    monkeypatch.setattr(mpg, "DATA", data)
    monkeypatch.setenv("Q_DISABLE_REPORT_CARDS", "1")

    rc = mpg.main()
    assert rc == 0
    shock = np.loadtxt(runs / "macro_shock_proxy.csv", delimiter=",").ravel()
    scalar = np.loadtxt(runs / "macro_risk_scalar.csv", delimiter=",").ravel()
    info = json.loads((runs / "macro_proxy_info.json").read_text(encoding="utf-8"))

    assert len(shock) == 220
    assert len(scalar) == 220
    assert float(np.min(shock)) >= 0.0
    assert float(np.max(shock)) <= 1.0
    assert float(np.min(scalar)) >= float(info["params"]["floor"]) - 1e-9
    assert float(np.max(scalar)) <= float(info["params"]["ceil"]) + 1e-9
    assert info["ok"] is True
