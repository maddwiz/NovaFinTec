import json
from pathlib import Path

import numpy as np
import pandas as pd

import tools.run_capacity_impact_guard as cig


def _write_price_vol(path: Path, close: np.ndarray, vol: np.ndarray) -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2021-01-01", periods=len(close), freq="B"),
            "Close": close.astype(float),
            "Volume": vol.astype(float),
        }
    )
    df.to_csv(path, index=False)


def test_capacity_impact_guard_outputs_scalars(tmp_path: Path, monkeypatch):
    root = tmp_path
    runs = root / "runs_plus"
    data = root / "data"
    runs.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)

    t = 180
    np.savetxt(runs / "asset_returns.csv", np.zeros((t, 2), dtype=float), delimiter=",")
    np.savetxt(runs / "weights_tail_blend.csv", np.tile(np.array([[0.50, 0.50]]), (t, 1)), delimiter=",")
    pd.DataFrame({"symbol": ["AAA", "BBB"]}).to_csv(runs / "asset_names.csv", index=False)

    x = np.linspace(0.0, 15.0, t)
    _write_price_vol(data / "AAA.csv", 100.0 + 2.0 * np.sin(x), 2.0e6 + 2.0e5 * np.cos(x))
    _write_price_vol(data / "BBB.csv", 60.0 + 1.5 * np.cos(x), 2.0e5 + 2.0e4 * np.sin(0.8 * x))

    monkeypatch.setattr(cig, "ROOT", root)
    monkeypatch.setattr(cig, "RUNS", runs)
    monkeypatch.setattr(cig, "DATA", data)
    monkeypatch.setenv("Q_CAPACITY_BOOK_USD", "20000000")
    monkeypatch.setenv("Q_CAPACITY_IMPACT_BETA", "0.45")
    monkeypatch.setenv("Q_CAPACITY_SQRT_SCALE", "40")

    rc = cig.main()
    assert rc == 0

    proxy = np.loadtxt(runs / "capacity_impact_proxy.csv", delimiter=",").ravel()
    scalar = np.loadtxt(runs / "capacity_impact_scalar.csv", delimiter=",").ravel()
    info = json.loads((runs / "capacity_impact_info.json").read_text(encoding="utf-8"))

    assert len(proxy) == t
    assert len(scalar) == t
    assert float(np.min(proxy)) >= 0.0
    assert float(np.max(proxy)) <= 3.0
    assert float(np.min(scalar)) >= float(info["params"]["floor"]) - 1e-9
    assert float(np.max(scalar)) <= float(info["params"]["ceil"]) + 1e-9
    assert float(np.mean(scalar)) < 1.0
