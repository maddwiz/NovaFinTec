import json
from pathlib import Path

import numpy as np
import pandas as pd

import tools.run_calendar_event_overlay as ce


def _write_price_csv(path: Path, n: int = 180) -> None:
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    close = 400.0 + np.cumsum(np.sin(np.linspace(0.0, 14.0, n)) * 0.8)
    df = pd.DataFrame({"Date": idx, "Close": close})
    df.to_csv(path, index=False)


def test_run_calendar_event_overlay_writes_outputs(tmp_path: Path, monkeypatch):
    root = tmp_path
    runs = root / "runs_plus"
    data = root / "data"
    runs.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)

    _write_price_csv(data / "SPY.csv", n=210)
    np.savetxt(runs / "asset_returns.csv", np.zeros((160, 5), dtype=float), delimiter=",")
    events = pd.DataFrame(
        {
            "DATE": ["2022-03-31", "2022-04-01", "2022-05-31"],
            "direction": [1, 1, -1],
            "impact": [1.0, 0.7, 0.5],
        }
    )
    events.to_csv(runs / "calendar_events.csv", index=False)

    monkeypatch.setattr(ce, "ROOT", root)
    monkeypatch.setattr(ce, "RUNS", runs)
    monkeypatch.setattr(ce, "DATA", data)
    monkeypatch.setenv("Q_DISABLE_REPORT_CARDS", "1")

    rc = ce.main()
    assert rc == 0

    sig = np.loadtxt(runs / "calendar_event_signal.csv", delimiter=",").ravel()
    ov = np.loadtxt(runs / "calendar_event_overlay.csv", delimiter=",").ravel()
    info = json.loads((runs / "calendar_event_info.json").read_text(encoding="utf-8"))

    assert len(sig) == 160
    assert len(ov) == 160
    assert float(np.min(sig)) >= -1.0 - 1e-9
    assert float(np.max(sig)) <= 1.0 + 1e-9
    assert float(np.min(ov)) >= float(info["params"]["floor"]) - 1e-9
    assert float(np.max(ov)) <= float(info["params"]["ceil"]) + 1e-9
    assert info["ok"] is True
    assert info["event_file_exists"] is True


def test_run_calendar_event_overlay_missing_calendar_writes_neutral(tmp_path: Path, monkeypatch):
    root = tmp_path
    runs = root / "runs_plus"
    data = root / "data"
    runs.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    np.savetxt(runs / "asset_returns.csv", np.zeros((40, 2), dtype=float), delimiter=",")

    monkeypatch.setattr(ce, "ROOT", root)
    monkeypatch.setattr(ce, "RUNS", runs)
    monkeypatch.setattr(ce, "DATA", data)
    monkeypatch.setenv("Q_DISABLE_REPORT_CARDS", "1")

    rc = ce.main()
    assert rc == 0
    sig = np.loadtxt(runs / "calendar_event_signal.csv", delimiter=",").ravel()
    ov = np.loadtxt(runs / "calendar_event_overlay.csv", delimiter=",").ravel()
    info = json.loads((runs / "calendar_event_info.json").read_text(encoding="utf-8"))

    assert len(sig) == 40
    assert len(ov) == 40
    assert float(np.max(np.abs(sig))) == 0.0
    assert float(np.min(ov)) == 1.0
    assert float(np.max(ov)) == 1.0
    assert info["ok"] is False
