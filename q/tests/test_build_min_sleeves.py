from pathlib import Path

import numpy as np
import pandas as pd

from tools.build_min_sleeves import build_min_sleeves


def _write_portfolio_plus(path: Path):
    dates = pd.date_range("2025-01-02", periods=6, freq="B")
    ret = pd.Series([0.010, -0.006, 0.004, -0.012, 0.007, 0.003], dtype=float)
    pd.DataFrame({"DATE": dates, "ret": ret}).to_csv(path / "portfolio_plus.csv", index=False)
    return dates


def test_build_min_sleeves_preserves_existing_symbolic_reflexive(tmp_path: Path):
    dates = _write_portfolio_plus(tmp_path)

    rows_sym = []
    rows_ref = []
    for dt in dates:
        rows_sym.append({"DATE": dt, "ASSET": "AAA", "sym_signal": 0.80, "confidence": 0.80})
        rows_sym.append({"DATE": dt, "ASSET": "BBB", "sym_signal": -0.40, "confidence": 0.20})
        rows_ref.append({"DATE": dt, "ASSET": "AAA", "reflexive_signal": 0.50, "reflex_confidence": 0.70})
        rows_ref.append({"DATE": dt, "ASSET": "BBB", "reflexive_signal": -0.10, "reflex_confidence": 0.30})

    pd.DataFrame(rows_sym).to_csv(tmp_path / "symbolic_signal.csv", index=False)
    pd.DataFrame(rows_ref).to_csv(tmp_path / "reflexive_signal.csv", index=False)

    info = build_min_sleeves(tmp_path)

    out_sym = pd.read_csv(tmp_path / "symbolic_signal.csv")
    out_ref = pd.read_csv(tmp_path / "reflexive_signal.csv")
    assert info["symbolic_source"] == "existing"
    assert info["reflexive_source"] == "existing"
    assert len(out_sym) == len(dates)
    assert len(out_ref) == len(dates)

    expected_sym = (0.80 * 0.80 + (-0.40) * 0.20) / (0.80 + 0.20)
    expected_ref = (0.50 * 0.70 + (-0.10) * 0.30) / (0.70 + 0.30)
    assert float(out_sym["sym_signal"].iloc[0]) == np.clip(expected_sym, -1.0, 1.0)
    assert float(out_ref["reflexive_signal"].iloc[0]) == np.clip(expected_ref, -1.0, 1.0)

    assert (tmp_path / "sleeve_vol.csv").exists()
    assert (tmp_path / "sleeve_osc.csv").exists()
    assert (tmp_path / "min_sleeves_info.json").exists()


def test_build_min_sleeves_fallbacks_when_symbolic_reflexive_missing(tmp_path: Path):
    _write_portfolio_plus(tmp_path)

    info = build_min_sleeves(tmp_path)
    out_sym = pd.read_csv(tmp_path / "symbolic_signal.csv")
    out_ref = pd.read_csv(tmp_path / "reflexive_signal.csv")

    assert info["symbolic_source"].startswith("fallback(")
    assert info["reflexive_source"].startswith("fallback(")
    assert out_sym["sym_signal"].between(-1.0, 1.0).all()
    assert out_ref["reflexive_signal"].between(-1.0, 1.0).all()
    assert float(out_sym["sym_signal"].abs().sum()) > 0.0
    assert float(out_ref["reflexive_signal"].abs().sum()) > 0.0
