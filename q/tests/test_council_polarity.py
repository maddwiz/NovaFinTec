import json
from pathlib import Path

import numpy as np
import pandas as pd

from qmods.council import _member_stats, run_council


def _frames(seed: int = 7, t: int = 160, n: int = 12):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02", periods=t, freq="B")
    cols = [f"A{i:02d}" for i in range(n)]
    base = pd.DataFrame(rng.normal(0.0, 1.0, size=(t, n)), index=idx, columns=cols)
    return base


def test_member_stats_detects_anti_predictive_signal():
    fwd = _frames()
    sig_pos = fwd.copy()
    sig_neg = -fwd
    sig_noise = _frames(seed=99)

    pos = _member_stats(sig_pos, fwd, lookback=120)
    neg = _member_stats(sig_neg, fwd, lookback=120)
    noise = _member_stats(sig_noise, fwd, lookback=120)

    assert pos["quality"] > 0.5
    assert pos["polarity"] > 0.5
    assert pos["mean_ic"] > 0.2

    assert neg["quality"] > 0.5
    assert neg["polarity"] < -0.5
    assert neg["mean_ic"] < -0.2

    assert abs(noise["polarity"]) <= 0.5
    assert noise["quality"] >= 0.10


def test_run_council_emits_polarity_metadata(tmp_path: Path):
    rng = np.random.default_rng(11)
    idx = pd.date_range("2023-01-03", periods=280, freq="B")
    cols = [f"S{i:02d}" for i in range(18)]
    rets = rng.normal(0.0004, 0.012, size=(len(idx), len(cols)))
    prices = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=cols)

    out_json = tmp_path / "council.json"
    out = run_council(prices, out_json=str(out_json))

    assert "member_quality" in out
    assert "member_polarity" in out
    assert "member_mean_ic" in out
    assert "member_ic_samples" in out
    assert set(out["member_quality"].keys()) == set(out["member_polarity"].keys())

    payload = json.loads(out_json.read_text())
    assert "member_polarity" in payload
