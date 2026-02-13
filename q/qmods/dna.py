# qmods/dna.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class DNAConfig:
    fast: int = 20    # ~1 trading month
    slow: int = 126   # ~6 months
    step: int = 21    # compare to ~1 month ago


def _moments(ret: pd.Series, w: int):
    r = ret.dropna()
    mu = r.rolling(w).mean()
    sd = r.rolling(w).std(ddof=1)
    skew = r.rolling(w).apply(
        lambda x: float(pd.Series(x).skew()) if len(x) >= 3 else np.nan,
        raw=False
    )
    return mu, sd, skew


def _latent_for_symbol(ret: pd.Series, cfg: DNAConfig) -> pd.DataFrame:
    mu_f, sd_f, sk_f = _moments(ret, cfg.fast)
    mu_s, sd_s, _     = _moments(ret, cfg.slow)
    Z = pd.concat([mu_f, mu_s, sd_f, sd_s, sk_f], axis=1)
    Z.columns = ["mu_fast", "mu_slow", "vol_fast", "vol_slow", "skew_fast"]
    return Z


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return float(np.dot(a, b) / (na * nb))


def _drift_series(Z: pd.DataFrame, step: int) -> pd.Series:
    """1 - cosine(latent_t, latent_{t-step}); 0=no change, higher=more change."""
    vals = []
    idx = Z.index
    for i in range(len(idx)):
        j = i - step
        if j < 0:
            vals.append(np.nan)
            continue
        a = Z.iloc[i].to_numpy(dtype=float)
        b = Z.iloc[j].to_numpy(dtype=float)
        a = np.nan_to_num(a, nan=0.0)
        b = np.nan_to_num(b, nan=0.0)
        vals.append(1.0 - _cosine(a, b))
    return pd.Series(vals, index=idx, name="dna_drift")


def compute_dna(
    prices: pd.DataFrame,
    out_json: str = "runs_plus/dna_drift.json",
    out_png: str = "runs_plus/dna_drift.png"
):
    """
    Writes:
      - runs_plus/dna_drift.json  (per-symbol time series; date keys as YYYY-MM-DD strings)
      - runs_plus/dna_drift.png   (avg drift chart)
    """
    cfg = DNAConfig()
    ret = prices.pct_change()
    drift_map: dict[str, dict[str, float]] = {}
    avg_series = []

    for sym in prices.columns:
        Z = _latent_for_symbol(ret[sym], cfg)
        d = _drift_series(Z, cfg.step)
        # Convert Timestamp keys -> ISO strings for JSON
        d_clean = d.dropna()
        d_str_keys = {ts.strftime("%Y-%m-%d"): float(val) for ts, val in d_clean.items()}
        drift_map[sym] = d_str_keys
        avg_series.append(d)

    # ensure output dir
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)

    # write JSON (keys are strings now)
    Path(out_json).write_text(json.dumps({"dna_drift": drift_map}, indent=2))

    # average drift plot (smoothed)
    if avg_series:
        df = pd.concat(avg_series, axis=1)
        df.columns = prices.columns
        avg_drift = df.mean(axis=1).rolling(5).mean()
        plt.figure(figsize=(8, 3))
        avg_drift.plot()
        plt.title("DNA Drift (avg across symbols)")
        plt.xlabel("")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()

    return out_json, out_png
