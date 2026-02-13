# qmods/heartbeat.py
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
class HBConfig:
    window: int = 20          # lookback for realized vol (trading days)
    base_bpm: float = 60.0    # calm heartbeat
    max_bpm: float = 180.0    # crisis heartbeat
    # map realized vol range [vol_lo, vol_hi] -> [base_bpm, max_bpm]
    vol_lo: float = 0.01      # 1% daily vol ~ very calm
    vol_hi: float = 0.05      # 5% daily vol ~ highly volatile
    smooth_span: int = 5      # smooth BPM jitter
    percentile_win: int = 126 # regime percentile horizon

def realized_vol(prices: pd.DataFrame, window: int) -> pd.Series:
    """
    Cross-sectional realized volatility: mean of asset stddev (daily returns) over 'window'.
    """
    rets = prices.pct_change()
    # daily std per symbol
    sd = rets.rolling(window).std(ddof=1)
    # cross-sectional average std per date
    vol = sd.mean(axis=1)
    return vol

def realized_vol_from_returns(returns: pd.Series, window: int) -> pd.Series:
    r = pd.Series(pd.to_numeric(returns, errors="coerce")).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return r.rolling(int(max(2, window)), min_periods=max(5, int(window // 3))).std(ddof=1).fillna(0.0)

def _robust_percentile_rank(x: pd.Series, win: int) -> pd.Series:
    def _pct(arr):
        a = pd.Series(arr).dropna().values
        if a.size < 5:
            return np.nan
        v = a[-1]
        return float((a <= v).mean())
    return x.rolling(win, min_periods=max(20, win // 4)).apply(_pct, raw=True)

def map_vol_to_bpm(vol: pd.Series, cfg: HBConfig) -> pd.Series:
    """
    Linearly map vol into [base_bpm, max_bpm] with clipping.
    """
    v = vol.clip(lower=cfg.vol_lo, upper=cfg.vol_hi)
    t = (v - cfg.vol_lo) / max(cfg.vol_hi - cfg.vol_lo, 1e-12)
    bpm = cfg.base_bpm + (cfg.max_bpm - cfg.base_bpm) * t
    bpm = bpm.ewm(span=max(2, int(cfg.smooth_span)), adjust=False).mean()
    return bpm.rename("heartbeat_bpm")

def bpm_to_exposure_scaler(bpm: pd.Series, cfg: HBConfig) -> pd.Series:
    # high BPM => lower risk budget
    pct = _robust_percentile_rank(bpm, win=cfg.percentile_win).fillna(0.5).clip(0.0, 1.0)
    scaler = 1.15 - 0.65 * pct
    return scaler.clip(0.45, 1.15).rename("exposure_scaler")

def _write_outputs(
    bpm: pd.Series,
    scaler: pd.Series,
    out_json: str,
    out_png: str,
    out_bpm_csv: str,
    out_scaler_csv: str,
    source: str,
):
    d = bpm.dropna()
    s = scaler.reindex(d.index).ffill().fillna(1.0)
    hb_map = {ts.strftime("%Y-%m-%d"): float(val) for ts, val in d.items()}
    sc_map = {ts.strftime("%Y-%m-%d"): float(val) for ts, val in s.items()}

    payload = {
        "heartbeat": hb_map,
        "exposure_scaler_series": sc_map,
        "exposure_scaler": float(s.iloc[-1]) if len(s) else 1.0,
        "source": source,
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(payload, indent=2))

    hb_csv = pd.DataFrame({"DATE": pd.to_datetime(d.index), "heartbeat_bpm": d.values})
    sc_csv = pd.DataFrame({"DATE": pd.to_datetime(s.index), "heartbeat_exposure_scaler": s.values})
    hb_csv.to_csv(out_bpm_csv, index=False)
    sc_csv.to_csv(out_scaler_csv, index=False)

    if not d.empty:
        plt.figure(figsize=(8, 3))
        d.plot(label="BPM")
        (60 + (180 - 60) * (1 - s)).plot(label="Risk Budget Proxy", alpha=0.6)
        plt.title("Heartbeat Metabolism (BPM)")
        plt.legend(loc="best")
        plt.xlabel("")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()

def compute_heartbeat(prices: pd.DataFrame,
                      out_json: str = "runs_plus/heartbeat.json",
                      out_png: str = "runs_plus/heartbeat.png",
                      out_bpm_csv: str = "runs_plus/heartbeat_bpm.csv",
                      out_scaler_csv: str = "runs_plus/heartbeat_exposure_scaler.csv"):
    """
    Writes:
      - runs_plus/heartbeat.json  -> {"heartbeat": {"YYYY-MM-DD": bpm, ...}}
      - runs_plus/heartbeat.png   -> line chart of bpm
    """
    cfg = HBConfig()
    vol = realized_vol(prices, cfg.window)
    bpm = map_vol_to_bpm(vol, cfg)
    scaler = bpm_to_exposure_scaler(bpm, cfg)
    _write_outputs(
        bpm=bpm,
        scaler=scaler,
        out_json=out_json,
        out_png=out_png,
        out_bpm_csv=out_bpm_csv,
        out_scaler_csv=out_scaler_csv,
        source="prices",
    )
    return out_json, out_png

def compute_heartbeat_from_returns(
    returns: pd.Series,
    out_json: str = "runs_plus/heartbeat.json",
    out_png: str = "runs_plus/heartbeat.png",
    out_bpm_csv: str = "runs_plus/heartbeat_bpm.csv",
    out_scaler_csv: str = "runs_plus/heartbeat_exposure_scaler.csv",
):
    cfg = HBConfig()
    vol = realized_vol_from_returns(returns, cfg.window)
    bpm = map_vol_to_bpm(vol, cfg)
    scaler = bpm_to_exposure_scaler(bpm, cfg)
    _write_outputs(
        bpm=bpm,
        scaler=scaler,
        out_json=out_json,
        out_png=out_png,
        out_bpm_csv=out_bpm_csv,
        out_scaler_csv=out_scaler_csv,
        source="returns",
    )
    return out_json, out_png
