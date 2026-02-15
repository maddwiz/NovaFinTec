# qmods/heartbeat.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

# Optional plotting (headless). Core heartbeat outputs do not require matplotlib.
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    matplotlib = None
    plt = None

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
    adaptive_win: int = 252   # dynamic vol band horizon
    adaptive_q_lo: float = 0.20
    adaptive_q_hi: float = 0.90
    adaptive_blend: float = 0.65  # blend static vs adaptive mapping

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
        lt = float((a < v).mean())
        eq = float((a == v).mean())
        return float(np.clip(lt + 0.5 * eq, 0.0, 1.0))
    return x.rolling(win, min_periods=max(20, win // 4)).apply(_pct, raw=True)


def heartbeat_stress_from_bpm(bpm: pd.Series, cfg: HBConfig) -> pd.Series:
    """
    Stress score in [0,1] from heartbeat level and acceleration.
    Higher means harsher risk conditions.
    """
    b = pd.Series(pd.to_numeric(bpm, errors="coerce")).replace([np.inf, -np.inf], np.nan).fillna(cfg.base_bpm)
    lvl_pct = _robust_percentile_rank(b, win=cfg.percentile_win).fillna(0.5).clip(0.0, 1.0)
    lvl_abs = ((b - cfg.base_bpm) / max(1e-9, cfg.max_bpm - cfg.base_bpm)).clip(0.0, 1.0)
    lvl = (0.45 * lvl_pct + 0.55 * lvl_abs).clip(0.0, 1.0)
    jerk = b.diff().abs().fillna(0.0)
    jerk = jerk.rolling(21, min_periods=5).mean().fillna(0.0)
    jrk = _robust_percentile_rank(jerk, win=cfg.percentile_win).fillna(0.5).clip(0.0, 1.0)

    # Directional asymmetry: rising BPM is riskier than falling BPM.
    slope = b.diff().fillna(0.0)
    rise = slope.clip(lower=0.0).rolling(21, min_periods=5).mean().fillna(0.0)
    fall = (-slope).clip(lower=0.0).rolling(21, min_periods=5).mean().fillna(0.0)
    rise_rank = _robust_percentile_rank(rise, win=cfg.percentile_win).fillna(0.5).clip(0.0, 1.0)
    fall_rank = _robust_percentile_rank(fall, win=cfg.percentile_win).fillna(0.5).clip(0.0, 1.0)

    stress = np.clip(0.58 * lvl + 0.24 * jrk + 0.26 * rise_rank - 0.08 * fall_rank, 0.0, 1.0)
    return pd.Series(stress, index=b.index, name="heartbeat_stress")


def map_vol_to_bpm(vol: pd.Series, cfg: HBConfig) -> pd.Series:
    """
    Map vol into [base_bpm, max_bpm] using a blend of static and adaptive
    rolling quantile bands to improve robustness across universes/regimes.
    """
    v = pd.Series(pd.to_numeric(vol, errors="coerce")).replace([np.inf, -np.inf], np.nan)
    v = v.ffill().bfill().fillna(cfg.vol_lo).clip(lower=0.0)
    t_static = ((v.clip(lower=cfg.vol_lo, upper=cfg.vol_hi) - cfg.vol_lo) / max(cfg.vol_hi - cfg.vol_lo, 1e-12)).clip(0.0, 1.0)

    aw = int(max(20, cfg.adaptive_win))
    lo = v.rolling(aw, min_periods=max(20, aw // 6)).quantile(float(np.clip(cfg.adaptive_q_lo, 0.01, 0.60)))
    hi = v.rolling(aw, min_periods=max(20, aw // 6)).quantile(float(np.clip(cfg.adaptive_q_hi, 0.40, 0.99)))
    lo = lo.fillna(cfg.vol_lo)
    hi = hi.fillna(cfg.vol_hi)
    span = (hi - lo).clip(lower=max(1e-6, 0.1 * (cfg.vol_hi - cfg.vol_lo)))
    t_adapt = ((v - lo) / span).clip(0.0, 1.0)

    stress_w = _robust_percentile_rank(v, win=cfg.percentile_win).fillna(0.5).clip(0.0, 1.0)
    base_blend = float(np.clip(cfg.adaptive_blend, 0.0, 1.0))
    blend = (base_blend + 0.15 * (stress_w - 0.5)).clip(0.25, 0.90)
    t = (1.0 - blend) * t_static + blend * t_adapt

    bpm = cfg.base_bpm + (cfg.max_bpm - cfg.base_bpm) * t
    # Damp sudden bpm jumps to reduce allocator whipsaw in transition zones.
    accel = bpm.diff().abs().fillna(0.0)
    accel_rank = _robust_percentile_rank(accel, win=cfg.percentile_win).fillna(0.5).clip(0.0, 1.0)
    bpm = cfg.base_bpm + (bpm - cfg.base_bpm) * (1.0 - 0.22 * accel_rank)
    bpm = bpm.ewm(span=max(2, int(cfg.smooth_span)), adjust=False).mean()
    return bpm.rename("heartbeat_bpm")

def bpm_to_exposure_scaler(bpm: pd.Series, cfg: HBConfig) -> pd.Series:
    # high BPM + high BPM acceleration => lower risk budget.
    stress = heartbeat_stress_from_bpm(bpm, cfg)
    scaler = 1.18 - 0.70 * stress
    return scaler.clip(0.40, 1.15).rename("exposure_scaler")

def _write_outputs(
    bpm: pd.Series,
    scaler: pd.Series,
    stress: pd.Series,
    out_json: str,
    out_png: str,
    out_bpm_csv: str,
    out_scaler_csv: str,
    out_stress_csv: str,
    source: str,
):
    d = bpm.dropna()
    s = scaler.reindex(d.index).ffill().fillna(1.0)
    h = stress.reindex(d.index).ffill().fillna(0.5)

    def _fmt_ts(ts):
        try:
            return ts.strftime("%Y-%m-%d")
        except Exception:
            try:
                t = pd.to_datetime(ts, errors="coerce")
                if pd.notna(t):
                    return t.strftime("%Y-%m-%d")
            except Exception:
                pass
            return str(ts)

    hb_map = {_fmt_ts(ts): float(val) for ts, val in d.items()}
    sc_map = {_fmt_ts(ts): float(val) for ts, val in s.items()}
    hs_map = {_fmt_ts(ts): float(val) for ts, val in h.items()}

    payload = {
        "heartbeat": hb_map,
        "exposure_scaler_series": sc_map,
        "exposure_scaler": float(s.iloc[-1]) if len(s) else 1.0,
        "stress_series": hs_map,
        "stress": float(h.iloc[-1]) if len(h) else 0.5,
        "source": source,
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(payload, indent=2))

    hb_csv = pd.DataFrame({"DATE": d.index, "heartbeat_bpm": d.values})
    sc_csv = pd.DataFrame({"DATE": s.index, "heartbeat_exposure_scaler": s.values})
    hs_csv = pd.DataFrame({"DATE": h.index, "heartbeat_stress": h.values})
    hb_csv.to_csv(out_bpm_csv, index=False)
    sc_csv.to_csv(out_scaler_csv, index=False)
    hs_csv.to_csv(out_stress_csv, index=False)

    if plt is not None and not d.empty:
        plt.figure(figsize=(8, 3))
        d.plot(label="BPM")
        (60 + (180 - 60) * (1 - s)).plot(label="Risk Budget Proxy", alpha=0.6)
        plt.title("Heartbeat Metabolism (BPM)")
        plt.legend(loc="best")
        plt.xlabel("")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
    elif plt is None:
        png_path = Path(out_png)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        png_path.touch(exist_ok=True)

def compute_heartbeat(prices: pd.DataFrame,
                      out_json: str = "runs_plus/heartbeat.json",
                      out_png: str = "runs_plus/heartbeat.png",
                      out_bpm_csv: str = "runs_plus/heartbeat_bpm.csv",
                      out_scaler_csv: str = "runs_plus/heartbeat_exposure_scaler.csv",
                      out_stress_csv: str = "runs_plus/heartbeat_stress.csv"):
    """
    Writes:
      - runs_plus/heartbeat.json  -> {"heartbeat": {"YYYY-MM-DD": bpm, ...}}
      - runs_plus/heartbeat.png   -> line chart of bpm
    """
    cfg = HBConfig()
    vol = realized_vol(prices, cfg.window)
    bpm = map_vol_to_bpm(vol, cfg)
    scaler = bpm_to_exposure_scaler(bpm, cfg)
    stress = heartbeat_stress_from_bpm(bpm, cfg)
    _write_outputs(
        bpm=bpm,
        scaler=scaler,
        stress=stress,
        out_json=out_json,
        out_png=out_png,
        out_bpm_csv=out_bpm_csv,
        out_scaler_csv=out_scaler_csv,
        out_stress_csv=out_stress_csv,
        source="prices",
    )
    return out_json, out_png

def compute_heartbeat_from_returns(
    returns: pd.Series,
    out_json: str = "runs_plus/heartbeat.json",
    out_png: str = "runs_plus/heartbeat.png",
    out_bpm_csv: str = "runs_plus/heartbeat_bpm.csv",
    out_scaler_csv: str = "runs_plus/heartbeat_exposure_scaler.csv",
    out_stress_csv: str = "runs_plus/heartbeat_stress.csv",
):
    cfg = HBConfig()
    vol = realized_vol_from_returns(returns, cfg.window)
    bpm = map_vol_to_bpm(vol, cfg)
    scaler = bpm_to_exposure_scaler(bpm, cfg)
    stress = heartbeat_stress_from_bpm(bpm, cfg)
    _write_outputs(
        bpm=bpm,
        scaler=scaler,
        stress=stress,
        out_json=out_json,
        out_png=out_png,
        out_bpm_csv=out_bpm_csv,
        out_scaler_csv=out_scaler_csv,
        out_stress_csv=out_stress_csv,
        source="returns",
    )
    return out_json, out_png
