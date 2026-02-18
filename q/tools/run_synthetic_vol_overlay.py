#!/usr/bin/env python3
"""
Synthetic vol overlay from forecast-vs-realized volatility spread.

Writes:
  - runs_plus/synthetic_vol_overlay.csv
  - runs_plus/synthetic_vol_info.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


def _append_card(title: str, html: str) -> None:
    if str(os.getenv("Q_DISABLE_REPORT_CARDS", "0")).strip().lower() in {"1", "true", "yes", "on"}:
        return
    for name in ["report_all.html", "report_best_plus.html", "report_plus.html", "report.html"]:
        p = ROOT / name
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        txt = txt.replace("</body>", card + "</body>") if "</body>" in txt else txt + card
        p.write_text(txt, encoding="utf-8")


def _load_series(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        a = np.loadtxt(path, delimiter=",")
    except Exception:
        try:
            a = np.loadtxt(path, delimiter=",", skiprows=1)
        except Exception:
            return None
    a = np.asarray(a, float)
    if a.ndim == 2:
        a = a[:, -1]
    a = a.ravel()
    if a.size == 0:
        return None
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def _rolling_realized_vol(r: np.ndarray, lookback: int = 21) -> np.ndarray:
    s = pd.Series(np.asarray(r, float).ravel())
    rv = s.rolling(int(max(5, lookback)), min_periods=max(5, int(lookback // 2))).std(ddof=1) * np.sqrt(252.0)
    return rv.ffill().bfill().fillna(0.0).values.astype(float)


def _rolling_z(x: np.ndarray, w: int = 63) -> np.ndarray:
    s = pd.Series(np.asarray(x, float).ravel())
    mu = s.rolling(int(max(10, w)), min_periods=max(10, int(w // 3))).mean()
    sd = s.rolling(int(max(10, w)), min_periods=max(10, int(w // 3))).std(ddof=1).replace(0.0, np.nan)
    z = (s - mu) / (sd + 1e-12)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(float)


def main() -> int:
    daily = _load_series(RUNS / "daily_returns.csv")
    if daily is None:
        daily = _load_series(RUNS / "wf_oos_returns.csv")
    if daily is None:
        print("(!) Missing returns for synthetic vol overlay; skipping.")
        return 0

    fcast = _load_series(RUNS / "vol_forecast.csv")
    if fcast is None:
        print("(!) Missing runs_plus/vol_forecast.csv; run tools/run_vol_forecast.py first.")
        return 0

    t = min(len(daily), len(fcast))
    r = daily[-t:]
    fv = fcast[-t:]

    lookback = int(np.clip(int(float(os.getenv("Q_SYNTH_VOL_LOOKBACK", "21"))), 5, 252))
    z_thr = float(np.clip(float(os.getenv("Q_SYNTH_VOL_Z_THRESHOLD", "1.0")), 0.25, 3.0))
    strength = float(np.clip(float(os.getenv("Q_SYNTH_VOL_STRENGTH", "0.20")), 0.0, 2.0))
    floor = float(np.clip(float(os.getenv("Q_SYNTH_VOL_FLOOR", "0.75")), 0.25, 2.0))
    ceil = float(np.clip(float(os.getenv("Q_SYNTH_VOL_CEIL", "1.25")), floor, 3.0))

    rv = _rolling_realized_vol(r, lookback=lookback)
    spread = fv - rv
    z = _rolling_z(spread, w=63)

    direction = np.zeros_like(z)
    direction[z > z_thr] = 1.0
    direction[z < -z_thr] = -1.0

    # Positive direction => sell-vol posture => permit more directional exposure.
    overlay = 1.0 + strength * direction * np.clip(np.abs(z) / max(1e-9, z_thr), 0.0, 2.0)
    overlay = np.clip(overlay, floor, ceil)

    synthetic_pnl = direction * np.abs(rv - fv)

    np.savetxt(RUNS / "synthetic_vol_overlay.csv", overlay.reshape(-1, 1), delimiter=",")

    info = {
        "ok": True,
        "rows": int(t),
        "params": {
            "lookback": int(lookback),
            "z_threshold": float(z_thr),
            "strength": float(strength),
            "floor": float(floor),
            "ceil": float(ceil),
        },
        "forecast_vol_mean": float(np.mean(fv)),
        "realized_vol_mean": float(np.mean(rv)),
        "spread_mean": float(np.mean(spread)),
        "direction_nonzero_share": float(np.mean(direction != 0.0)),
        "synthetic_pnl_mean": float(np.mean(synthetic_pnl)),
        "synthetic_pnl_sharpe_like": float(np.mean(synthetic_pnl) / (np.std(synthetic_pnl, ddof=1) + 1e-12) * np.sqrt(252.0))
        if t > 2
        else 0.0,
        "overlay_mean": float(np.mean(overlay)),
        "overlay_min": float(np.min(overlay)),
        "overlay_max": float(np.max(overlay)),
    }
    (RUNS / "synthetic_vol_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    _append_card(
        "Synthetic Vol Overlay ✔",
        (
            f"<p>rows={t}, nonzero_signal_share={info['direction_nonzero_share']:.3f}</p>"
            f"<p>overlay_mean={info['overlay_mean']:.3f}, range=[{info['overlay_min']:.3f},{info['overlay_max']:.3f}]</p>"
        ),
    )

    print(f"✅ Wrote {RUNS/'synthetic_vol_overlay.csv'}")
    print(f"✅ Wrote {RUNS/'synthetic_vol_info.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
