#!/usr/bin/env python3
"""
Forward-looking volatility forecast overlay (HAR-RV).

Writes:
  - runs_plus/vol_forecast.csv
  - runs_plus/vol_forecast_overlay.csv
  - runs_plus/vol_forecast_info.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qengine.vol_forecast import forecast_vol


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
    return rv.fillna(method="ffill").fillna(method="bfill").fillna(0.0).values.astype(float)


def _zscore_rolling(x: np.ndarray, window: int = 63) -> np.ndarray:
    s = pd.Series(np.asarray(x, float).ravel())
    w = int(max(10, window))
    mu = s.rolling(w, min_periods=max(10, w // 3)).mean()
    sd = s.rolling(w, min_periods=max(10, w // 3)).std(ddof=1).replace(0.0, np.nan)
    z = (s - mu) / (sd + 1e-12)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(float)


def main() -> int:
    r = _load_series(RUNS / "daily_returns.csv")
    if r is None:
        r = _load_series(RUNS / "wf_oos_returns.csv")
    if r is None:
        print("(!) Missing returns for vol forecast overlay; skipping.")
        return 0

    train_window = int(np.clip(int(float(os.getenv("Q_VOL_FORECAST_TRAIN_WINDOW", "504"))), 126, max(126, len(r))))
    step = int(np.clip(int(float(os.getenv("Q_VOL_FORECAST_STEP", "21"))), 1, 126))
    lookback = int(np.clip(int(float(os.getenv("Q_VOL_FORECAST_REALIZED_LOOKBACK", "21"))), 5, 252))
    strength = float(np.clip(float(os.getenv("Q_VOL_FORECAST_OVERLAY_STRENGTH", "0.18")), 0.0, 1.5))
    floor = float(np.clip(float(os.getenv("Q_VOL_FORECAST_OVERLAY_FLOOR", "0.75")), 0.25, 2.0))
    ceil = float(np.clip(float(os.getenv("Q_VOL_FORECAST_OVERLAY_CEIL", "1.25")), floor, 3.0))

    fcast = forecast_vol(r, train_window=train_window, step=step)
    realized = _rolling_realized_vol(r, lookback=lookback)
    surprise = np.nan_to_num(fcast - realized, nan=0.0, posinf=0.0, neginf=0.0)
    z = _zscore_rolling(surprise, window=63)

    overlay = 1.0 + strength * np.tanh(z)
    overlay = np.clip(overlay, floor, ceil)

    np.savetxt(RUNS / "vol_forecast.csv", fcast.reshape(-1, 1), delimiter=",")
    np.savetxt(RUNS / "vol_forecast_overlay.csv", overlay.reshape(-1, 1), delimiter=",")

    info = {
        "ok": True,
        "rows": int(len(r)),
        "params": {
            "train_window": int(train_window),
            "step": int(step),
            "realized_lookback": int(lookback),
            "strength": float(strength),
            "floor": float(floor),
            "ceil": float(ceil),
        },
        "forecast_mean": float(np.mean(fcast)),
        "realized_mean": float(np.mean(realized)),
        "surprise_mean": float(np.mean(surprise)),
        "overlay_mean": float(np.mean(overlay)),
        "overlay_min": float(np.min(overlay)),
        "overlay_max": float(np.max(overlay)),
    }
    (RUNS / "vol_forecast_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    _append_card(
        "Vol Forecast Overlay ✔",
        (
            f"<p>rows={len(r)}, forecast_mean={np.mean(fcast):.3f}, realized_mean={np.mean(realized):.3f}</p>"
            f"<p>overlay_mean={np.mean(overlay):.3f}, range=[{np.min(overlay):.3f},{np.max(overlay):.3f}]</p>"
        ),
    )

    print(f"✅ Wrote {RUNS/'vol_forecast.csv'}")
    print(f"✅ Wrote {RUNS/'vol_forecast_overlay.csv'}")
    print(f"✅ Wrote {RUNS/'vol_forecast_info.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
