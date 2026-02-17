#!/usr/bin/env python3
"""
Cross-sectional relative momentum overlay.

Builds a market-wide scalar from cross-sectional momentum breadth and spread
across the current asset universe.

Writes:
  - runs_plus/cross_sectional_momentum_signal.csv   (signed signal in [-1, 1])
  - runs_plus/cross_sectional_momentum_overlay.csv  (exposure scalar around 1.0)
  - runs_plus/cross_sectional_momentum_info.json
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


def _load_returns() -> np.ndarray | None:
    p = RUNS / "asset_returns.csv"
    if not p.exists():
        return None
    try:
        a = np.loadtxt(p, delimiter=",")
    except Exception:
        try:
            a = np.loadtxt(p, delimiter=",", skiprows=1)
        except Exception:
            return None
    a = np.asarray(a, float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2 or a.size == 0:
        return None
    return a


def _roll_z(s: pd.Series, lookback: int, minp: int) -> np.ndarray:
    if s.empty:
        return np.zeros(0, dtype=float)
    mu = s.rolling(lookback, min_periods=minp).mean()
    sd = s.rolling(lookback, min_periods=minp).std(ddof=1).replace(0.0, np.nan)
    z = (s - mu) / (sd + 1e-12)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)


def main() -> int:
    arr = _load_returns()
    if arr is None:
        print("(!) Missing runs_plus/asset_returns.csv; skipping.")
        return 0

    t, n = arr.shape
    min_assets = int(np.clip(int(float(os.getenv("Q_CSM_MIN_ASSETS", "15"))), 5, 5000))
    short_lb = int(np.clip(int(float(os.getenv("Q_CSM_SHORT_LOOKBACK", "20"))), 5, 252))
    long_lb = int(np.clip(int(float(os.getenv("Q_CSM_LONG_LOOKBACK", "60"))), max(10, short_lb), 756))
    z_lb = int(np.clip(int(float(os.getenv("Q_CSM_Z_LOOKBACK", "126"))), 20, 756))
    smooth_alpha = float(np.clip(float(os.getenv("Q_CSM_SMOOTH_ALPHA", "0.18")), 0.01, 0.90))
    beta = float(np.clip(float(os.getenv("Q_CSM_BETA", "0.16")), 0.0, 1.2))
    floor = float(np.clip(float(os.getenv("Q_CSM_FLOOR", "0.82")), 0.2, 1.2))
    ceil = float(np.clip(float(os.getenv("Q_CSM_CEIL", "1.18")), floor, 1.8))

    if (n < min_assets) or (t < max(40, long_lb + 5)):
        sig = np.zeros(t, dtype=float)
        ov = np.ones(t, dtype=float)
        np.savetxt(RUNS / "cross_sectional_momentum_signal.csv", sig, delimiter=",")
        np.savetxt(RUNS / "cross_sectional_momentum_overlay.csv", ov, delimiter=",")
        (RUNS / "cross_sectional_momentum_info.json").write_text(
            json.dumps(
                {
                    "ok": False,
                    "reason": "insufficient_shape",
                    "rows": int(t),
                    "assets": int(n),
                    "min_assets": int(min_assets),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return 0

    df = pd.DataFrame(arr)
    mom_short = df.rolling(short_lb, min_periods=max(5, short_lb // 3)).sum().to_numpy(dtype=float)
    mom_long = df.rolling(long_lb, min_periods=max(10, long_lb // 3)).sum().to_numpy(dtype=float)
    mom = 0.65 * mom_short + 0.35 * mom_long

    breadth = np.zeros(t, dtype=float)
    spread = np.zeros(t, dtype=float)
    dispersion = np.zeros(t, dtype=float)
    valid_rows = 0

    for i in range(t):
        row = np.asarray(mom[i, :], float)
        mask = np.isfinite(row)
        if int(np.sum(mask)) < min_assets:
            continue
        z = row[mask]
        mu = float(np.mean(z))
        sd = float(np.std(z, ddof=1))
        if not np.isfinite(sd) or sd <= 1e-12:
            continue
        z = (z - mu) / (sd + 1e-12)
        breadth[i] = float(np.mean(z > 0.0) - 0.5)
        q = max(1, int(round(0.20 * len(z))))
        zs = np.sort(z)
        spread[i] = float(np.mean(zs[-q:]) - np.mean(zs[:q]))
        dispersion[i] = float(np.std(z, ddof=1))
        valid_rows += 1

    z_breadth = _roll_z(pd.Series(breadth), lookback=z_lb, minp=max(20, z_lb // 4))
    z_spread = _roll_z(pd.Series(spread), lookback=z_lb, minp=max(20, z_lb // 4))
    z_disp = _roll_z(pd.Series(dispersion), lookback=z_lb, minp=max(20, z_lb // 4))

    raw = np.clip(0.95 * z_breadth + 0.70 * z_spread - 0.35 * np.maximum(0.0, z_disp), -6.0, 6.0)
    sig = np.tanh(raw / 2.0)
    sig = pd.Series(sig).ewm(alpha=smooth_alpha, adjust=False).mean().clip(-1.0, 1.0).to_numpy(dtype=float)
    ov = np.clip(1.0 + beta * sig, floor, ceil)

    np.savetxt(RUNS / "cross_sectional_momentum_signal.csv", sig, delimiter=",")
    np.savetxt(RUNS / "cross_sectional_momentum_overlay.csv", ov, delimiter=",")

    info = {
        "ok": True,
        "rows": int(t),
        "assets": int(n),
        "valid_rows": int(valid_rows),
        "valid_row_share": float(valid_rows / max(1, t)),
        "params": {
            "min_assets": int(min_assets),
            "short_lookback": int(short_lb),
            "long_lookback": int(long_lb),
            "z_lookback": int(z_lb),
            "smooth_alpha": float(smooth_alpha),
            "beta": float(beta),
            "floor": float(floor),
            "ceil": float(ceil),
        },
        "signal_mean": float(np.mean(sig)),
        "signal_min": float(np.min(sig)),
        "signal_max": float(np.max(sig)),
        "overlay_mean": float(np.mean(ov)),
        "overlay_min": float(np.min(ov)),
        "overlay_max": float(np.max(ov)),
    }
    (RUNS / "cross_sectional_momentum_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    _append_card(
        "Cross-Sectional Momentum Overlay ✔",
        (
            f"<p>rows={t}, assets={n}, valid_share={info['valid_row_share']:.2f}, "
            f"signal_mean={info['signal_mean']:.3f}, overlay_range=[{info['overlay_min']:.3f},{info['overlay_max']:.3f}].</p>"
        ),
    )
    print(f"✅ Wrote {RUNS/'cross_sectional_momentum_signal.csv'}")
    print(f"✅ Wrote {RUNS/'cross_sectional_momentum_overlay.csv'}")
    print(f"✅ Wrote {RUNS/'cross_sectional_momentum_info.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
