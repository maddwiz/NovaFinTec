#!/usr/bin/env python3
"""
Low-vol anomaly sleeve (cross-sectional, market-neutral).

Writes:
  - runs_plus/weights_low_vol_sleeve.csv
  - runs_plus/low_vol_sleeve_info.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


def _load_matrix(path: Path) -> np.ndarray | None:
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
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.size == 0:
        return None
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def _rolling_std(x: np.ndarray, win: int) -> np.ndarray:
    arr = np.asarray(x, float)
    T, N = arr.shape
    out = np.zeros((T, N), float)
    w = int(max(2, win))
    for t in range(T):
        lo = max(0, t - w + 1)
        seg = arr[lo : t + 1]
        out[t] = np.std(seg, axis=0) if seg.size else 0.0
    return out


def _rolling_downside_std(x: np.ndarray, win: int) -> np.ndarray:
    arr = np.asarray(x, float)
    T, N = arr.shape
    out = np.zeros((T, N), float)
    w = int(max(2, win))
    for t in range(T):
        lo = max(0, t - w + 1)
        seg = arr[lo : t + 1]
        seg = np.minimum(seg, 0.0)
        out[t] = np.std(seg, axis=0) if seg.size else 0.0
    return out


def _cs_zscore_rowwise(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, float)
    mu = np.mean(arr, axis=1, keepdims=True)
    sd = np.std(arr, axis=1, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return np.nan_to_num((arr - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)


def build_low_vol_sleeve(
    asset_returns: np.ndarray,
    *,
    lookback: int = 63,
    downside_weight: float = 0.35,
    gross_target: float = 1.0,
    per_asset_cap: float = 0.10,
) -> np.ndarray:
    r = np.asarray(asset_returns, float)
    vol = _rolling_std(r, lookback)
    dvol = _rolling_downside_std(r, lookback)
    dw = float(np.clip(float(downside_weight), 0.0, 1.0))
    score = -(1.0 - dw) * vol - dw * dvol
    z = _cs_zscore_rowwise(score)
    z = z - np.mean(z, axis=1, keepdims=True)

    gt = float(max(0.0, gross_target))
    w = z.copy()
    l1 = np.sum(np.abs(w), axis=1, keepdims=True)
    l1 = np.where(l1 < 1e-8, 1.0, l1)
    w = gt * (w / l1)

    cap = float(max(0.0, per_asset_cap))
    if cap > 0:
        w = np.clip(w, -cap, cap)
        w = w - np.mean(w, axis=1, keepdims=True)
        l1b = np.sum(np.abs(w), axis=1, keepdims=True)
        l1b = np.where(l1b < 1e-8, 1.0, l1b)
        w = gt * (w / l1b)
    return np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)


def append_card(title: str, html: str) -> None:
    for name in ["report_all.html", "report_best_plus.html", "report_plus.html", "report.html"]:
        p = ROOT / name
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        txt = txt.replace("</body>", card + "</body>") if "</body>" in txt else txt + card
        p.write_text(txt, encoding="utf-8")


def main() -> int:
    ar = _load_matrix(RUNS / "asset_returns.csv")
    if ar is None:
        print("(!) Missing runs_plus/asset_returns.csv; skipping low-vol sleeve.")
        return 0

    lookback = int(np.clip(int(float(os.getenv("Q_LOWVOL_LOOKBACK", "63"))), 5, 252))
    downside_weight = float(np.clip(float(os.getenv("Q_LOWVOL_DOWNSIDE_WEIGHT", "0.35")), 0.0, 1.0))
    gross = float(np.clip(float(os.getenv("Q_LOWVOL_GROSS", "1.00")), 0.0, 2.5))
    cap = float(np.clip(float(os.getenv("Q_LOWVOL_CAP", "0.10")), 0.0, 1.0))

    w = build_low_vol_sleeve(
        ar,
        lookback=lookback,
        downside_weight=downside_weight,
        gross_target=gross,
        per_asset_cap=cap,
    )
    np.savetxt(RUNS / "weights_low_vol_sleeve.csv", w, delimiter=",")

    info = {
        "rows": int(w.shape[0]),
        "assets": int(w.shape[1]),
        "lookback": int(lookback),
        "downside_weight": float(downside_weight),
        "gross_target": float(gross),
        "per_asset_cap": float(cap),
        "gross_mean": float(np.mean(np.sum(np.abs(w), axis=1))) if w.size else 0.0,
        "net_mean": float(np.mean(np.sum(w, axis=1))) if w.size else 0.0,
    }
    (RUNS / "low_vol_sleeve_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    html = (
        f"<p>Low-vol sleeve built (T={w.shape[0]}, N={w.shape[1]}), "
        f"gross≈{info['gross_mean']:.3f}, net≈{info['net_mean']:.3f}.</p>"
        f"<p>lookback={lookback}, downside_weight={downside_weight:.2f}, cap={cap:.2f}.</p>"
    )
    append_card("Low-Vol Sleeve ✔", html)

    print(f"✅ Wrote {RUNS/'weights_low_vol_sleeve.csv'}")
    print(f"✅ Wrote {RUNS/'low_vol_sleeve_info.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
