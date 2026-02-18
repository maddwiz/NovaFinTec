#!/usr/bin/env python3
"""
Adaptive signal decomposition council members.

Writes:
  - runs_plus/council_adaptive_trend.csv
  - runs_plus/council_adaptive_cycle.csv
  - runs_plus/adaptive_signal_composite.csv
  - runs_plus/adaptive_signal_info.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qengine.adaptive_signals import decompose_and_signal


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


def _target_shape() -> tuple[int, int]:
    for p in [
        RUNS / "asset_returns.csv",
        RUNS / "portfolio_weights_final.csv",
        RUNS / "portfolio_weights.csv",
        ROOT / "portfolio_weights.csv",
    ]:
        m = _load_matrix(p)
        if m is not None:
            return int(m.shape[0]), int(m.shape[1])
    return 0, 0


def _returns_to_price(r: np.ndarray) -> np.ndarray:
    x = np.asarray(r, float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    x = np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), -0.95, 0.95)
    return np.cumprod(1.0 + x, axis=0)


def main() -> int:
    arr = _load_matrix(RUNS / "asset_returns.csv")
    t, n = _target_shape()

    if arr is None and t <= 0:
        print("(!) Missing asset returns and target shape; skipping adaptive decomposition.")
        return 0

    if arr is None:
        trend = np.zeros((t, n), dtype=float)
        cycle = np.zeros((t, n), dtype=float)
        comp = np.zeros((t, n), dtype=float)
        info = {
            "ok": False,
            "reason": "missing_asset_returns",
            "rows": int(t),
            "cols": int(n),
        }
    else:
        px = _returns_to_price(arr)
        tt, nn = px.shape

        max_imfs = int(np.clip(int(float(os.getenv("Q_ADAPTIVE_SIGNAL_MAX_IMFS", "5"))), 2, 10))
        trend_thresh = int(np.clip(int(float(os.getenv("Q_ADAPTIVE_SIGNAL_TREND_PERIOD_THRESHOLD", "126"))), 21, 504))

        trend = np.zeros((tt, nn), dtype=float)
        cycle = np.zeros((tt, nn), dtype=float)
        comp = np.zeros((tt, nn), dtype=float)

        imf_counts = []
        trend_w = []
        cycle_w = []

        for j in range(nn):
            out = decompose_and_signal(px[:, j], max_imfs=max_imfs, trend_period_threshold=trend_thresh)
            trend[:, j] = np.asarray(out.get("trend_signal", np.zeros(tt)), float)[:tt]
            cycle[:, j] = np.asarray(out.get("cycle_signal", np.zeros(tt)), float)[:tt]
            comp[:, j] = np.asarray(out.get("composite", np.zeros(tt)), float)[:tt]
            imf_counts.append(int(out.get("imf_count", 0)))
            trend_w.append(float(out.get("trend_weight_mean", 0.5)))
            cycle_w.append(float(out.get("cycle_weight_mean", 0.5)))

        info = {
            "ok": True,
            "rows": int(tt),
            "cols": int(nn),
            "params": {
                "max_imfs": int(max_imfs),
                "trend_period_threshold": int(trend_thresh),
            },
            "imf_count_mean": float(np.mean(imf_counts)) if imf_counts else 0.0,
            "trend_weight_mean": float(np.mean(trend_w)) if trend_w else 0.0,
            "cycle_weight_mean": float(np.mean(cycle_w)) if cycle_w else 0.0,
            "trend_abs_mean": float(np.mean(np.abs(trend))) if trend.size else 0.0,
            "cycle_abs_mean": float(np.mean(np.abs(cycle))) if cycle.size else 0.0,
            "composite_abs_mean": float(np.mean(np.abs(comp))) if comp.size else 0.0,
        }

    if (t > 0 and n > 0) and (trend.shape != (t, n)):
        zt = np.zeros((t, n), dtype=float)
        zc = np.zeros((t, n), dtype=float)
        zm = np.zeros((t, n), dtype=float)
        L = min(t, trend.shape[0])
        K = min(n, trend.shape[1])
        zt[-L:, :K] = trend[-L:, :K]
        zc[-L:, :K] = cycle[-L:, :K]
        zm[-L:, :K] = comp[-L:, :K]
        trend, cycle, comp = zt, zc, zm

    trend = np.clip(np.nan_to_num(trend, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
    cycle = np.clip(np.nan_to_num(cycle, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
    comp = np.clip(np.nan_to_num(comp, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)

    np.savetxt(RUNS / "council_adaptive_trend.csv", trend, delimiter=",")
    np.savetxt(RUNS / "council_adaptive_cycle.csv", cycle, delimiter=",")
    np.savetxt(RUNS / "adaptive_signal_composite.csv", comp, delimiter=",")
    (RUNS / "adaptive_signal_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    _append_card(
        "Adaptive Signal Decomposition ✔",
        (
            f"<p>rows={trend.shape[0]}, cols={trend.shape[1]}, ok={bool(info.get('ok', False))}.</p>"
            f"<p>trend_abs_mean={float(np.mean(np.abs(trend))):.3f}, cycle_abs_mean={float(np.mean(np.abs(cycle))):.3f}</p>"
        ),
    )

    print(f"✅ Wrote {RUNS/'council_adaptive_trend.csv'}")
    print(f"✅ Wrote {RUNS/'council_adaptive_cycle.csv'}")
    print(f"✅ Wrote {RUNS/'adaptive_signal_composite.csv'}")
    print(f"✅ Wrote {RUNS/'adaptive_signal_info.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
