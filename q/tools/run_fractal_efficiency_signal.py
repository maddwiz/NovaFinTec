#!/usr/bin/env python3
"""
Fractal-efficiency council signal.

Writes:
  - runs_plus/council_fractal_efficiency.csv
  - runs_plus/fractal_efficiency_info.json
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


def fractal_efficiency_signal(close: np.ndarray, lookback: int = 21) -> np.ndarray:
    c = np.asarray(close, float).ravel()
    n = len(c)
    er = np.full(n, 0.5, dtype=float)
    lb = int(max(5, lookback))

    for t in range(lb, n):
        w = c[t - lb : t + 1]
        net_move = abs(float(w[-1] - w[0]))
        path_length = float(np.sum(np.abs(np.diff(w))) + 1e-12)
        er[t] = net_move / path_length

    s = pd.Series(er)
    er_change = s.diff(5)
    mu = er_change.rolling(63, min_periods=20).mean()
    sd = er_change.rolling(63, min_periods=20).std(ddof=1).replace(0.0, np.nan)
    z = ((er_change - mu) / (sd + 1e-12)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return np.clip(z.values.astype(float), -3.0, 3.0)


def _returns_to_price(r: np.ndarray) -> np.ndarray:
    x = np.asarray(r, float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    x = np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), -0.95, 0.95)
    return np.cumprod(1.0 + x, axis=0)


def main() -> int:
    arr = _load_matrix(RUNS / "asset_returns.csv")
    if arr is None:
        print("(!) Missing runs_plus/asset_returns.csv; skipping fractal efficiency signal.")
        return 0

    px = _returns_to_price(arr)
    t, n = px.shape
    lookback = int(np.clip(int(float(os.getenv("Q_FRACTAL_EFF_LOOKBACK", "21"))), 5, 126))
    smooth_alpha = float(np.clip(float(os.getenv("Q_FRACTAL_EFF_SMOOTH_ALPHA", "0.20")), 0.0, 0.95))

    out = np.zeros((t, n), dtype=float)
    for j in range(n):
        sig = fractal_efficiency_signal(px[:, j], lookback=lookback)
        out[:, j] = np.tanh(sig)

    if smooth_alpha > 0.0:
        sm = out.copy()
        for j in range(n):
            for i in range(1, t):
                sm[i, j] = smooth_alpha * sm[i, j] + (1.0 - smooth_alpha) * sm[i - 1, j]
        out = sm

    out = np.clip(out, -1.0, 1.0)
    np.savetxt(RUNS / "council_fractal_efficiency.csv", out, delimiter=",")

    info = {
        "ok": True,
        "rows": int(t),
        "cols": int(n),
        "params": {"lookback": int(lookback), "smooth_alpha": float(smooth_alpha)},
        "signal_mean": float(np.mean(out)),
        "signal_abs_mean": float(np.mean(np.abs(out))),
        "signal_std": float(np.std(out, ddof=1)) if out.size > 1 else 0.0,
    }
    (RUNS / "fractal_efficiency_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    _append_card(
        "Fractal Efficiency Council ✔",
        (
            f"<p>rows={t}, cols={n}, abs_mean={info['signal_abs_mean']:.3f}</p>"
            f"<p>std={info['signal_std']:.3f}</p>"
        ),
    )

    print(f"✅ Wrote {RUNS/'council_fractal_efficiency.csv'}")
    print(f"✅ Wrote {RUNS/'fractal_efficiency_info.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
