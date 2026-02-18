#!/usr/bin/env python3
"""
Cross-asset lead-lag network council member.

Writes:
  - runs_plus/council_leadlag_network.csv
  - runs_plus/leadlag_network_info.json
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


def _rolling_corr_score(x: np.ndarray, y: np.ndarray, window: int, lag: int = 1) -> np.ndarray:
    xv = np.asarray(x, float).ravel()
    yv = np.asarray(y, float).ravel()
    n = min(len(xv), len(yv))
    out = np.zeros(n, dtype=float)
    if n < window + lag + 2:
        return out

    xs = pd.Series(xv)
    ys = pd.Series(yv)

    x_lag = xs.shift(int(lag))
    corr = x_lag.rolling(int(window), min_periods=max(20, int(window // 3))).corr(ys)
    cov = x_lag.rolling(int(window), min_periods=max(20, int(window // 3))).cov(ys)

    c = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    cv = cov.replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    score = np.sign(cv) * np.sqrt(np.clip(np.abs(c), 0.0, 1.0))
    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


def _load_asset_names(n: int) -> list[str]:
    p = RUNS / "asset_names.csv"
    names: list[str] = []
    if p.exists():
        try:
            df = pd.read_csv(p)
            if not df.empty:
                col = None
                for c in df.columns:
                    if str(c).strip().lower() in {"asset", "symbol", "ticker", "name"}:
                        col = c
                        break
                if col is None:
                    col = df.columns[0]
                names = [str(x).strip().upper() for x in df[col].tolist() if str(x).strip()]
        except Exception:
            names = []
    if len(names) < n:
        names.extend([f"ASSET_{i+1}" for i in range(len(names), n)])
    return names[:n]


def main() -> int:
    arr = _load_matrix(RUNS / "asset_returns.csv")
    if arr is None:
        print("(!) Missing runs_plus/asset_returns.csv; skipping lead-lag network.")
        return 0

    r = np.asarray(arr, float)
    t, n = r.shape

    window = int(np.clip(int(float(os.getenv("Q_LEADLAG_WINDOW", "126"))), 30, max(30, t)))
    lag = int(np.clip(int(float(os.getenv("Q_LEADLAG_LAG", "1"))), 1, 5))
    smooth_alpha = float(np.clip(float(os.getenv("Q_LEADLAG_SMOOTH_ALPHA", "0.15")), 0.0, 0.95))

    pair_score = np.zeros((t, n, n), dtype=float)
    mean_abs_pair = np.zeros((n, n), dtype=float)

    for j in range(n):
        for i in range(n):
            if i == j:
                continue
            s = _rolling_corr_score(r[:, j], r[:, i], window=window, lag=lag)
            pair_score[:, j, i] = s
            mean_abs_pair[j, i] = float(np.mean(np.abs(s)))

    lead = np.zeros((t, n), dtype=float)
    r_sign = np.sign(r)
    for i in range(n):
        contrib = []
        for j in range(n):
            if i == j:
                continue
            c = pair_score[:, j, i] * np.roll(r_sign[:, j], 1)
            c[0] = 0.0
            contrib.append(c)
        if contrib:
            lead[:, i] = np.mean(np.column_stack(contrib), axis=1)

    if smooth_alpha > 0.0:
        out = lead.copy()
        for j in range(n):
            for k in range(1, t):
                out[k, j] = smooth_alpha * out[k, j] + (1.0 - smooth_alpha) * out[k - 1, j]
        lead = out

    z = np.zeros_like(lead)
    for i in range(n):
        s = pd.Series(lead[:, i])
        mu = s.rolling(63, min_periods=20).mean()
        sd = s.rolling(63, min_periods=20).std(ddof=1).replace(0.0, np.nan)
        z[:, i] = ((s - mu) / (sd + 1e-12)).replace([np.inf, -np.inf], np.nan).fillna(0.0).values

    signal = np.clip(np.tanh(z), -1.0, 1.0)

    names = _load_asset_names(n)
    tri = []
    for j in range(n):
        for i in range(n):
            if i == j:
                continue
            tri.append((mean_abs_pair[j, i], names[j], names[i]))
    tri.sort(key=lambda x: x[0], reverse=True)

    info = {
        "ok": True,
        "rows": int(t),
        "cols": int(n),
        "params": {"window": int(window), "lag": int(lag), "smooth_alpha": float(smooth_alpha)},
        "signal_mean": float(np.mean(signal)),
        "signal_abs_mean": float(np.mean(np.abs(signal))),
        "network_density": float(np.mean(np.abs(pair_score) > 0.10)),
        "top_pairs": [
            {"leader": a, "follower": b, "score": float(v)} for v, a, b in tri[:20]
        ],
    }

    np.savetxt(RUNS / "council_leadlag_network.csv", signal, delimiter=",")
    (RUNS / "leadlag_network_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    _append_card(
        "Lead-Lag Network Council ✔",
        (
            f"<p>rows={t}, cols={n}, density={info['network_density']:.3f}</p>"
            f"<p>signal_abs_mean={info['signal_abs_mean']:.3f}</p>"
        ),
    )

    print(f"✅ Wrote {RUNS/'council_leadlag_network.csv'}")
    print(f"✅ Wrote {RUNS/'leadlag_network_info.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
