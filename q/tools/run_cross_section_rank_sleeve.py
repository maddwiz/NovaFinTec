#!/usr/bin/env python3
"""
Cross-sectional rank sleeve.

Builds a neutral long/short sleeve from asset-return ranks.

Writes:
  - runs_plus/weights_rank_sleeve.csv
  - runs_plus/rank_sleeve_info.json
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
    return a


def _rolling_sum(x: np.ndarray, win: int) -> np.ndarray:
    arr = np.asarray(x, float)
    T, N = arr.shape
    out = np.zeros((T, N), float)
    w = int(max(1, win))
    c = np.cumsum(arr, axis=0)
    out[:w] = c[:w]
    if T > w:
        out[w:] = c[w:] - c[:-w]
    return out


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


def _cs_zscore_rowwise(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, float)
    mu = np.mean(arr, axis=1, keepdims=True)
    sd = np.std(arr, axis=1, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    z = (arr - mu) / sd
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def build_rank_sleeve(
    asset_returns: np.ndarray,
    *,
    mom_short: int = 20,
    mom_long: int = 63,
    rev_short: int = 5,
    gross_target: float = 1.0,
    per_asset_cap: float = 0.10,
) -> np.ndarray:
    r = np.asarray(asset_returns, float)
    T, N = r.shape
    ms = _rolling_sum(r, mom_short)
    ml = _rolling_sum(r, mom_long)
    rv = -_rolling_sum(r, rev_short)
    vol = _rolling_std(r, mom_short)
    vol = np.clip(vol, 1e-6, None)

    score = (0.50 * ms + 0.35 * ml + 0.15 * rv) / vol
    z = _cs_zscore_rowwise(score)

    # Market-neutral cross-sectional sleeve.
    z = z - np.mean(z, axis=1, keepdims=True)
    w = z.copy()
    l1 = np.sum(np.abs(w), axis=1, keepdims=True)
    l1 = np.where(l1 < 1e-8, 1.0, l1)
    gt = float(max(0.0, gross_target))
    w = gt * (w / l1)

    def _renorm_neutral(arr: np.ndarray) -> np.ndarray:
        arr = arr - np.mean(arr, axis=1, keepdims=True)
        l1n = np.sum(np.abs(arr), axis=1, keepdims=True)
        l1n = np.where(l1n < 1e-8, 1.0, l1n)
        return gt * (arr / l1n)

    cap = float(max(0.0, per_asset_cap))
    if cap > 0:
        # Clipping can introduce net directional bias; re-center and re-normalize.
        w = np.clip(w, -cap, cap)
        w = _renorm_neutral(w)
        # One more clip + neutralization pass keeps weights stable and near-capped.
        w = np.clip(w, -cap, cap)
        w = _renorm_neutral(w)
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
        print("(!) Missing runs_plus/asset_returns.csv; skipping rank sleeve.")
        return 0
    gross = float(np.clip(float(os.getenv("Q_RANK_SLEEVE_GROSS", "1.00")), 0.0, 2.5))
    cap = float(np.clip(float(os.getenv("Q_RANK_SLEEVE_CAP", "0.10")), 0.0, 1.0))
    mom_s = int(np.clip(int(float(os.getenv("Q_RANK_MOM_SHORT", "20"))), 5, 120))
    mom_l = int(np.clip(int(float(os.getenv("Q_RANK_MOM_LONG", "63"))), 10, 252))
    rev_s = int(np.clip(int(float(os.getenv("Q_RANK_REV_SHORT", "5"))), 2, 30))

    w = build_rank_sleeve(
        ar,
        mom_short=mom_s,
        mom_long=mom_l,
        rev_short=rev_s,
        gross_target=gross,
        per_asset_cap=cap,
    )
    np.savetxt(RUNS / "weights_rank_sleeve.csv", w, delimiter=",")

    gross_mean = float(np.mean(np.sum(np.abs(w), axis=1))) if w.size else 0.0
    net_mean = float(np.mean(np.sum(w, axis=1))) if w.size else 0.0
    info = {
        "rows": int(w.shape[0]),
        "assets": int(w.shape[1]),
        "mom_short": int(mom_s),
        "mom_long": int(mom_l),
        "rev_short": int(rev_s),
        "gross_target": float(gross),
        "per_asset_cap": float(cap),
        "gross_mean": gross_mean,
        "net_mean": net_mean,
    }
    (RUNS / "rank_sleeve_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    html = (
        f"<p>Rank sleeve built (T={w.shape[0]}, N={w.shape[1]}), gross≈{gross_mean:.3f}, net≈{net_mean:.3f}.</p>"
        f"<p>Lookbacks: mom({mom_s},{mom_l}), rev({rev_s}), cap={cap:.2f}.</p>"
    )
    append_card("Cross-Section Rank Sleeve ✔", html)

    print(f"✅ Wrote {RUNS/'weights_rank_sleeve.csv'}")
    print(f"✅ Wrote {RUNS/'rank_sleeve_info.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
