#!/usr/bin/env python3
"""
Entropy-based conviction sizing scalar from council vote distribution.

Writes:
  - runs_plus/entropy_sizing_scalar.csv
  - runs_plus/entropy_sizing_info.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

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


def main() -> int:
    cv = _load_matrix(RUNS / "council_votes.csv")
    if cv is None:
        print("(!) Missing runs_plus/council_votes.csv; skipping entropy sizing.")
        return 0

    floor = float(np.clip(float(os.getenv("Q_ENTROPY_SIZING_FLOOR", "0.30")), 0.01, 2.0))
    ceil = float(np.clip(float(os.getenv("Q_ENTROPY_SIZING_CEIL", "1.50")), floor, 3.0))

    av = np.abs(cv)
    w = av / np.maximum(np.sum(av, axis=1, keepdims=True), 1e-12)
    H = -np.sum(w * np.log(np.clip(w, 1e-12, None)), axis=1)
    Hmax = np.log(max(2, cv.shape[1]))
    Hn = H / max(Hmax, 1e-12)

    scalar = np.clip(1.5 - Hn, floor, ceil)

    np.savetxt(RUNS / "entropy_sizing_scalar.csv", scalar.reshape(-1, 1), delimiter=",")

    info = {
        "ok": True,
        "rows": int(cv.shape[0]),
        "members": int(cv.shape[1]),
        "floor": float(floor),
        "ceil": float(ceil),
        "entropy_mean": float(np.mean(Hn)),
        "entropy_min": float(np.min(Hn)),
        "entropy_max": float(np.max(Hn)),
        "scalar_mean": float(np.mean(scalar)),
        "scalar_min": float(np.min(scalar)),
        "scalar_max": float(np.max(scalar)),
    }
    (RUNS / "entropy_sizing_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    _append_card(
        "Entropy Sizing ✔",
        (
            f"<p>rows={cv.shape[0]}, members={cv.shape[1]}, entropy_mean={info['entropy_mean']:.3f}</p>"
            f"<p>scalar_mean={info['scalar_mean']:.3f}, range=[{info['scalar_min']:.3f},{info['scalar_max']:.3f}]</p>"
        ),
    )

    print(f"✅ Wrote {RUNS/'entropy_sizing_scalar.csv'}")
    print(f"✅ Wrote {RUNS/'entropy_sizing_info.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
