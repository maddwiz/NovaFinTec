#!/usr/bin/env python3
"""
Uncertainty-aware sizing scalar.

Combines calibrated confidence, disagreement, and meta-execution uncertainty
into a smooth scalar applied to exposure.

Writes:
  - runs_plus/uncertainty_size_scalar.csv
  - runs_plus/uncertainty_sizing_info.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


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


def _align(vec: np.ndarray | None, T: int, fill: float) -> np.ndarray:
    if vec is None:
        return np.full(T, float(fill), float)
    v = np.asarray(vec, float).ravel()
    if len(v) >= T:
        return v[:T]
    out = np.full(T, float(fill), float)
    out[: len(v)] = v
    return out


def build_uncertainty_scalar(
    T: int,
    *,
    conf_cal: np.ndarray | None,
    conf_raw: np.ndarray | None,
    disagreement_gate: np.ndarray | None,
    meta_exec_prob: np.ndarray | None,
    shock_mask: np.ndarray | None,
    macro_shock: np.ndarray | None = None,
    beta: float = 0.45,
    shock_penalty: float = 0.20,
    macro_shock_blend: float = 0.0,
    floor: float = 0.55,
    ceiling: float = 1.10,
) -> dict:
    ccal = np.clip(_align(conf_cal, T, 0.5), 0.0, 1.0)
    craw = np.clip(_align(conf_raw, T, 0.5), 0.0, 1.0)
    dgate = np.clip(_align(disagreement_gate, T, 0.7), 0.0, 1.0)
    mprob = np.clip(_align(meta_exec_prob, T, 0.5), 0.0, 1.0)
    shock = np.clip(_align(shock_mask, T, 0.0), 0.0, 1.0)
    mshock = np.clip(_align(macro_shock, T, 0.0), 0.0, 1.0)
    mblend = float(np.clip(macro_shock_blend, 0.0, 1.0))
    shock_eff = np.clip((1.0 - mblend) * shock + mblend * mshock, 0.0, 1.0)

    meta_unc = 1.0 - (2.0 * np.abs(mprob - 0.5))  # 0 confident, 1 uncertain
    conf = np.clip(0.45 * ccal + 0.20 * craw + 0.20 * dgate + 0.15 * (1.0 - meta_unc), 0.0, 1.0)
    unc = np.clip(1.0 - conf, 0.0, 1.0)

    b = float(np.clip(beta, 0.0, 1.5))
    sp = float(np.clip(shock_penalty, 0.0, 1.0))
    scalar = 1.0 - b * unc - sp * shock_eff
    scalar = np.clip(scalar, float(floor), float(ceiling))
    return {
        "scalar": scalar.astype(float),
        "confidence": conf.astype(float),
        "uncertainty": unc.astype(float),
        "shock_effective": shock_eff.astype(float),
    }


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
    # Determine length from daily returns or fallback artifacts.
    r = _load_series(RUNS / "daily_returns.csv")
    if r is None:
        g = _load_series(RUNS / "global_governor.csv")
        if g is None:
            print("(!) Missing base series for uncertainty sizing; skipping.")
            return 0
        T = len(g)
    else:
        T = len(r)

    beta = float(np.clip(float(os.getenv("Q_UNCERTAINTY_BETA", "0.45")), 0.0, 1.5))
    shock_pen = float(np.clip(float(os.getenv("Q_UNCERTAINTY_SHOCK_PENALTY", "0.20")), 0.0, 1.0))
    macro_blend = float(np.clip(float(os.getenv("Q_UNCERTAINTY_MACRO_SHOCK_BLEND", "0.00")), 0.0, 1.0))
    floor = float(np.clip(float(os.getenv("Q_UNCERTAINTY_FLOOR", "0.55")), 0.10, 1.5))
    ceil = float(np.clip(float(os.getenv("Q_UNCERTAINTY_CEIL", "1.10")), floor, 1.8))

    out = build_uncertainty_scalar(
        T,
        conf_cal=_load_series(RUNS / "meta_mix_confidence_calibrated.csv"),
        conf_raw=_load_series(RUNS / "meta_mix_confidence_raw.csv"),
        disagreement_gate=_load_series(RUNS / "disagreement_gate.csv"),
        meta_exec_prob=_load_series(RUNS / "meta_execution_prob.csv"),
        shock_mask=_load_series(RUNS / "shock_mask.csv"),
        macro_shock=_load_series(RUNS / "macro_shock_proxy.csv"),
        beta=beta,
        shock_penalty=shock_pen,
        macro_shock_blend=macro_blend,
        floor=floor,
        ceiling=ceil,
    )

    np.savetxt(RUNS / "uncertainty_size_scalar.csv", out["scalar"], delimiter=",")
    info = {
        "rows": int(T),
        "beta": float(beta),
        "shock_penalty": float(shock_pen),
        "macro_shock_blend": float(macro_blend),
        "floor": float(floor),
        "ceiling": float(ceil),
        "scalar_mean": float(np.mean(out["scalar"])),
        "scalar_min": float(np.min(out["scalar"])),
        "scalar_max": float(np.max(out["scalar"])),
        "confidence_mean": float(np.mean(out["confidence"])),
        "uncertainty_mean": float(np.mean(out["uncertainty"])),
        "shock_effective_mean": float(np.mean(out["shock_effective"])),
    }
    (RUNS / "uncertainty_sizing_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    html = (
        f"<p>Uncertainty sizing scalar built (rows={T}): mean={info['scalar_mean']:.3f}, "
        f"range=[{info['scalar_min']:.3f},{info['scalar_max']:.3f}].</p>"
        f"<p>Confidence mean={info['confidence_mean']:.3f}, uncertainty mean={info['uncertainty_mean']:.3f}.</p>"
    )
    append_card("Uncertainty Sizing ✔", html)

    print(f"✅ Wrote {RUNS/'uncertainty_size_scalar.csv'}")
    print(f"✅ Wrote {RUNS/'uncertainty_sizing_info.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
