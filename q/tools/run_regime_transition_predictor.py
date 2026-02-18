#!/usr/bin/env python3
"""
Regime-transition risk predictor using online HMM.

Writes:
  - runs_plus/regime_transition_risk.csv
  - runs_plus/regime_transition_scalar.csv
  - runs_plus/regime_transition_info.json
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

from qengine.regime_hmm import RegimeHMM


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


def _rolling_z(x: np.ndarray, w: int = 63) -> np.ndarray:
    s = pd.Series(np.asarray(x, float).ravel())
    mu = s.rolling(int(max(10, w)), min_periods=max(10, int(w // 3))).mean()
    sd = s.rolling(int(max(10, w)), min_periods=max(10, int(w // 3))).std(ddof=1).replace(0.0, np.nan)
    z = (s - mu) / (sd + 1e-12)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(float)


def _load_label_series(t: int) -> np.ndarray | None:
    for p in [RUNS / "regime_series.csv", RUNS / "regime_labels.csv"]:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty:
            continue
        col = None
        for c in df.columns:
            if str(c).strip().lower() in {"regime", "label", "regime_label"}:
                col = c
                break
        if col is None:
            col = df.columns[-1]
        vals = df[col].astype(str).str.strip().str.lower().values
        if len(vals) >= t:
            return vals[-t:]
        out = np.array(["unknown"] * t, dtype=object)
        out[-len(vals) :] = vals
        return out
    return None


def _map_anchor(lbl: str) -> int | None:
    s = str(lbl).strip().lower()
    if s in {"calm_trend", "trend", "trending"}:
        return 0
    if s in {"calm_chop", "chop", "choppy", "range", "range_day", "rotational"}:
        return 1
    if s in {"highvol_trend", "highvol", "squeeze", "breakout", "volatility"}:
        return 2
    if s in {"crisis", "risk_off", "panic"}:
        return 3
    return None


def main() -> int:
    r = _load_series(RUNS / "daily_returns.csv")
    if r is None:
        r = _load_series(RUNS / "wf_oos_returns.csv")
    if r is None:
        print("(!) Missing returns for regime transition predictor; skipping.")
        return 0

    t = len(r)

    vol20 = pd.Series(r).rolling(20, min_periods=10).std(ddof=1).fillna(0.0).values * np.sqrt(252.0)
    trend63 = pd.Series(r).rolling(63, min_periods=20).sum().fillna(0.0).values
    chop21 = pd.Series(np.abs(r)).rolling(21, min_periods=10).mean().fillna(0.0).values / (
        np.abs(pd.Series(r).rolling(21, min_periods=10).sum().fillna(0.0).values) + 1e-9
    )

    credit = _load_series(RUNS / "credit_leadlag_signal.csv")
    if credit is None:
        credit = np.zeros(t, dtype=float)
    if len(credit) < t:
        pad = np.zeros(t - len(credit), dtype=float)
        credit = np.concatenate([pad, credit], axis=0)
    credit = credit[-t:]
    credit_chg = np.concatenate([[0.0], np.diff(credit)])

    vix_proxy = _load_series(RUNS / "vol_forecast.csv")
    if vix_proxy is None:
        vix_proxy = vol20.copy()
    if len(vix_proxy) < t:
        pad = np.full(t - len(vix_proxy), float(vix_proxy[-1]) if len(vix_proxy) else 0.0)
        vix_proxy = np.concatenate([pad, vix_proxy], axis=0)
    vix_proxy = vix_proxy[-t:]
    vix_slope = np.concatenate([[0.0], np.diff(vix_proxy)])

    dna_vel = _load_series(RUNS / "dna_drift.csv")
    if dna_vel is None:
        dna_vel = np.zeros(t, dtype=float)
    if len(dna_vel) < t:
        pad = np.zeros(t - len(dna_vel), dtype=float)
        dna_vel = np.concatenate([pad, dna_vel], axis=0)
    dna_vel = dna_vel[-t:]
    dna_vel = np.concatenate([[0.0], np.diff(dna_vel)])

    X = np.column_stack(
        [
            _rolling_z(vol20, 63),
            _rolling_z(trend63, 63),
            _rolling_z(chop21, 63),
            _rolling_z(credit_chg, 63),
            _rolling_z(vix_slope, 63),
            _rolling_z(dna_vel, 63),
        ]
    )

    labels = _load_label_series(t)
    hmm = RegimeHMM(n_features=X.shape[1])
    horizon = int(np.clip(int(float(os.getenv("Q_REGIME_TRANSITION_HORIZON", "5"))), 1, 20))

    rows = []
    scalars = np.ones(t, dtype=float)

    for i in range(t):
        anchor = _map_anchor(labels[i]) if labels is not None else None
        hmm.update(X[i], known_state=anchor)
        risk = hmm.transition_risk(horizon=horizon)

        trans = float(risk.get("transition_prob", 0.0))
        crisis = float(risk.get("crisis_risk_5d", 0.0))

        s1 = 1.0
        if crisis > 0.15:
            s1 *= max(0.0, 1.0 - crisis)
        if trans > 0.40:
            s1 *= max(0.6, 1.0 - 0.5 * trans)
        scalars[i] = float(np.clip(s1, 0.0, 1.2))

        rows.append(
            {
                "t": int(i),
                "transition_prob": trans,
                "crisis_risk_5d": crisis,
                "current_regime": str(risk.get("current_regime", "CALM_CHOP")),
                "current_confidence": float(risk.get("current_confidence", 0.25)),
                "scalar": float(scalars[i]),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(RUNS / "regime_transition_risk.csv", index=False)
    np.savetxt(RUNS / "regime_transition_scalar.csv", scalars.reshape(-1, 1), delimiter=",")

    info = {
        "ok": True,
        "rows": int(t),
        "horizon": int(horizon),
        "transition_prob_mean": float(df["transition_prob"].mean()),
        "crisis_risk_mean": float(df["crisis_risk_5d"].mean()),
        "scalar_mean": float(np.mean(scalars)),
        "scalar_min": float(np.min(scalars)),
        "scalar_max": float(np.max(scalars)),
        "anchor_used": bool(labels is not None),
        "high_transition_share": float(np.mean(df["transition_prob"].values > 0.40)),
        "high_crisis_share": float(np.mean(df["crisis_risk_5d"].values > 0.15)),
    }
    (RUNS / "regime_transition_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    _append_card(
        "Regime Transition Predictor ✔",
        (
            f"<p>rows={t}, transition_mean={info['transition_prob_mean']:.3f}, crisis_mean={info['crisis_risk_mean']:.3f}</p>"
            f"<p>scalar_mean={info['scalar_mean']:.3f}, scalar_min={info['scalar_min']:.3f}</p>"
        ),
    )

    print(f"✅ Wrote {RUNS/'regime_transition_risk.csv'}")
    print(f"✅ Wrote {RUNS/'regime_transition_scalar.csv'}")
    print(f"✅ Wrote {RUNS/'regime_transition_info.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
