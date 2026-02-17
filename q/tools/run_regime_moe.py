#!/usr/bin/env python3
"""
Regime specialists (Mixture-of-Experts) governor.

Builds trend / mean-reversion / shock specialists and blends them by a
regime gate, then emits a scalar governor.

Writes:
  - runs_plus/regime_moe_signal.csv
  - runs_plus/regime_moe_governor.csv
  - runs_plus/regime_moe_info.json
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


def _zscore(x: np.ndarray, win: int) -> np.ndarray:
    v = np.asarray(x, float).ravel()
    T = len(v)
    out = np.zeros(T, float)
    w = int(max(2, win))
    for t in range(T):
        lo = max(0, t - w + 1)
        seg = v[lo : t + 1]
        mu = float(np.mean(seg))
        sd = float(np.std(seg) + 1e-9)
        out[t] = (v[t] - mu) / sd
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    v = np.asarray(x, float).ravel()
    out = np.zeros_like(v, dtype=float)
    a = 2.0 / (max(1, int(span)) + 1.0)
    for i in range(len(v)):
        if i == 0:
            out[i] = v[i]
        else:
            out[i] = a * v[i] + (1.0 - a) * out[i - 1]
    return out


def _softmax3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.column_stack([a, b, c]).astype(float)
    A = A - np.max(A, axis=1, keepdims=True)
    E = np.exp(np.clip(A, -40.0, 40.0))
    S = np.sum(E, axis=1, keepdims=True)
    S = np.where(S < 1e-9, 1.0, S)
    W = E / S
    return W[:, 0], W[:, 1], W[:, 2]


def build_regime_moe(
    daily_returns: np.ndarray,
    base_signal: np.ndarray,
    *,
    governor_alpha: float = 0.22,
    governor_min: float = 0.70,
    governor_max: float = 1.20,
) -> dict:
    r = np.asarray(daily_returns, float).ravel()
    s = np.asarray(base_signal, float).ravel()
    T = min(len(r), len(s))
    r = r[:T]
    s = s[:T]

    z_trend = _zscore(_ema(r, 42), 126)
    z_vol = _zscore(np.abs(r), 63)
    z_shock = _zscore(np.abs(r), 21)

    trend_logit = z_trend - 0.40 * z_vol
    mr_logit = -np.abs(z_trend) - 0.20 * z_vol
    shock_logit = z_shock + 0.35 * z_vol
    w_tr, w_mr, w_sh = _softmax3(trend_logit, mr_logit, shock_logit)

    trend_sig = np.tanh(1.25 * _ema(s, 5))
    mr_sig = np.tanh(-0.9 * _ema(s, 2))
    shock_sig = np.tanh(-1.0 * np.sign(r) * np.minimum(1.0, np.abs(z_shock)))
    moe_signal = w_tr * trend_sig + w_mr * mr_sig + w_sh * shock_sig

    a = float(np.clip(governor_alpha, 0.0, 1.0))
    g = 1.0 + a * np.tanh(moe_signal)
    g = np.clip(g, float(governor_min), float(governor_max))

    return {
        "signal": np.nan_to_num(moe_signal, nan=0.0, posinf=0.0, neginf=0.0),
        "governor": np.nan_to_num(g, nan=1.0, posinf=1.0, neginf=1.0),
        "w_trend": w_tr,
        "w_meanrev": w_mr,
        "w_shock": w_sh,
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
    r = _load_series(RUNS / "daily_returns.csv")
    if r is None:
        print("(!) Missing runs_plus/daily_returns.csv; skipping regime MoE.")
        return 0

    base = None
    for rel in ["meta_mix.csv", "meta_stack_pred.csv", "synapses_pred.csv", "regime_signal.csv"]:
        s = _load_series(RUNS / rel)
        if s is not None:
            base = s if base is None else (0.5 * base[: min(len(base), len(s))] + 0.5 * s[: min(len(base), len(s))])
    if base is None:
        # fallback to lagged return sign
        b = np.zeros_like(r)
        b[1:] = np.sign(r[:-1])
        base = b

    alpha = float(np.clip(float(os.getenv("Q_REGIME_MOE_ALPHA", "0.22")), 0.0, 1.0))
    gmin = float(np.clip(float(os.getenv("Q_REGIME_MOE_MIN", "0.70")), 0.10, 1.5))
    gmax = float(np.clip(float(os.getenv("Q_REGIME_MOE_MAX", "1.20")), max(gmin, 0.2), 1.8))
    out = build_regime_moe(r, base, governor_alpha=alpha, governor_min=gmin, governor_max=gmax)

    np.savetxt(RUNS / "regime_moe_signal.csv", out["signal"], delimiter=",")
    np.savetxt(RUNS / "regime_moe_governor.csv", out["governor"], delimiter=",")

    info = {
        "rows": int(len(out["signal"])),
        "alpha": float(alpha),
        "governor_min": float(gmin),
        "governor_max": float(gmax),
        "governor_mean": float(np.mean(out["governor"])),
        "governor_min_seen": float(np.min(out["governor"])),
        "governor_max_seen": float(np.max(out["governor"])),
        "w_trend_mean": float(np.mean(out["w_trend"])),
        "w_meanrev_mean": float(np.mean(out["w_meanrev"])),
        "w_shock_mean": float(np.mean(out["w_shock"])),
    }
    (RUNS / "regime_moe_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    html = (
        f"<p>Regime MoE governor built (rows={info['rows']}). "
        f"mean={info['governor_mean']:.3f} range=[{info['governor_min_seen']:.3f},{info['governor_max_seen']:.3f}].</p>"
        f"<p>Regime mix means: trend={info['w_trend_mean']:.3f}, mean-rev={info['w_meanrev_mean']:.3f}, shock={info['w_shock_mean']:.3f}.</p>"
    )
    append_card("Regime MoE ✔", html)

    print(f"✅ Wrote {RUNS/'regime_moe_signal.csv'}")
    print(f"✅ Wrote {RUNS/'regime_moe_governor.csv'}")
    print(f"✅ Wrote {RUNS/'regime_moe_info.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
