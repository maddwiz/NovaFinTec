#!/usr/bin/env python3
# Council Meta-Tuning: confidence-aware blend of meta_stack + synapses
# Reads:
#   runs_plus/meta_stack_pred.csv, runs_plus/synapses_pred.csv
#   optional: runs_plus/meta_stack_confidence.csv, runs_plus/synapses_confidence.csv
#   runs_plus/daily_returns.csv or daily_returns.csv (for scoring)
# Writes:
#   runs_plus/meta_mix.csv            (final position signal in [-1,1])
#   runs_plus/meta_mix_leverage.csv   (exposure multiplier around 1.0)
#   runs_plus/meta_mix_info.json      (best params + diagnostics)

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from qmods.confidence_calibration import (
    apply_empirical_calibrator,
    fit_empirical_calibrator,
    reliability_governor_from_calibrated,
)
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


def load_series(rel):
    p = ROOT / rel
    if not p.exists():
        return None
    try:
        return np.loadtxt(p, delimiter=",").ravel()
    except Exception:
        return np.loadtxt(p, delimiter=",", skiprows=1).ravel()


def first_series(paths):
    for rel in paths:
        a = load_series(rel)
        if a is not None:
            return a
    return None


def zscore(x):
    x = np.asarray(x, float)
    mu = np.nanmean(x)
    sd = np.nanstd(x) + 1e-12
    z = (x - mu) / sd
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def annualized_sharpe(r):
    r = np.asarray(r, float).ravel()
    r = r[np.isfinite(r)]
    if r.size < 4:
        return 0.0
    mu = np.nanmean(r)
    sd = np.nanstd(r) + 1e-12
    return float((mu / sd) * np.sqrt(252.0))


def downside_vol(r):
    r = np.asarray(r, float).ravel()
    r = r[np.isfinite(r)]
    neg = r[r < 0.0]
    if neg.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(neg**2)) * np.sqrt(252.0))


def smooth_signal(sig, beta):
    sig = np.asarray(sig, float).ravel()
    if len(sig) == 0 or beta <= 0:
        return sig
    beta = float(np.clip(beta, 0.0, 0.95))
    out = np.zeros_like(sig)
    out[0] = sig[0]
    for t in range(1, len(sig)):
        out[t] = beta * out[t - 1] + (1.0 - beta) * sig[t]
    return out


def append_card(title, html):
    for name in ["report_all.html", "report_best_plus.html", "report_plus.html", "report.html"]:
        f = ROOT / name
        if not f.exists():
            continue
        txt = f.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        f.write_text(txt.replace("</body>", card + "</body>") if "</body>" in txt else txt + card, encoding="utf-8")


if __name__ == "__main__":
    m = load_series("runs_plus/meta_stack_pred.csv")
    s = load_series("runs_plus/synapses_pred.csv")
    if m is None or s is None:
        print("(!) Need meta_stack_pred.csv and synapses_pred.csv; skipping.")
        raise SystemExit(0)

    mc = load_series("runs_plus/meta_stack_confidence.csv")
    sc = load_series("runs_plus/synapses_confidence.csv")
    y = first_series(["runs_plus/daily_returns.csv", "daily_returns.csv"])
    if y is None:
        print("(!) No returns found; skipping.")
        raise SystemExit(0)

    T = min(len(m), len(s), len(y))
    if mc is not None:
        T = min(T, len(mc))
    if sc is not None:
        T = min(T, len(sc))
    m = m[:T]
    s = s[:T]
    y = y[:T]
    mc = np.clip(mc[:T], 0.0, 1.0) if mc is not None else np.ones(T, float)
    sc = np.clip(sc[:T], 0.0, 1.0) if sc is not None else np.ones(T, float)

    zm = zscore(m)
    zs = zscore(s)
    zm_eff = zm * (0.4 + 0.6 * mc)
    zs_eff = zs * (0.4 + 0.6 * sc)

    # leakage guard: score on lagged signal against forward returns
    y_fwd = y[1:]
    grid_a = np.linspace(0.0, 1.0, 21)       # weight on meta
    grid_beta = np.array([0.00, 0.20, 0.35]) # temporal smoothing
    grid_gross = np.array([0.18, 0.24, 0.30])# position sensitivity

    best = {
        "score": -1e9,
        "alpha_meta": 0.5,
        "smooth_beta": 0.2,
        "gross": 0.24,
        "sharpe": 0.0,
        "downside_vol": 0.0,
        "turnover": 0.0,
        "hit_rate": 0.0,
    }

    for a in grid_a:
        for beta in grid_beta:
            for gross in grid_gross:
                raw = a * zm_eff + (1.0 - a) * zs_eff
                smooth = smooth_signal(raw, beta=float(beta))
                pos = np.tanh(float(gross) * smooth)
                lag_pos = pos[:-1]
                pnl = lag_pos * y_fwd

                sh = annualized_sharpe(pnl)
                dsv = downside_vol(pnl)
                to = float(np.mean(np.abs(np.diff(pos)))) if len(pos) > 2 else 0.0
                hit = float(np.mean(np.sign(lag_pos) == np.sign(y_fwd))) if len(lag_pos) > 0 else 0.0
                score = sh - 0.15 * dsv - 0.08 * to
                if score > best["score"]:
                    best.update(
                        {
                            "score": float(score),
                            "alpha_meta": float(a),
                            "smooth_beta": float(beta),
                            "gross": float(gross),
                            "sharpe": float(sh),
                            "downside_vol": float(dsv),
                            "turnover": float(to),
                            "hit_rate": float(hit),
                        }
                    )

    best_raw = best["alpha_meta"] * zm_eff + (1.0 - best["alpha_meta"]) * zs_eff
    best_smooth = smooth_signal(best_raw, beta=best["smooth_beta"])
    best_pos = np.tanh(best["gross"] * best_smooth)
    mix_conf = np.clip(best["alpha_meta"] * mc + (1.0 - best["alpha_meta"]) * sc, 0.0, 1.0)
    raw_conf = np.clip(0.55 * mix_conf + 0.45 * np.abs(best_pos), 0.0, 1.0)

    # Confidence calibration: map raw confidence to empirical directional hit rates.
    # Pred at t maps to y[t+1], so use lag-aligned outcomes for fit.
    lag_pos = best_pos[:-1]
    out_hit = (np.sign(lag_pos) == np.sign(y_fwd)).astype(float)
    conf_pred = raw_conf[:-1]
    cal = fit_empirical_calibrator(conf_pred, out_hit, n_bins=10, min_count=24)
    conf_cal = apply_empirical_calibrator(raw_conf, cal)
    rel_gov = reliability_governor_from_calibrated(conf_cal, lo=0.72, hi=1.16, smooth=0.88)

    leverage = np.clip(1.0 + 0.25 * np.abs(best_pos) * mix_conf, 0.80, 1.35)

    np.savetxt(RUNS / "meta_mix.csv", best_pos, delimiter=",")
    np.savetxt(RUNS / "meta_mix_leverage.csv", leverage, delimiter=",")
    np.savetxt(RUNS / "meta_mix_confidence_raw.csv", raw_conf, delimiter=",")
    np.savetxt(RUNS / "meta_mix_confidence_calibrated.csv", conf_cal, delimiter=",")
    np.savetxt(RUNS / "meta_mix_reliability_governor.csv", rel_gov, delimiter=",")

    hit_pred = float(np.mean(out_hit)) if len(out_hit) else 0.0
    hit_cal = float(np.mean(np.where(conf_pred >= 0.5, 1.0, 0.0) == out_hit)) if len(out_hit) else 0.0
    brier_raw = float(np.mean((conf_pred - out_hit) ** 2)) if len(out_hit) else None
    brier_cal = float(np.mean((conf_cal[:-1] - out_hit) ** 2)) if len(out_hit) else None

    info = {
        "length": int(T),
        "best_alpha_meta": best["alpha_meta"],
        "best_smooth_beta": best["smooth_beta"],
        "best_gross": best["gross"],
        "score": best["score"],
        "oos_like_sharpe": best["sharpe"],
        "downside_vol": best["downside_vol"],
        "turnover": best["turnover"],
        "hit_rate": best["hit_rate"],
        "directional_hit_rate": hit_pred,
        "confidence_hit_rate_at_0_5": hit_cal,
        "brier_raw": brier_raw,
        "brier_calibrated": brier_cal,
        "mean_confidence_raw": float(np.mean(raw_conf)) if len(raw_conf) else 0.0,
        "mean_confidence_calibrated": float(np.mean(conf_cal)) if len(conf_cal) else 0.0,
        "mean_leverage": float(np.mean(leverage)) if len(leverage) else 1.0,
        "mean_confidence": float(np.mean(mix_conf)) if len(mix_conf) else 0.0,
        "mean_reliability_governor": float(np.mean(rel_gov)) if len(rel_gov) else 1.0,
        "calibration": cal,
    }
    (RUNS / "meta_mix_info.json").write_text(json.dumps(info, indent=2))

    append_card(
        "Best Council Mix ✔",
        (
            f"<p>alpha(meta)={best['alpha_meta']:.2f}, beta={best['smooth_beta']:.2f}, gross={best['gross']:.2f}</p>"
            f"<p>Sharpe={best['sharpe']:.3f}, downside={best['downside_vol']:.3f}, turnover={best['turnover']:.3f}, "
            f"hit={best['hit_rate']:.3f}, mean lev={info['mean_leverage']:.3f}</p>"
            f"<p>Conf raw={info['mean_confidence_raw']:.3f}, cal={info['mean_confidence_calibrated']:.3f}, "
            f"rel gov={info['mean_reliability_governor']:.3f}, brier(raw/cal)="
            f"{(info['brier_raw'] or 0.0):.4f}/{(info['brier_calibrated'] or 0.0):.4f}</p>"
        ),
    )
    print(
        "✅ Saved runs_plus/meta_mix.csv and runs_plus/meta_mix_leverage.csv "
        f"(alpha={best['alpha_meta']:.2f}, sharpe={best['sharpe']:.3f})"
    )
