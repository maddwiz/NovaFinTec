#!/usr/bin/env python3
# Meta-learner over council votes (leakage-safe ridge + CV alpha)
# Reads:
#   runs_plus/council_votes.csv
#   runs_plus/target_returns.csv OR runs_plus/daily_returns.csv OR daily_returns.csv
# Writes:
#   runs_plus/meta_stack_pred.csv
#   runs_plus/meta_stack_confidence.csv
#   runs_plus/meta_stack_summary.json

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmods.meta_stack_v1 import MetaStackV1

RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


def load_first(paths):
    for p in paths:
        f = ROOT / p
        if not f.exists():
            continue
        try:
            return np.loadtxt(f, delimiter=",")
        except Exception:
            try:
                return np.loadtxt(f, delimiter=",", skiprows=1)
            except Exception:
                pass
    return None


def annualized_sharpe(r):
    r = np.asarray(r, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size < 4:
        return 0.0
    mu = float(np.mean(r))
    sd = float(np.std(r) + 1e-12)
    return float((mu / sd) * np.sqrt(252.0))


def safe_corr(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    x = a[:n]
    y = b[:n]
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return 0.0
    try:
        c = float(np.corrcoef(x[m], y[m])[0, 1])
        return c if np.isfinite(c) else 0.0
    except Exception:
        return 0.0


def append_card(title, html):
    for name in ["report_all.html", "report_best_plus.html", "report_plus.html", "report.html"]:
        p = ROOT / name
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        txt = txt.replace("</body>", card + "</body>") if "</body>" in txt else txt + card
        p.write_text(txt, encoding="utf-8")
        print(f"✅ Appended card to {name}")


if __name__ == "__main__":
    V = load_first(["runs_plus/council_votes.csv"])
    y = load_first(
        [
            "runs_plus/target_returns.csv",
            "runs_plus/daily_returns.csv",
            "daily_returns.csv",
            "portfolio_daily_returns.csv",
        ]
    )
    if V is None or y is None:
        print("(!) Missing council_votes or returns. Skipping.")
        raise SystemExit(0)
    if V.ndim == 1:
        V = V.reshape(-1, 1)
    y = np.asarray(y).ravel()
    T = min(len(y), V.shape[0])
    V = V[:T]
    y = y[:T]

    alphas = np.geomspace(0.01, 30.0, 16)
    model = MetaStackV1(alpha=1.0, alphas=alphas, min_train=84, val_size=63, step=21, winsor=5.0).fit(V, y)

    pred = model.predict(V)
    conf = model.predict_confidence(V)
    np.savetxt(RUNS / "meta_stack_pred.csv", pred, delimiter=",")
    np.savetxt(RUNS / "meta_stack_confidence.csv", conf, delimiter=",")

    lag_p = pred[:-1]
    fwd_y = y[1:]
    strat = lag_p * fwd_y
    sh = annualized_sharpe(strat)
    hit = float(np.mean(np.sign(lag_p) == np.sign(fwd_y))) if len(lag_p) > 0 else 0.0
    corr = safe_corr(pred[1:], y[1:])

    summary = {
        "rows": int(T),
        "features": int(V.shape[1]),
        "selected_alpha": float(model.alpha_),
        "cv_score": float(model.cv_score_) if model.cv_score_ is not None and np.isfinite(model.cv_score_) else None,
        "oos_like_sharpe": float(sh),
        "oos_like_hit_rate": float(hit),
        "pred_y_corr": float(corr),
        "residual_std": float(model.resid_std_),
        "mean_confidence": float(np.mean(conf)) if len(conf) else 0.0,
    }
    (RUNS / "meta_stack_summary.json").write_text(json.dumps(summary, indent=2))

    html = (
        f"<p>MetaStackV1 rows={T}, K={V.shape[1]}, alpha={summary['selected_alpha']:.4f}.</p>"
        f"<p>OOS-like Sharpe={summary['oos_like_sharpe']:.3f}, hit={summary['oos_like_hit_rate']:.3f}, "
        f"corr={summary['pred_y_corr']:.3f}, conf={summary['mean_confidence']:.3f}.</p>"
    )
    append_card("Meta-Learner (Ridge CV) ✔", html)
    print(f"✅ Wrote {RUNS/'meta_stack_pred.csv'}")
    print(f"✅ Wrote {RUNS/'meta_stack_confidence.csv'}")
    print(f"✅ Wrote {RUNS/'meta_stack_summary.json'}")
