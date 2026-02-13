#!/usr/bin/env python3
# Builds a shock mask from vol/news and gates a signal.

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from qmods.news_shock_guard import apply_shock_guard, shock_mask

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


def _load_series(rel):
    p = ROOT / rel
    if not p.exists():
        return None
    try:
        a = np.loadtxt(p, delimiter=",")
    except Exception:
        try:
            a = np.loadtxt(p, delimiter=",", skiprows=1)
        except Exception:
            return None
    a = np.asarray(a, float)
    if a.ndim == 2 and a.shape[1] >= 1:
        a = a[:, -1]
    return a.ravel()


def _first_series(paths):
    for rel in paths:
        a = _load_series(rel)
        if a is not None:
            return a
    return None


def _append_card(title, html):
    for name in ["report_all.html", "report_best_plus.html", "report_plus.html", "report.html"]:
        f = ROOT / name
        if not f.exists():
            continue
        txt = f.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        f.write_text(txt.replace("</body>", card + "</body>") if "</body>" in txt else txt + card, encoding="utf-8")


if __name__ == "__main__":
    r = _first_series(["runs_plus/daily_returns.csv", "daily_returns.csv"])
    if r is None:
        print("(!) No returns; skipping.")
        raise SystemExit(0)

    vol = np.abs(np.asarray(r, float).ravel())
    z = float(os.getenv("NEWS_SHOCK_Z", "2.5"))
    min_len = int(max(1, int(float(os.getenv("NEWS_SHOCK_MIN_LEN", "2")))))
    lb = int(max(10, int(float(os.getenv("NEWS_SHOCK_LOOKBACK", "63")))))
    cooldown = int(max(0, int(float(os.getenv("NEWS_SHOCK_COOLDOWN", "3")))))
    quantile = os.getenv("NEWS_SHOCK_QUANTILE", "0.985").strip()
    qv = float(quantile) if quantile else None

    mask = shock_mask(vol, z=z, min_len=min_len, lookback=lb, cooldown=cooldown, quantile=qv)

    # Optional explicit news events hard-trigger.
    news = _load_series("runs_plus/news_events.csv")
    if news is not None:
        L = min(len(mask), len(news))
        mask[:L] = np.maximum(mask[:L], (news[:L] > 0).astype(int))

    np.savetxt(RUNS / "shock_mask.csv", mask, delimiter=",")

    sig = _first_series(["runs_plus/meta_stack_pred.csv", "runs_plus/synapses_pred.csv", "runs_plus/reflex_signal.csv"])
    gated_written = False
    if sig is not None:
        L = min(len(sig), len(mask))
        alpha = float(np.clip(float(os.getenv("NEWS_SHOCK_ALPHA", "0.50")), 0.0, 1.0))
        gated = apply_shock_guard(sig[:L], mask[:L], alpha=alpha)
        np.savetxt(RUNS / "signal_shock_gated.csv", gated, delimiter=",")
        gated_written = True
    else:
        alpha = float(np.clip(float(os.getenv("NEWS_SHOCK_ALPHA", "0.50")), 0.0, 1.0))

    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "params": {
            "z": z,
            "min_len": min_len,
            "lookback": lb,
            "cooldown": cooldown,
            "quantile": qv,
            "alpha": alpha,
        },
        "length": int(len(mask)),
        "shock_days": int(np.sum(mask)),
        "shock_rate": float(np.mean(mask)) if len(mask) else 0.0,
        "signal_gated": bool(gated_written),
    }
    (RUNS / "shock_mask_info.json").write_text(json.dumps(info, indent=2))

    _append_card(
        "Shock/News Sentinel ✔",
        f"<p>shock_days={info['shock_days']}, shock_rate={info['shock_rate']:.3f}, signal_gated={bool(gated_written)}</p>",
    )
    print("✅ Wrote runs_plus/shock_mask.csv")
    if gated_written:
        print("✅ Wrote runs_plus/signal_shock_gated.csv")
    print("✅ Wrote runs_plus/shock_mask_info.json")
