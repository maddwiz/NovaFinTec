#!/usr/bin/env python3
# Cross-Hive Arbitration (adaptive, from hive_signals)
# Reads: runs_plus/hive_signals.csv
# Writes:
#   runs_plus/cross_hive_weights.csv
#   runs_plus/hive_score_<hive>.csv
#   runs_plus/cross_hive_summary.json
# Appends a report card.

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from qmods.cross_hive_arb_v1 import arb_weights

RUNS = ROOT/"runs_plus"; RUNS.mkdir(exist_ok=True)

def append_card(title, html):
    for name in ["report_all.html", "report_best_plus.html", "report_plus.html"]:
        p = ROOT/name
        if not p.exists(): continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        txt = txt.replace("</body>", card + "</body>") if "</body>" in txt else txt+card
        p.write_text(txt, encoding="utf-8")
        print(f"✅ Appended card to {name}")


def dynamic_quality_multipliers(index_dates, hives):
    """
    Build DATE x HIVE multipliers from hive_wf_oos_returns.csv rolling quality.
    Returns DataFrame indexed by DATE with hive columns, values in [0.70, 1.40].
    """
    p = RUNS / "hive_wf_oos_returns.csv"
    idx = pd.DatetimeIndex(index_dates)
    out = pd.DataFrame(index=idx, columns=list(hives), data=1.0, dtype=float)
    if not p.exists():
        return out
    try:
        df = pd.read_csv(p)
    except Exception:
        return out
    need = {"DATE", "HIVE", "hive_oos_ret"}
    if not need.issubset(df.columns):
        return out
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"]).sort_values(["DATE", "HIVE"])
    if df.empty:
        return out

    for hive, g in df.groupby("HIVE"):
        hname = str(hive)
        if hname not in out.columns:
            continue
        rr = pd.to_numeric(g["hive_oos_ret"], errors="coerce").fillna(0.0).values.astype(float)
        dates = pd.to_datetime(g["DATE"], errors="coerce")
        if len(rr) < 8:
            continue
        s = pd.Series(rr, index=dates)
        mu = s.rolling(63, min_periods=15).mean()
        sd = s.rolling(63, min_periods=15).std(ddof=1).replace(0.0, np.nan)
        sh = (mu / (sd + 1e-12)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        mult = np.clip(1.0 + 0.25 * np.tanh(sh / 1.5), 0.70, 1.40)
        out[hname] = mult.reindex(idx).ffill().fillna(1.0).values
    return out


def novaspine_hive_multipliers(hives):
    """
    Optional per-hive multipliers from NovaSpine feedback.
    Returns dict hive->mult in [0.80, 1.20].
    """
    p = RUNS / "novaspine_hive_feedback.json"
    out = {str(h): 1.0 for h in hives}
    if not p.exists():
        return out
    try:
        obj = json.loads(p.read_text())
    except Exception:
        return out
    ph = obj.get("per_hive", {}) if isinstance(obj, dict) else {}
    if not isinstance(ph, dict):
        return out
    for h in list(out.keys()):
        rec = ph.get(h, {})
        try:
            b = float(rec.get("boost", 1.0))
        except Exception:
            b = 1.0
        out[h] = float(np.clip(b, 0.80, 1.20))
    return out

if __name__ == "__main__":
    p = RUNS / "hive_signals.csv"
    if not p.exists():
        raise SystemExit("Missing runs_plus/hive_signals.csv (run tools/make_hive.py first)")

    h = pd.read_csv(p)
    need = {"DATE", "HIVE", "hive_signal"}
    if not need.issubset(h.columns):
        raise SystemExit("hive_signals.csv missing required columns: DATE,HIVE,hive_signal")
    h["DATE"] = pd.to_datetime(h["DATE"], errors="coerce")
    h = h.dropna(subset=["DATE"]).sort_values(["DATE", "HIVE"])

    if "hive_health" not in h.columns:
        # fallback to rolling Sharpe proxy
        out = []
        for hive, g in h.groupby("HIVE"):
            gg = g.sort_values("DATE").copy()
            mu = gg["hive_signal"].rolling(63, min_periods=20).mean()
            sd = gg["hive_signal"].rolling(63, min_periods=20).std(ddof=1).replace(0, np.nan)
            gg["hive_health"] = np.tanh((mu / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0) / 2.0)
            out.append(gg)
        h = pd.concat(out, ignore_index=True)

    if "hive_stability" not in h.columns:
        out = []
        for hive, g in h.groupby("HIVE"):
            gg = g.sort_values("DATE").copy()
            gg["hive_stability"] = (1.0 - gg["hive_signal"].rolling(21, min_periods=7).std(ddof=1).fillna(0.0)).clip(0.0, 1.0)
            out.append(gg)
        h = pd.concat(out, ignore_index=True)

    # Build score + penalties per hive on aligned dates
    pivot_sig = h.pivot(index="DATE", columns="HIVE", values="hive_signal").fillna(0.0)
    pivot_health = h.pivot(index="DATE", columns="HIVE", values="hive_health").reindex(pivot_sig.index).fillna(0.0)
    pivot_stab = h.pivot(index="DATE", columns="HIVE", values="hive_stability").reindex(pivot_sig.index).fillna(0.0)
    # pseudo drawdown on cumulative hive signal
    eq = (1.0 + pivot_sig.clip(-0.95, 0.95)).cumprod()
    dd = (eq / np.maximum(eq.cummax(), 1e-12) - 1.0).clip(-1.0, 0.0).abs()
    disagree = (1.0 - pivot_stab).clip(0.0, 1.0)

    scores = {}
    dd_pen = {}
    dg_pen = {}
    for hive in pivot_sig.columns:
        score = 0.55 * pivot_health[hive].values + 0.35 * pivot_sig[hive].rolling(5, min_periods=2).mean().values + 0.10 * pivot_stab[hive].values
        scores[str(hive)] = np.nan_to_num(score, nan=0.0)
        dd_pen[str(hive)] = np.nan_to_num(dd[hive].values, nan=0.0)
        dg_pen[str(hive)] = np.nan_to_num(disagree[hive].values, nan=0.0)
        np.savetxt(RUNS / f"hive_score_{hive}.csv", scores[str(hive)], delimiter=",")

    # Optional static quality priors from per-hive walk-forward metrics.
    priors = {}
    m = RUNS / "hive_wf_metrics.csv"
    if m.exists():
        try:
            met = pd.read_csv(m)
            if {"HIVE", "sharpe_oos"}.issubset(met.columns):
                for _, row in met.iterrows():
                    hname = str(row["HIVE"])
                    sh = float(row.get("sharpe_oos", 0.0))
                    # Map sharpe into [0.75, 1.35] multiplier.
                    priors[hname] = float(np.clip(1.0 + 0.20 * np.tanh(sh / 1.5), 0.75, 1.35))
        except Exception:
            priors = {}
    if priors:
        for hive in list(scores.keys()):
            mult = float(priors.get(hive, 1.0))
            scores[hive] = scores[hive] * mult

    # Optional NovaSpine per-hive memory boosts.
    ns_mult = novaspine_hive_multipliers(pivot_sig.columns.tolist())
    if ns_mult:
        for hive in list(scores.keys()):
            scores[hive] = scores[hive] * float(ns_mult.get(hive, 1.0))

    # Optional dynamic quality multipliers from per-hive OOS streams.
    dyn_mult = dynamic_quality_multipliers(pivot_sig.index, pivot_sig.columns.tolist())
    dyn_means = {}
    if len(dyn_mult):
        for hive in list(scores.keys()):
            if hive in dyn_mult.columns:
                mvec = np.asarray(dyn_mult[hive].values, float)
                scores[hive] = scores[hive] * np.nan_to_num(mvec, nan=1.0, posinf=1.0, neginf=1.0)
                dyn_means[hive] = float(np.mean(mvec))

    alpha = float(np.clip(float(os.getenv("CROSS_HIVE_ALPHA", "2.2")), 0.2, 10.0))
    inertia = float(np.clip(float(os.getenv("CROSS_HIVE_INERTIA", "0.80")), 0.0, 0.98))
    max_w = float(np.clip(float(os.getenv("CROSS_HIVE_MAX_W", "0.65")), 0.10, 1.0))
    min_w = float(np.clip(float(os.getenv("CROSS_HIVE_MIN_W", "0.02")), 0.0, 0.30))

    names, W = arb_weights(
        scores,
        alpha=alpha,
        drawdown_penalty=dd_pen,
        disagreement_penalty=dg_pen,
        inertia=inertia,
        max_weight=max_w,
        min_weight=min_w,
    )
    out = pd.DataFrame(W, index=pivot_sig.index, columns=names).reset_index().rename(columns={"index": "DATE"})
    out.to_csv(RUNS / "cross_hive_weights.csv", index=False)

    if len(out) > 1 and len(names) > 0:
        turn = float(np.mean(np.sum(np.abs(np.diff(W, axis=0)), axis=1)))
    else:
        turn = 0.0

    summary = {
        "hives": names,
        "rows": int(len(out)),
        "alpha": alpha,
        "inertia": inertia,
        "max_weight": max_w,
        "min_weight": min_w,
        "mean_turnover": turn,
        "quality_priors": {k: float(v) for k, v in priors.items()},
        "dynamic_quality_multiplier_mean": dyn_means,
        "novaspine_hive_boosts": {k: float(v) for k, v in ns_mult.items()},
        "date_min": str(out["DATE"].min().date()) if len(out) else None,
        "date_max": str(out["DATE"].max().date()) if len(out) else None,
        "latest_weights": {k: float(out.iloc[-1][k]) for k in names} if len(out) else {},
    }
    (RUNS / "cross_hive_summary.json").write_text(json.dumps(summary, indent=2))

    html = (
        f"<p>Cross-hive weights over {len(names)} hives saved to cross_hive_weights.csv</p>"
        f"<p>Latest: {summary['latest_weights']}</p>"
        f"<p>alpha={alpha:.2f}, inertia={inertia:.2f}, turnover={turn:.4f}</p>"
    )
    append_card("Cross-Hive Arbitration ✔", html)
    print(f"✅ Wrote {RUNS/'cross_hive_weights.csv'}")
    print(f"✅ Wrote {RUNS/'cross_hive_summary.json'}")
