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
    Returns DataFrame indexed by DATE with hive columns, values in [0.60, 1.45].
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
        q_sh = np.clip(0.5 + 0.5 * np.tanh(sh / 1.8), 0.0, 1.0)

        hit = (s > 0.0).astype(float).rolling(63, min_periods=15).mean().fillna(0.5)
        q_hit = np.clip((hit - 0.42) / 0.20, 0.0, 1.0)

        eq = (1.0 + np.clip(s, -0.95, 0.95)).cumprod()
        peak = np.maximum(eq.cummax(), 1e-12)
        dd = (eq / peak - 1.0).clip(-1.0, 0.0)
        q_dd = np.clip(1.0 - np.abs(dd) / 0.22, 0.0, 1.0)

        vol = s.rolling(21, min_periods=7).std(ddof=1).fillna(0.0)
        vbase = float(np.nanmedian(vol.values)) if np.isfinite(np.nanmedian(vol.values)) else 0.0
        q_vol = np.clip(1.0 - vol / (3.0 * max(vbase, 1e-6)), 0.0, 1.0)

        q = 0.45 * q_sh + 0.25 * q_hit + 0.20 * q_dd + 0.10 * q_vol
        q = np.clip(q, 0.0, 1.0)
        mult = np.clip(0.60 + 0.85 * q, 0.60, 1.45)
        out[hname] = mult.reindex(idx).ffill().fillna(1.0).values

    # Optional shock dampener: in high-shock windows, compress multipliers toward 1.0.
    smp = RUNS / "shock_mask.csv"
    if smp.exists():
        sm = _load_series(smp)
        if sm is not None and len(sm):
            L = min(len(out), len(sm))
            damp = np.clip(1.0 - 0.12 * np.clip(sm[:L], 0.0, 1.0), 0.88, 1.0)
            out.iloc[:L, :] = out.iloc[:L, :].values * damp.reshape(-1, 1)
            out.iloc[:L, :] = np.clip(out.iloc[:L, :].values, 0.60, 1.45)
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


def ecosystem_hive_multipliers(hives):
    """
    Optional per-hive multipliers from prior ecosystem evolution.
    Returns (dict hive->mult in [0.80, 1.20], diagnostics).
    """
    p = RUNS / "hive_evolution.json"
    out = {str(h): 1.0 for h in hives}
    diag = {
        "loaded": False,
        "action_pressure_mean": 0.0,
        "pressure_scalar": 1.0,
    }
    if not p.exists():
        return out, diag
    try:
        obj = json.loads(p.read_text())
    except Exception:
        return out, diag
    if not isinstance(obj, dict):
        return out, diag

    diag["loaded"] = True
    latest_vitality = obj.get("latest_vitality", {})
    if not isinstance(latest_vitality, dict):
        latest_vitality = {}

    try:
        action_pressure_mean = float(obj.get("action_pressure_mean", 0.0))
    except Exception:
        action_pressure_mean = 0.0
    action_pressure_mean = float(np.clip(action_pressure_mean, 0.0, 1.0))
    pressure_scalar = float(np.clip(1.03 - 0.18 * action_pressure_mean, 0.85, 1.03))
    diag["action_pressure_mean"] = action_pressure_mean
    diag["pressure_scalar"] = pressure_scalar

    for h in list(out.keys()):
        try:
            vit = float(latest_vitality.get(h, 1.0))
        except Exception:
            vit = 1.0
        vit = float(np.clip(vit, 0.10, 1.25))
        # Map vitality [0.10,1.25] -> multiplier [0.82,1.18], then apply pressure scalar.
        base = 0.82 + 0.36 * ((vit - 0.10) / 1.15)
        out[h] = float(np.clip(base * pressure_scalar, 0.80, 1.20))
    return out, diag


def _load_series(path: Path):
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
    if a.ndim == 2 and a.shape[1] >= 1:
        a = a[:, -1]
    return a.ravel()


def _load_named_csv_series(path: Path, column: str):
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if column not in df.columns:
        return None
    s = pd.to_numeric(df[column], errors="coerce").fillna(0.0).values.astype(float)
    return s.ravel() if len(s) else None


def dynamic_downside_penalties(index_dates, hives):
    """
    Build DATE x HIVE downside penalties from hive_wf_oos_returns.csv.
    Output values are in [0,1], where 1 means elevated downside stress.
    """
    p = RUNS / "hive_wf_oos_returns.csv"
    idx = pd.DatetimeIndex(index_dates)
    out = pd.DataFrame(index=idx, columns=list(hives), data=0.0, dtype=float)
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
        neg = (-s).clip(lower=0.0)

        semi = np.sqrt((neg * neg).rolling(63, min_periods=15).mean()).fillna(0.0)
        tail = neg.rolling(63, min_periods=15).quantile(0.90).fillna(0.0)
        loss_persist = (s < 0.0).astype(float).rolling(63, min_periods=15).mean().fillna(0.0)

        def _norm(x):
            den = float(np.nanpercentile(x.values, 90)) if len(x) else 0.0
            den = max(den, 1e-9)
            return np.clip(x / den, 0.0, 1.0)

        p_semi = _norm(semi)
        p_tail = _norm(tail)
        p_persist = np.clip(loss_persist, 0.0, 1.0)
        raw = np.clip(0.45 * p_semi + 0.35 * p_tail + 0.20 * p_persist, 0.0, 1.0)
        out[hname] = raw.reindex(idx).ffill().fillna(0.0).values

    # Optional shock amplification.
    smp = RUNS / "shock_mask.csv"
    if smp.exists():
        sm = _load_series(smp)
        if sm is not None and len(sm):
            L = min(len(out), len(sm))
            amp = np.clip(1.0 + 0.25 * np.clip(sm[:L], 0.0, 1.0), 1.0, 1.25)
            out.iloc[:L, :] = np.clip(out.iloc[:L, :].values * amp.reshape(-1, 1), 0.0, 1.0)
    return out


def dynamic_crowding_penalties(index_dates, hives):
    """
    Build DATE x HIVE crowding penalties from rolling absolute cross-hive correlations.
    Output values are in [0,1], where 1 means high crowding risk.
    """
    p = RUNS / "hive_signals.csv"
    idx = pd.DatetimeIndex(index_dates)
    out = pd.DataFrame(index=idx, columns=list(hives), data=0.0, dtype=float)
    if not p.exists():
        return out
    try:
        df = pd.read_csv(p)
    except Exception:
        return out
    need = {"DATE", "HIVE", "hive_signal"}
    if not need.issubset(df.columns):
        return out
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"]).sort_values(["DATE", "HIVE"])
    if df.empty:
        return out

    sig = df.pivot(index="DATE", columns="HIVE", values="hive_signal")
    sig = sig.reindex(columns=list(hives))
    sig = sig.reindex(idx).ffill().fillna(0.0)
    if sig.shape[1] <= 1:
        return out

    lookback = 63
    min_points = 20
    vals = np.zeros((len(sig), sig.shape[1]), float)
    x = sig.values.astype(float)
    n_h = x.shape[1]
    for t in range(len(sig)):
        lo = max(0, t - lookback + 1)
        win = x[lo : t + 1, :]
        if win.shape[0] < min_points:
            continue
        corr = np.corrcoef(win, rowvar=False)
        corr = np.asarray(corr, float)
        if corr.ndim != 2 or corr.shape[0] != n_h:
            continue
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 0.0)
        vals[t, :] = np.mean(np.abs(corr), axis=1)

    # Map mean abs corr into penalty.
    # corr <= 0.25 : near-zero crowding, corr >= 0.80 : full crowding penalty.
    pen = np.clip((vals - 0.25) / 0.55, 0.0, 1.0)

    # Optional shock amplification to reduce crowded books in stress windows.
    smp = RUNS / "shock_mask.csv"
    if smp.exists():
        sm = _load_series(smp)
        if sm is not None and len(sm):
            L = min(len(pen), len(sm))
            amp = np.clip(1.0 + 0.25 * np.clip(sm[:L], 0.0, 1.0), 1.0, 1.25)
            pen[:L, :] = np.clip(pen[:L, :] * amp.reshape(-1, 1), 0.0, 1.0)

    out.loc[:, :] = pen
    return out


def _entropy_norm(w: np.ndarray) -> float:
    a = np.asarray(w, float).ravel()
    if len(a) <= 1:
        return 1.0
    a = np.clip(a, 0.0, None)
    s = float(np.sum(a))
    if s <= 0:
        return 1.0
    p = a / s
    h = -np.sum(np.where(p > 0.0, p * np.log(p), 0.0))
    return float(np.clip(h / np.log(len(p)), 0.0, 1.0))


def adaptive_arb_schedules(base_alpha, base_inertia, pivot_stab):
    """
    Build adaptive alpha/inertia schedules from hive disagreement + council divergence.
    High disagreement/divergence => lower alpha (less concentration), higher inertia.
    """
    T = int(pivot_stab.shape[0])
    if T <= 0:
        return np.asarray([], float), np.asarray([], float), {}

    disagree = (1.0 - pivot_stab.mean(axis=1).values).astype(float)
    disagree = np.nan_to_num(disagree, nan=0.0, posinf=0.0, neginf=0.0)
    disagree = np.clip(disagree, 0.0, 1.0)

    # Optional council divergence from meta stack vs synapses.
    m = _load_series(RUNS / "meta_stack_pred.csv")
    s = _load_series(RUNS / "synapses_pred.csv")
    div = np.zeros(T, float)
    if m is not None and s is not None and len(m) > 0 and len(s) > 0:
        L = min(T, len(m), len(s))
        d = np.abs(np.asarray(m[:L], float) - np.asarray(s[:L], float))
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        p90 = float(np.percentile(d, 90)) if len(d) else 0.0
        div[:L] = np.clip(d / (p90 + 1e-9), 0.0, 1.0)

    # Cross-hive stability dispersion: if high, more confidence in specialization.
    disp = pivot_stab.std(axis=1).values.astype(float)
    disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
    p90d = float(np.percentile(disp, 90)) if len(disp) else 0.0
    disp = np.clip(disp / (p90d + 1e-9), 0.0, 1.0)

    # Regime fracture pressure from disagreement/volatility/breadth stress engine.
    frac = np.zeros(T, float)
    fs = _load_named_csv_series(RUNS / "regime_fracture_signal.csv", "regime_fracture_score")
    if fs is not None and len(fs):
        L = min(T, len(fs))
        frac[:L] = np.clip(np.asarray(fs[:L], float), 0.0, 1.0)

    alpha_t = base_alpha * (1.0 - 0.35 * disagree - 0.25 * div + 0.15 * disp - 0.30 * frac)
    alpha_t = np.where(frac >= 0.85, alpha_t * 0.82, alpha_t)
    alpha_t = np.where((frac >= 0.72) & (frac < 0.85), alpha_t * 0.90, alpha_t)
    alpha_t = np.clip(alpha_t, 0.7, 4.5)

    inertia_t = base_inertia + 0.12 * disagree + 0.10 * div - 0.08 * disp + 0.18 * frac
    inertia_t = np.clip(inertia_t, 0.40, 0.97)

    diag = {
        "mean_disagreement": float(np.mean(disagree)),
        "mean_council_divergence": float(np.mean(div)),
        "mean_stability_dispersion": float(np.mean(disp)),
        "mean_regime_fracture": float(np.mean(frac)),
        "max_regime_fracture": float(np.max(frac)),
        "alpha_mean": float(np.mean(alpha_t)),
        "alpha_min": float(np.min(alpha_t)),
        "alpha_max": float(np.max(alpha_t)),
        "inertia_mean": float(np.mean(inertia_t)),
        "inertia_min": float(np.min(inertia_t)),
        "inertia_max": float(np.max(inertia_t)),
    }
    return alpha_t, inertia_t, diag


def adaptive_entropy_schedules(base_target, base_strength, crowd_tbl, pivot_stab):
    """
    Build time-varying entropy controls.
    High crowding/fracture -> higher entropy target + stronger diversification pull.
    """
    T = int(pivot_stab.shape[0])
    if T <= 0:
        return np.asarray([], float), np.asarray([], float), {}

    crowd = np.zeros(T, float)
    if isinstance(crowd_tbl, pd.DataFrame) and len(crowd_tbl):
        c = crowd_tbl.mean(axis=1).values.astype(float)
        c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        L = min(T, len(c))
        crowd[:L] = np.clip(c[:L], 0.0, 1.0)

    # Stability dispersion: when this is high, specialization signal is stronger.
    disp = pivot_stab.std(axis=1).values.astype(float)
    disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
    p90d = float(np.percentile(disp, 90)) if len(disp) else 0.0
    disp = np.clip(disp / (p90d + 1e-9), 0.0, 1.0)

    frac = np.zeros(T, float)
    fs = _load_named_csv_series(RUNS / "regime_fracture_signal.csv", "regime_fracture_score")
    if fs is not None and len(fs):
        L = min(T, len(fs))
        frac[:L] = np.clip(np.asarray(fs[:L], float), 0.0, 1.0)

    et = float(np.clip(base_target, 0.0, 1.0))
    es = float(np.clip(base_strength, 0.0, 1.0))
    target_t = et + 0.18 * crowd + 0.10 * frac - 0.08 * disp
    strength_t = es + 0.55 * crowd + 0.20 * frac - 0.18 * disp

    # Emergency diversification in crowded fractured regimes.
    hot = (crowd >= 0.58) & (frac >= 0.32)
    target_t = np.where(hot, target_t + 0.06, target_t)
    strength_t = np.where(hot, strength_t + 0.08, strength_t)

    target_t = np.clip(target_t, 0.45, 0.92)
    strength_t = np.clip(strength_t, 0.05, 1.00)
    diag = {
        "crowding_mean": float(np.mean(crowd)),
        "crowding_max": float(np.max(crowd)),
        "fracture_mean": float(np.mean(frac)),
        "dispersion_mean": float(np.mean(disp)),
        "entropy_target_mean": float(np.mean(target_t)),
        "entropy_target_min": float(np.min(target_t)),
        "entropy_target_max": float(np.max(target_t)),
        "entropy_strength_mean": float(np.mean(strength_t)),
        "entropy_strength_min": float(np.min(strength_t)),
        "entropy_strength_max": float(np.max(strength_t)),
    }
    return target_t, strength_t, diag

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
    dn_pen = {}
    cr_pen = {}
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

    # Optional ecosystem-age priors from prior run (closed-loop memory).
    eco_enabled = str(os.getenv("CROSS_HIVE_USE_ECOSYSTEM_PRIOR", "1")).strip().lower() in {"1", "true", "yes", "on"}
    eco_mult, eco_diag = ecosystem_hive_multipliers(pivot_sig.columns.tolist()) if eco_enabled else ({str(h): 1.0 for h in pivot_sig.columns.tolist()}, {"loaded": False, "action_pressure_mean": 0.0, "pressure_scalar": 1.0})
    if eco_mult:
        for hive in list(scores.keys()):
            scores[hive] = scores[hive] * float(eco_mult.get(hive, 1.0))

    # Optional dynamic quality multipliers from per-hive OOS streams.
    dyn_mult = dynamic_quality_multipliers(pivot_sig.index, pivot_sig.columns.tolist())
    dyn_means = {}
    dyn_table = None
    if len(dyn_mult):
        dyn_table = dyn_mult.copy()
        for hive in list(scores.keys()):
            if hive in dyn_mult.columns:
                mvec = np.asarray(dyn_mult[hive].values, float)
                scores[hive] = scores[hive] * np.nan_to_num(mvec, nan=1.0, posinf=1.0, neginf=1.0)
                dyn_means[hive] = float(np.mean(mvec))
    if dyn_table is not None and len(dyn_table):
        dyn_out = dyn_table.reset_index().rename(columns={"index": "DATE"})
        dyn_out.to_csv(RUNS / "hive_dynamic_quality.csv", index=False)

    downside_tbl = dynamic_downside_penalties(pivot_sig.index, pivot_sig.columns.tolist())
    downside_means = {}
    if len(downside_tbl):
        for hive in list(scores.keys()):
            if hive in downside_tbl.columns:
                pvec = np.asarray(downside_tbl[hive].values, float)
                dn_pen[hive] = np.nan_to_num(pvec, nan=0.0, posinf=0.0, neginf=0.0)
                downside_means[hive] = float(np.mean(pvec))
        downside_tbl.reset_index().rename(columns={"index": "DATE"}).to_csv(RUNS / "hive_downside_penalty.csv", index=False)

    crowd_tbl = dynamic_crowding_penalties(pivot_sig.index, pivot_sig.columns.tolist())
    crowd_means = {}
    if len(crowd_tbl):
        for hive in list(scores.keys()):
            if hive in crowd_tbl.columns:
                cvec = np.asarray(crowd_tbl[hive].values, float)
                cr_pen[hive] = np.nan_to_num(cvec, nan=0.0, posinf=0.0, neginf=0.0)
                crowd_means[hive] = float(np.mean(cvec))
        crowd_tbl.reset_index().rename(columns={"index": "DATE"}).to_csv(RUNS / "hive_crowding_penalty.csv", index=False)

    alpha = float(np.clip(float(os.getenv("CROSS_HIVE_ALPHA", "2.2")), 0.2, 10.0))
    inertia = float(np.clip(float(os.getenv("CROSS_HIVE_INERTIA", "0.80")), 0.0, 0.98))
    max_w = float(np.clip(float(os.getenv("CROSS_HIVE_MAX_W", "0.65")), 0.10, 1.0))
    min_w = float(np.clip(float(os.getenv("CROSS_HIVE_MIN_W", "0.02")), 0.0, 0.30))
    max_step_turnover = float(np.clip(float(os.getenv("CROSS_HIVE_MAX_STEP_TURNOVER", "0.0")), 0.0, 2.0))
    turnover_window = int(np.clip(float(os.getenv("CROSS_HIVE_TURNOVER_WINDOW", "5")), 1, 126))
    turnover_limit = float(np.clip(float(os.getenv("CROSS_HIVE_TURNOVER_LIMIT", "0.0")), 0.0, 5.0))
    entropy_target = float(np.clip(float(os.getenv("CROSS_HIVE_ENTROPY_TARGET", "0.60")), 0.0, 1.0))
    entropy_strength = float(np.clip(float(os.getenv("CROSS_HIVE_ENTROPY_STRENGTH", "0.25")), 0.0, 1.0))
    adaptive = str(os.getenv("CROSS_HIVE_ADAPTIVE", "1")).strip().lower() in {"1", "true", "yes", "on"}

    if adaptive:
        alpha_sched, inertia_sched, adaptive_diag = adaptive_arb_schedules(alpha, inertia, pivot_stab)
        ent_target_sched, ent_strength_sched, ent_diag = adaptive_entropy_schedules(
            entropy_target, entropy_strength, crowd_tbl, pivot_stab
        )
    else:
        alpha_sched, inertia_sched = alpha, inertia
        ent_target_sched, ent_strength_sched = entropy_target, entropy_strength
        adaptive_diag = {"enabled": False}
        ent_diag = {"enabled": False}

    if adaptive and isinstance(ent_target_sched, np.ndarray) and len(ent_target_sched) and len(pivot_sig.index) == len(ent_target_sched):
        pd.DataFrame(
            {
                "DATE": pd.DatetimeIndex(pivot_sig.index),
                "entropy_target": np.asarray(ent_target_sched, float),
                "entropy_strength": np.asarray(ent_strength_sched, float),
                "crowding_mean": np.asarray(crowd_tbl.mean(axis=1).values if len(crowd_tbl) else np.zeros(len(pivot_sig.index)), float),
            }
        ).to_csv(RUNS / "hive_entropy_schedule.csv", index=False)

    names, W = arb_weights(
        scores,
        alpha=alpha_sched,
        drawdown_penalty=dd_pen,
        disagreement_penalty=dg_pen,
        downside_penalty=dn_pen if dn_pen else None,
        crowding_penalty=cr_pen if cr_pen else None,
        inertia=inertia_sched,
        max_weight=max_w,
        min_weight=min_w,
        entropy_target=ent_target_sched,
        entropy_strength=ent_strength_sched,
        max_step_turnover=max_step_turnover if max_step_turnover > 0.0 else None,
        rolling_turnover_window=turnover_window if turnover_limit > 0.0 else None,
        rolling_turnover_limit=turnover_limit if turnover_limit > 0.0 else None,
    )
    out = pd.DataFrame(W, index=pivot_sig.index, columns=names)
    if adaptive and len(out) == len(alpha_sched):
        out["arb_alpha"] = np.asarray(alpha_sched, float)
        out["arb_inertia"] = np.asarray(inertia_sched, float)
    out = out.reset_index().rename(columns={"index": "DATE"})
    out.to_csv(RUNS / "cross_hive_weights.csv", index=False)

    if len(out) > 1 and len(names) > 0:
        turns = np.sum(np.abs(np.diff(W, axis=0)), axis=1)
        turn = float(np.mean(turns))
        turn_max = float(np.max(turns))
        if turnover_limit > 0.0:
            roll = np.zeros_like(turns)
            for i in range(len(turns)):
                j0 = max(0, i - (turnover_window - 1))
                roll[i] = float(np.sum(turns[j0 : i + 1]))
            turn_roll_mean = float(np.mean(roll))
            turn_roll_max = float(np.max(roll))
        else:
            turn_roll_mean = None
            turn_roll_max = None
    else:
        turn = 0.0
        turn_max = 0.0
        turn_roll_mean = None
        turn_roll_max = None
    ent = float(np.mean([_entropy_norm(r) for r in W])) if len(W) else None

    summary = {
        "hives": names,
        "rows": int(len(out)),
        "alpha_base": alpha,
        "inertia_base": inertia,
        "adaptive_enabled": bool(adaptive),
        "adaptive_diagnostics": adaptive_diag,
        "entropy_adaptive_diagnostics": ent_diag,
        "max_weight": max_w,
        "min_weight": min_w,
        "entropy_target": entropy_target,
        "entropy_strength": entropy_strength,
        "max_step_turnover": max_step_turnover,
        "rolling_turnover_window": turnover_window if turnover_limit > 0.0 else None,
        "rolling_turnover_limit": turnover_limit if turnover_limit > 0.0 else None,
        "entropy_schedule_file": str(RUNS / "hive_entropy_schedule.csv") if adaptive else None,
        "mean_entropy_norm": ent,
        "mean_turnover": turn,
        "max_turnover": turn_max,
        "rolling_turnover_mean": turn_roll_mean,
        "rolling_turnover_max": turn_roll_max,
        "quality_priors": {k: float(v) for k, v in priors.items()},
        "dynamic_quality_multiplier_mean": dyn_means,
        "dynamic_quality_file": str(RUNS / "hive_dynamic_quality.csv") if dyn_table is not None else None,
        "downside_penalty_mean": downside_means,
        "downside_penalty_file": str(RUNS / "hive_downside_penalty.csv") if len(downside_tbl) else None,
        "crowding_penalty_mean": crowd_means,
        "crowding_penalty_file": str(RUNS / "hive_crowding_penalty.csv") if len(crowd_tbl) else None,
        "novaspine_hive_boosts": {k: float(v) for k, v in ns_mult.items()},
        "ecosystem_prior_enabled": bool(eco_enabled),
        "ecosystem_hive_boosts": {k: float(v) for k, v in eco_mult.items()},
        "ecosystem_prior_diagnostics": eco_diag,
        "date_min": str(out["DATE"].min().date()) if len(out) else None,
        "date_max": str(out["DATE"].max().date()) if len(out) else None,
        "latest_weights": {k: float(out.iloc[-1][k]) for k in names} if len(out) else {},
    }
    (RUNS / "cross_hive_summary.json").write_text(json.dumps(summary, indent=2))

    html = (
        f"<p>Cross-hive weights over {len(names)} hives saved to cross_hive_weights.csv</p>"
        f"<p>Latest: {summary['latest_weights']}</p>"
        f"<p>alpha_base={alpha:.2f}, inertia_base={inertia:.2f}, "
        f"turnover(mean/max)={turn:.4f}/{turn_max:.4f}, "
        f"step_cap={max_step_turnover:.3f}, "
        f"roll_budget={turnover_limit:.3f}@{turnover_window}, adaptive={bool(adaptive)}</p>"
    )
    append_card("Cross-Hive Arbitration ✔", html)
    print(f"✅ Wrote {RUNS/'cross_hive_weights.csv'}")
    print(f"✅ Wrote {RUNS/'cross_hive_summary.json'}")
