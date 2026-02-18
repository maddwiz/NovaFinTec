# === CLEAN META MODULE (known-good) ===
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .bandit import ExpWeightsBandit
from .bandit_v2 import ThompsonBandit


# ---------- helpers ----------
def _roll_z(s: pd.Series, w: int = 60):
    s = s.astype(float)
    m = s.rolling(w, min_periods=max(10, w // 5)).mean()
    sd = s.rolling(w, min_periods=max(10, w // 5)).std(ddof=1).replace(0, np.nan)
    z = (s - m) / (sd + 1e-9)
    return z.clip(-3, 3).fillna(0.0)


def _znorm(s: pd.Series, w: int = 60):
    return _roll_z(s, w)


def _bandit_prior_path() -> str | None:
    p = os.getenv("Q_BANDIT_PRIOR_FILE", "").strip()
    if p:
        return p
    default = Path(__file__).resolve().parents[1] / "runs_plus" / "novaspine_signal_priors.json"
    return str(default) if default.exists() else None


def _fit_bandit(signals: dict, returns: pd.Series, eta: float):
    bandit_type = os.getenv("Q_BANDIT_TYPE", "thompson").strip().lower()
    if bandit_type == "thompson":
        b = ThompsonBandit(
            n_arms=len(signals),
            decay=float(np.clip(float(os.getenv("Q_THOMPSON_DECAY", "0.995")), 0.90, 1.0)),
            magnitude_scaling=str(os.getenv("Q_THOMPSON_MAGNITUDE_SCALING", "1")).strip().lower() not in {"0", "false", "off", "no"},
            prior_file=_bandit_prior_path(),
        ).fit(signals, returns)
        return b
    return ExpWeightsBandit(eta=eta).fit(signals, returns)


# ---------- macro scores ----------
def vix_macro_score(vix_close: pd.Series) -> pd.Series:
    v = vix_close.astype(float).ffill()
    vz = _roll_z(v, 60)
    dv = v.pct_change().fillna(0.0)
    score = (-vz) + (-_roll_z(dv, 30))  # calm & falling vol → risk-on
    return score


def vix_term_score(vix_9d: pd.Series, vix_3m: pd.Series = None) -> pd.Series:
    a = vix_9d.astype(float).ffill()
    c = vix_3m.astype(float).ffill() if vix_3m is not None else a.rolling(5, min_periods=1).mean()
    contango = c - a  # positive = term > spot → risk-on
    return _roll_z(contango, 60)


def curve_score(dgs10: pd.Series, dgs3m: pd.Series) -> pd.Series:
    slope = (dgs10.astype(float) - dgs3m.astype(float)).ffill()
    return _roll_z(slope.diff().fillna(0.0), 60) + _roll_z(slope, 120)


def credit_score(hyg: pd.Series, lqd: pd.Series) -> pd.Series:
    ratio = (hyg.astype(float) / lqd.astype(float)).ffill()
    r = ratio.pct_change().fillna(0.0)
    return _roll_z(r, 60)


# ---------- price council ----------
def fit_price_council(train_df: pd.DataFrame, eta: float = 0.6) -> dict:
    signals = {"dna": train_df["dna_sig"], "trend": train_df["trend_sig"], "mom": train_df["mom_sig"]}
    returns = train_df["Close"].pct_change().fillna(0.0)
    b = _fit_bandit(signals, returns, eta=eta)
    return b.get_weights()


def price_council_score(df: pd.DataFrame, W: dict) -> pd.Series:
    s = (
        df["dna_sig"] * W.get("dna", 0.0)
        + df["trend_sig"] * W.get("trend", 0.0)
        + df["mom_sig"] * W.get("mom", 0.0)
    )
    return s.astype(float)


def apply_price_council(df: pd.DataFrame, W: dict) -> pd.Series:
    s = (
        df["dna_sig"] * W.get("dna", 0.0)
        + df["trend_sig"] * W.get("trend", 0.0)
        + df["mom_sig"] * W.get("mom", 0.0)
    )
    return s.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))


# ---------- meta learner (floors, caps, entropy prior, 1-day macro lag) ----------
def fit_meta(
    signals_dir: dict,
    returns: pd.Series,
    eta: float = 0.6,
    *,
    price_floor: float = 0.65,
    macro_total_floor: float = 0.20,
    macro_single_cap: float = 0.40,
    alpha_soft: float = 0.35,
    tau: float = 0.6,
) -> dict:
    """
    Learn weights across {price, vix_dir, vix_term, curve, credit} with:
      - per-signal z-norm on a rolling window
      - 1-day lag for non-price signals (avoid leakage)
      - entropy prior (softmax of signed corr)
      - floors/caps: price >= price_floor; macros total >= macro_total_floor; each macro <= macro_single_cap
    """
    # 1) per-fold z-norm & lag macros
    Z = {}
    for k, v in signals_dir.items():
        z = _znorm(pd.Series(v).reindex(returns.index).fillna(0.0), 60)
        if k != "price":
            z = z.shift(1).fillna(0.0)
        Z[k] = z

    # 2) bandit weights
    b = _fit_bandit(Z, returns.fillna(0.0), eta=eta)
    Wb = b.get_weights()

    # 3) entropy-friendly prior from signed corr
    keys = list(Z.keys())
    corr = {}
    ret = returns.fillna(0.0)
    for k in keys:
        s = Z[k]
        den = s.std(ddof=1) * ret.std(ddof=1)
        corr[k] = float((s * ret).mean() / (den + 1e-9)) if den else 0.0
    svec = np.array([corr[k] for k in keys], dtype=float)
    sm = np.exp(svec / tau)
    sm = sm / (sm.sum() + 1e-9)
    Ws = {k: float(w) for k, w in zip(keys, sm)}

    # 4) blend prior + bandit
    W = {k: (1.0 - alpha_soft) * float(Wb.get(k, 0.0)) + alpha_soft * float(Ws.get(k, 0.0)) for k in keys}

    # 5) enforce floors/caps
    # price floor
    if "price" in W and W["price"] < price_floor:
        need = price_floor - W["price"]
        macros = [k for k in keys if k != "price"]
        pool = sum(max(W[m], 0.0) for m in macros) + 1e-9
        if pool > 0:
            for m in macros:
                take = need * (max(W[m], 0.0) / pool)
                W[m] = max(0.0, W[m] - take)
            W["price"] += need

    # macro total floor
    macros = [k for k in keys if k != "price"]
    if macros:
        msum = sum(max(W[k], 0.0) for k in macros)
        if msum < macro_total_floor:
            need = macro_total_floor - msum
            if "price" in W and W["price"] > need:
                W["price"] -= need
                base = {k: 1.0 for k in macros}
                tot = sum(base.values()) + 1e-9
                for k in macros:
                    W[k] += need * (base[k] / tot)

    # per-macro cap
    for k in list(W.keys()):
        if k != "price" and W[k] > macro_single_cap:
            spill = W[k] - macro_single_cap
            W[k] = macro_single_cap
            W["price"] += spill

    # normalize
    tot = sum(W.values()) + 1e-9
    W = {k: float(v / tot) for k, v in W.items()}
    return W


def apply_meta(signals_dir: dict, Wmeta: dict) -> pd.Series:
    # regime gate using VIX magnitude if available
    vix_sig = signals_dir.get("vix_dir", None)
    if vix_sig is not None:
        mag = _znorm(pd.Series(vix_sig).abs(), 60).clip(0, 3)
        gate = 0.3 + 0.7 * (mag / 3.0)  # calm→0.3, stressed→~1.0
    else:
        gate = None

    s = None
    for k, vec in signals_dir.items():
        z = _znorm(pd.Series(vec), 60)
        if k != "price":
            z = z.shift(1).fillna(0.0)
            if gate is not None:
                z = z * gate
        w = Wmeta.get(k, 0.0)
        s = (z * w) if s is None else (s + z * w)
    return s.fillna(0.0).apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
