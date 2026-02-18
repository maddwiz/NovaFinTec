#!/usr/bin/env python3
"""
Regime-conditional council weighting.

Builds regime-specific council-member weights with walk-forward training and
an explicit embargo between regime-classifier calibration and signal fitting.

Writes:
  - runs_plus/regime_council_weights.csv  (T x K, per-row council-member weights)
  - runs_plus/weights_regime_council.csv  (T x N, base-weight candidate for assembler)
  - runs_plus/regime_council_info.json
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

from qengine.bandit import ExpWeightsBandit
from qengine.bandit_v2 import ThompsonBandit

REGIMES = ("trending", "mean_reverting", "choppy", "squeeze")


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


def _load_series(path: Path) -> np.ndarray | None:
    m = _load_matrix(path)
    if m is None:
        return None
    if m.shape[1] == 1:
        return m[:, 0]
    return np.nan_to_num(np.mean(m, axis=1), nan=0.0, posinf=0.0, neginf=0.0)


def _zscore_rolling(x: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(np.asarray(x, float).ravel())
    w = int(max(5, window))
    mu = s.rolling(w, min_periods=max(8, w // 4)).mean()
    sd = s.rolling(w, min_periods=max(8, w // 4)).std(ddof=1)
    z = (s - mu) / (sd + 1e-9)
    return z.fillna(0.0).values.astype(float)


def _rolling_lag1_autocorr(x: np.ndarray, window: int) -> np.ndarray:
    v = np.asarray(x, float).ravel()
    out = np.zeros_like(v, dtype=float)
    w = int(max(5, window))
    for t in range(len(v)):
        i0 = max(0, t - w + 1)
        seg = v[i0 : t + 1]
        if len(seg) < 8:
            out[t] = 0.0
            continue
        a = seg[1:]
        b = seg[:-1]
        if np.std(a, ddof=1) <= 1e-9 or np.std(b, ddof=1) <= 1e-9:
            out[t] = 0.0
            continue
        out[t] = float(np.corrcoef(a, b)[0, 1])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def classify_regimes_from_returns(returns: np.ndarray, lookback: int = 63) -> np.ndarray:
    r = np.asarray(returns, float).ravel()
    if r.size == 0:
        return np.asarray([], dtype=object)

    lb = int(max(10, lookback))
    ret_lb = pd.Series(r).rolling(lb, min_periods=max(10, lb // 3)).sum().fillna(0.0).values
    vol_lb = pd.Series(r).rolling(lb, min_periods=max(10, lb // 3)).std(ddof=1).fillna(0.0).values

    ret_z = _zscore_rolling(ret_lb, max(63, lb))
    vol_z = _zscore_rolling(vol_lb, max(63, lb))
    ac21 = _rolling_lag1_autocorr(r, window=21)

    labels = np.full(r.size, "choppy", dtype=object)
    labels[ret_z > 1.0] = "trending"
    squeeze_mask = (vol_z < -0.5) & (np.abs(ret_z) < 0.35)
    labels[squeeze_mask] = "squeeze"
    mr_mask = ac21 < -0.15
    labels[mr_mask] = "mean_reverting"
    # Preserve trending precedence where both conditions happen.
    labels[(ret_z > 1.0)] = "trending"
    return labels


def _regime_features(returns: np.ndarray, lookback: int = 63) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(returns, float).ravel()
    if r.size == 0:
        z = np.asarray([], dtype=float)
        return z, z, z
    lb = int(max(10, lookback))
    ret_lb = pd.Series(r).rolling(lb, min_periods=max(10, lb // 3)).sum().fillna(0.0).values
    vol_lb = pd.Series(r).rolling(lb, min_periods=max(10, lb // 3)).std(ddof=1).fillna(0.0).values
    ret_z = _zscore_rolling(ret_lb, max(63, lb))
    vol_z = _zscore_rolling(vol_lb, max(63, lb))
    ac21 = _rolling_lag1_autocorr(r, window=21)
    return ret_z, vol_z, ac21


def _fit_regime_thresholds(ret_z: np.ndarray, vol_z: np.ndarray, ac21: np.ndarray) -> dict[str, float]:
    rz = np.asarray(ret_z, float).ravel()
    vz = np.asarray(vol_z, float).ravel()
    ac = np.asarray(ac21, float).ravel()
    rz = rz[np.isfinite(rz)]
    vz = vz[np.isfinite(vz)]
    ac = ac[np.isfinite(ac)]

    if rz.size < 12 or vz.size < 12 or ac.size < 12:
        return {
            "trend_ret_z_min": 1.0,
            "squeeze_vol_z_max": -0.5,
            "squeeze_abs_ret_z_max": 0.35,
            "mean_reverting_ac_max": -0.15,
        }

    trend = float(np.clip(np.nanquantile(rz, 0.80), 0.8, 2.5))
    sq_vol = float(np.clip(np.nanquantile(vz, 0.30), -2.5, -0.1))
    sq_abs_ret = float(np.clip(np.nanquantile(np.abs(rz), 0.45), 0.15, 0.75))
    mr_ac = float(np.clip(np.nanquantile(ac, 0.20), -0.9, -0.05))
    return {
        "trend_ret_z_min": trend,
        "squeeze_vol_z_max": sq_vol,
        "squeeze_abs_ret_z_max": sq_abs_ret,
        "mean_reverting_ac_max": mr_ac,
    }


def _classify_with_thresholds(
    ret_z: np.ndarray,
    vol_z: np.ndarray,
    ac21: np.ndarray,
    thresholds: dict[str, float],
) -> np.ndarray:
    rz = np.asarray(ret_z, float).ravel()
    vz = np.asarray(vol_z, float).ravel()
    ac = np.asarray(ac21, float).ravel()
    n = int(min(rz.size, vz.size, ac.size))
    if n <= 0:
        return np.asarray([], dtype=object)
    rz = rz[:n]
    vz = vz[:n]
    ac = ac[:n]

    tr = float(thresholds.get("trend_ret_z_min", 1.0))
    sqv = float(thresholds.get("squeeze_vol_z_max", -0.5))
    sqr = float(thresholds.get("squeeze_abs_ret_z_max", 0.35))
    mr = float(thresholds.get("mean_reverting_ac_max", -0.15))

    labels = np.full(n, "choppy", dtype=object)
    labels[rz > tr] = "trending"
    squeeze_mask = (vz < sqv) & (np.abs(rz) < sqr)
    labels[squeeze_mask] = "squeeze"
    labels[ac < mr] = "mean_reverting"
    labels[rz > tr] = "trending"
    return labels


def _normalize_labels(raw: np.ndarray) -> np.ndarray:
    vals = np.asarray(raw).ravel()
    out = np.full(vals.size, "choppy", dtype=object)
    map_num = {0: "choppy", 1: "trending", 2: "mean_reverting", 3: "squeeze"}
    for i, x in enumerate(vals):
        sx = str(x).strip().lower()
        if sx in REGIMES:
            out[i] = sx
            continue
        try:
            xi = int(float(x))
        except Exception:
            out[i] = "choppy"
            continue
        out[i] = map_num.get(xi, "choppy")
    return out


def _load_regime_labels(path: Path, t: int) -> tuple[np.ndarray | None, str]:
    if not path.exists():
        return None, "fallback"
    try:
        df = pd.read_csv(path)
    except Exception:
        return None, "fallback"
    if df.empty:
        return None, "fallback"
    col = None
    for c in df.columns:
        if str(c).strip().lower() in {"regime", "label", "regime_label"}:
            col = c
            break
    if col is None:
        col = df.columns[-1]
    lab = _normalize_labels(df[col].values)
    if lab.size <= 0:
        return None, "fallback"
    if lab.size >= t:
        return lab[-t:], "file"
    out = np.full(t, "choppy", dtype=object)
    out[-lab.size :] = lab
    return out, "file_padded"


def _fit_bandit_weights(signals: np.ndarray, returns: np.ndarray, eta: float) -> np.ndarray:
    s = np.asarray(signals, float)
    r = np.asarray(returns, float).ravel()
    t, k = s.shape
    if t < 8 or k <= 0:
        return np.ones(k, dtype=float) / max(k, 1)
    idx = pd.RangeIndex(t)
    sig = {f"s{i}": pd.Series(s[:, i], index=idx) for i in range(k)}
    ret = pd.Series(r, index=idx)
    bandit_type = str(os.getenv("Q_BANDIT_TYPE", "thompson")).strip().lower()
    try:
        if bandit_type == "thompson":
            prior_file = str(os.getenv("Q_BANDIT_PRIOR_FILE", "")).strip() or None
            w = (
                ThompsonBandit(
                    n_arms=k,
                    decay=float(np.clip(float(os.getenv("Q_THOMPSON_DECAY", "0.995")), 0.90, 1.0)),
                    magnitude_scaling=str(os.getenv("Q_THOMPSON_MAGNITUDE_SCALING", "1")).strip().lower()
                    not in {"0", "false", "off", "no"},
                    prior_file=prior_file,
                )
                .fit(sig, ret)
                .get_weights()
            )
        else:
            w = ExpWeightsBandit(eta=float(eta)).fit(sig, ret).get_weights()
    except Exception:
        w = {}
    vec = np.asarray([float(w.get(f"s{i}", 0.0)) for i in range(k)], float)
    if not np.isfinite(vec).all() or vec.sum() <= 1e-9:
        vec = np.ones(k, dtype=float) / max(k, 1)
    else:
        vec = np.clip(vec, 0.0, np.inf)
        vec = vec / max(vec.sum(), 1e-12)
    return vec


def compute_regime_council_weights(
    council_votes: np.ndarray,
    returns: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    min_regime_days: int = 60,
    lookback: int = 63,
    train_min: int = 252,
    test_step: int = 21,
    embargo: int = 5,
    eta: float = 0.6,
    dynamic_regime_classifier: bool = False,
) -> tuple[np.ndarray, dict]:
    v = np.asarray(council_votes, float)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    y = np.asarray(returns, float).ravel()
    has_labels = labels is not None
    lab_in = _normalize_labels(labels) if has_labels else None
    t = min(v.shape[0], y.size, int(lab_in.size) if lab_in is not None else y.size)
    if t <= 0:
        return np.ones((0, 0), dtype=float), {
            "rows": 0,
            "cols": 0,
            "folds": [],
            "regime_fallback_counts": {r: 0 for r in REGIMES},
            "dynamic_regime_classifier": bool(dynamic_regime_classifier),
        }

    v = np.nan_to_num(v[:t], nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y[:t], nan=0.0, posinf=0.0, neginf=0.0)
    if lab_in is not None:
        lab = lab_in[:t].copy()
    else:
        lab = classify_regimes_from_returns(y, lookback=lookback).astype(object)

    k = v.shape[1]
    min_days = int(max(10, min_regime_days))
    lb = int(max(20, lookback))
    train_min = int(max(min_days + 20, train_min))
    test_step = int(max(5, test_step))
    emb = int(max(5, embargo))

    out = np.full((t, k), np.nan, dtype=float)
    fallback_counts = {r: 0 for r in REGIMES}
    fold_info: list[dict] = []
    dyn_cls = bool(dynamic_regime_classifier)
    ret_z, vol_z, ac21 = _regime_features(y, lookback=lb) if dyn_cls else (None, None, None)

    # Seed very-early rows with global train weights.
    seed_end = min(t, train_min)
    seed = _fit_bandit_weights(v[:seed_end], y[:seed_end], eta=eta)
    out[:seed_end] = seed.reshape(1, -1)
    if dyn_cls and ret_z is not None and vol_z is not None and ac21 is not None and seed_end > 0:
        seed_cls_end = min(t, max(lb + 10, seed_end))
        cls_win = max(126, 4 * lb)
        seed_cls_start = max(0, seed_cls_end - cls_win)
        seed_th = _fit_regime_thresholds(
            ret_z[seed_cls_start:seed_cls_end],
            vol_z[seed_cls_start:seed_cls_end],
            ac21[seed_cls_start:seed_cls_end],
        )
        lab[:seed_end] = _classify_with_thresholds(
            ret_z[:seed_end],
            vol_z[:seed_end],
            ac21[:seed_end],
            seed_th,
        )

    for train_end in range(train_min, t, test_step):
        test_end = min(t, train_end + test_step)
        if test_end <= train_end:
            continue

        # Nested split: first segment calibrates regime state context, second fits signals.
        classifier_end = max(lb + 10, int(0.60 * train_end))
        classifier_end = min(classifier_end, max(lb + 10, train_end - emb - 10))
        signal_start = min(train_end, classifier_end + emb)
        signal_idx = np.arange(signal_start, train_end, dtype=int)
        if signal_idx.size < 8:
            signal_idx = np.arange(max(0, train_end - max(min_days, 40)), train_end, dtype=int)

        fold_thresholds = None
        signal_labels = None
        test_labels = None
        classifier_start = None
        if dyn_cls and ret_z is not None and vol_z is not None and ac21 is not None:
            cls_win = max(126, 4 * lb)
            classifier_start = max(0, classifier_end - cls_win)
            fold_thresholds = _fit_regime_thresholds(
                ret_z[classifier_start:classifier_end],
                vol_z[classifier_start:classifier_end],
                ac21[classifier_start:classifier_end],
            )
            signal_labels = _classify_with_thresholds(
                ret_z[signal_idx],
                vol_z[signal_idx],
                ac21[signal_idx],
                fold_thresholds,
            )
            test_labels = _classify_with_thresholds(
                ret_z[train_end:test_end],
                vol_z[train_end:test_end],
                ac21[train_end:test_end],
                fold_thresholds,
            )

        global_w = _fit_bandit_weights(v[signal_idx], y[signal_idx], eta=eta)
        by_regime: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}
        for rg in REGIMES:
            if signal_labels is None:
                ridx = signal_idx[lab[signal_idx] == rg]
            else:
                ridx = signal_idx[np.asarray(signal_labels) == rg]
            counts[rg] = int(ridx.size)
            if ridx.size >= min_days:
                by_regime[rg] = _fit_bandit_weights(v[ridx], y[ridx], eta=eta)
            else:
                by_regime[rg] = global_w.copy()
                fallback_counts[rg] += 1

        for i in range(train_end, test_end):
            if test_labels is None:
                rg = str(lab[i]).strip().lower()
            else:
                rg = str(test_labels[i - train_end]).strip().lower()
                lab[i] = rg if rg in REGIMES else "choppy"
            if rg not in by_regime:
                out[i] = global_w
            else:
                out[i] = by_regime[rg]

        row = {
            "train_end": int(train_end),
            "test_end": int(test_end),
            "classifier_end": int(classifier_end),
            "signal_start": int(signal_start),
            "embargo_gap": int(max(0, signal_start - classifier_end)),
            "regime_counts_signal_train": counts,
        }
        if classifier_start is not None:
            row["classifier_start"] = int(classifier_start)
        if isinstance(fold_thresholds, dict):
            row["classifier_thresholds"] = {k: float(v) for k, v in fold_thresholds.items()}
        fold_info.append(row)

    # Fill any holes with nearest previous row, then uniform fallback.
    for i in range(t):
        if np.isfinite(out[i]).all():
            continue
        if i > 0 and np.isfinite(out[i - 1]).all():
            out[i] = out[i - 1]
        else:
            out[i] = np.ones(k, dtype=float) / max(k, 1)

    out = np.clip(out, 0.0, np.inf)
    row_sum = np.sum(out, axis=1, keepdims=True)
    out = np.divide(out, np.where(row_sum <= 1e-12, 1.0, row_sum))

    regime_means = {}
    for rg in REGIMES:
        m = lab == rg
        if int(np.count_nonzero(m)) > 0:
            regime_means[rg] = np.mean(out[m], axis=0).tolist()
        else:
            regime_means[rg] = []

    info = {
        "rows": int(t),
        "cols": int(k),
        "train_min": int(train_min),
        "test_step": int(test_step),
        "lookback": int(lb),
        "min_regime_days": int(min_days),
        "embargo": int(emb),
        "regime_counts": {r: int(np.count_nonzero(lab == r)) for r in REGIMES},
        "regime_fallback_counts": fallback_counts,
        "folds": fold_info,
        "regime_mean_weights": regime_means,
        "dynamic_regime_classifier": bool(dyn_cls),
    }
    return out, info


def _build_weight_candidate(base_weights: np.ndarray, council_votes: np.ndarray, regime_w: np.ndarray) -> np.ndarray:
    b = np.asarray(base_weights, float)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    v = np.asarray(council_votes, float)
    rw = np.asarray(regime_w, float)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    if rw.ndim == 1:
        rw = rw.reshape(-1, 1)
    l = min(b.shape[0], v.shape[0], rw.shape[0])
    if l <= 0:
        return np.ones((0, b.shape[1] if b.ndim == 2 else 0), dtype=float)
    # Regime-weighted council confidence in [-1, 1].
    conf = np.sum(np.tanh(v[:l]) * rw[:l], axis=1)
    conf = np.tanh(conf)
    scale = 1.0 + 0.15 * conf  # mild leverage tilt: [0.85, 1.15]
    out = b[:l] * scale.reshape(-1, 1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _first_base_weights() -> tuple[np.ndarray | None, str | None]:
    cands = [
        RUNS / "weights_regime.csv",
        RUNS / "weights_tail_blend.csv",
        RUNS / "portfolio_weights.csv",
        ROOT / "portfolio_weights.csv",
    ]
    for p in cands:
        m = _load_matrix(p)
        if m is not None:
            return m, str(p.relative_to(ROOT))
    return None, None


def main() -> int:
    enabled = str(os.getenv("Q_REGIME_COUNCIL_ENABLED", "0")).strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        print("… skip: regime council disabled (Q_REGIME_COUNCIL_ENABLED=0)")
        return 0

    lookback = int(np.clip(int(float(os.getenv("Q_REGIME_COUNCIL_LOOKBACK", "63"))), 20, 252))
    min_regime_days = int(np.clip(int(float(os.getenv("Q_REGIME_COUNCIL_MIN_REGIME_DAYS", "60"))), 20, 400))
    train_min = int(np.clip(int(float(os.getenv("Q_REGIME_COUNCIL_TRAIN_MIN", "252"))), 80, 1000))
    test_step = int(np.clip(int(float(os.getenv("Q_REGIME_COUNCIL_TEST_STEP", "21"))), 5, 252))
    embargo = int(np.clip(int(float(os.getenv("Q_REGIME_COUNCIL_EMBARGO", "5"))), 5, 30))
    eta = float(np.clip(float(os.getenv("Q_REGIME_COUNCIL_BANDIT_ETA", "0.6")), 0.05, 2.0))
    dyn_cls_enabled = str(os.getenv("Q_REGIME_COUNCIL_DYNAMIC_CLASSIFIER", "1")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    votes = _load_matrix(RUNS / "council_votes.csv")
    if votes is None:
        print("(!) Missing runs_plus/council_votes.csv; skipping regime council.")
        return 0

    r_asset = _load_matrix(RUNS / "asset_returns.csv")
    r_daily = _load_series(RUNS / "daily_returns.csv")
    if r_asset is not None:
        y = np.mean(r_asset, axis=1)
    elif r_daily is not None:
        y = r_daily
    else:
        print("(!) Missing returns inputs for regime council; skipping.")
        return 0

    t = min(votes.shape[0], y.size)
    votes = votes[:t]
    y = y[:t]

    labels, label_source = _load_regime_labels(RUNS / "regime_labels.csv", t=t)
    use_dynamic_classifier = False
    if labels is None and dyn_cls_enabled:
        label_source = "walkforward_fallback"
        use_dynamic_classifier = True
    elif labels is None:
        labels = classify_regimes_from_returns(y, lookback=lookback)
        label_source = "fallback_static"

    regime_w, info = compute_regime_council_weights(
        votes,
        y,
        labels,
        min_regime_days=min_regime_days,
        lookback=lookback,
        train_min=train_min,
        test_step=test_step,
        embargo=embargo,
        eta=eta,
        dynamic_regime_classifier=use_dynamic_classifier,
    )

    np.savetxt(RUNS / "regime_council_weights.csv", regime_w, delimiter=",")

    base_w, base_source = _first_base_weights()
    candidate_written = False
    if base_w is not None:
        cand = _build_weight_candidate(base_w, votes, regime_w)
        np.savetxt(RUNS / "weights_regime_council.csv", cand, delimiter=",")
        candidate_written = True

    info.update(
        {
            "enabled": True,
            "label_source": label_source,
            "base_weights_source": base_source,
            "weights_regime_council_written": bool(candidate_written),
            "bandit_eta": float(eta),
        }
    )
    (RUNS / "regime_council_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    _append_card(
        "Regime Council Weights ✔",
        (
            f"<p>rows={int(info.get('rows', 0))}, signals={int(info.get('cols', 0))}, "
            f"label_source={label_source}, candidate_written={candidate_written}.</p>"
            f"<p>fallbacks={int(sum(info.get('regime_fallback_counts', {}).values()))}, "
            f"embargo={int(info.get('embargo', embargo))} days.</p>"
        ),
    )

    print(f"✅ Wrote {RUNS/'regime_council_weights.csv'}")
    print(f"✅ Wrote {RUNS/'regime_council_info.json'}")
    if candidate_written:
        print(f"✅ Wrote {RUNS/'weights_regime_council.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
