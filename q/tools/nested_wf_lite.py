#!/usr/bin/env python3
"""
Nested walk-forward (purged/embargoed) over per-asset OOS streams.

Inputs (per asset directory under runs_plus):
  - oos_plus.csv preferred (ret + pos_plus)
  - oos.csv fallback (ret + pos)

Outputs:
  - runs_plus/nested_wf_summary.json
  - runs_plus/nested_wf_asset_table.csv
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)

TRAIN_MIN = int(os.environ.get("NWF_TRAIN_MIN", "504"))
OUTER_STEP = int(os.environ.get("NWF_OUTER_STEP", "63"))
INNER_MIN = int(os.environ.get("NWF_INNER_MIN", "252"))
INNER_STEP = int(os.environ.get("NWF_INNER_STEP", "21"))
INNER_FOLDS = int(os.environ.get("NWF_INNER_FOLDS", "4"))
EMBARGO = int(os.environ.get("NWF_EMBARGO", "5"))
PURGE = int(os.environ.get("NWF_PURGE", "3"))
MIN_ROWS = int(os.environ.get("NWF_MIN_ROWS", "900"))
MAX_ROWS = int(os.environ.get("NWF_MAX_ROWS", "3000"))
COST_BPS = float(os.environ.get("NWF_COST_BPS", "1.0"))
WINSOR_PCT = float(os.environ.get("NWF_WINSOR_PCT", "0.005"))


def _winsor(x: np.ndarray, p: float) -> np.ndarray:
    p = float(max(0.0, min(0.49, p)))
    if x.size == 0 or p <= 0:
        return x
    lo, hi = np.nanquantile(x, p), np.nanquantile(x, 1.0 - p)
    return np.clip(x, lo, hi)


def _sharpe(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    xx = _winsor(np.asarray(x, float), WINSOR_PCT)
    mu = float(np.nanmean(xx))
    sd = float(np.nanstd(xx, ddof=1)) + 1e-12
    return float(mu / sd * math.sqrt(252.0))


def _max_dd(pnl: np.ndarray) -> float:
    if pnl.size == 0:
        return 0.0
    eq = np.cumprod(1.0 + np.clip(np.asarray(pnl, float), -0.95, 0.95))
    peak = np.maximum.accumulate(eq)
    dd = (eq / np.maximum(peak, 1e-12)) - 1.0
    return float(np.nanmin(dd))


def _candidate_grid():
    caps = [0.60, 0.80, 1.00]
    deadbands = [0.00, 0.05]
    spans = [1, 3, 5]
    out = []
    for cap in caps:
        for db in deadbands:
            for span in spans:
                out.append({"cap": float(cap), "deadband": float(db), "span": int(span)})
    return out


def _transform_pos(base_pos: np.ndarray, cap: float, deadband: float, span: int) -> np.ndarray:
    p = np.clip(np.asarray(base_pos, float), -abs(cap), abs(cap))
    if deadband > 0:
        p[np.abs(p) < float(deadband)] = 0.0
    if int(span) > 1:
        p = pd.Series(p).ewm(span=int(span), adjust=False).mean().values
        p = np.clip(p, -abs(cap), abs(cap))
    return p


def _pnl_from_pos_ret(pos: np.ndarray, ret: np.ndarray, cost_bps: float) -> np.ndarray:
    pos = np.asarray(pos, float)
    ret = np.asarray(ret, float)
    pos_lag = np.roll(pos, 1)
    if pos_lag.size:
        pos_lag[0] = 0.0
    turnover = np.abs(np.diff(np.r_[0.0, pos]))
    cost = turnover * (float(cost_bps) / 10000.0)
    pnl = pos_lag * ret - cost
    return np.clip(pnl, -0.95, 0.95)


def _load_asset_stream(asset_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    p_plus = asset_dir / "oos_plus.csv"
    p_oos = asset_dir / "oos.csv"

    if p_plus.exists():
        df = pd.read_csv(p_plus)
        if {"ret", "pos_plus"}.issubset(df.columns):
            ret = pd.to_numeric(df["ret"], errors="coerce").fillna(0.0).clip(-0.20, 0.20).values
            pos = pd.to_numeric(df["pos_plus"], errors="coerce").fillna(0.0).clip(-1.0, 1.0).values
            return ret.astype(float), pos.astype(float)

    if p_oos.exists():
        df = pd.read_csv(p_oos)
        if {"ret", "pos"}.issubset(df.columns):
            ret = pd.to_numeric(df["ret"], errors="coerce").fillna(0.0).clip(-0.20, 0.20).values
            pos = pd.to_numeric(df["pos"], errors="coerce").fillna(0.0).clip(-1.0, 1.0).values
            return ret.astype(float), pos.astype(float)

    return None


def _inner_score(ret_train: np.ndarray, pos_train: np.ndarray, cfg: dict) -> float:
    n = len(ret_train)
    if n < max(120, INNER_MIN + INNER_STEP):
        return -1e9

    def _train_slices(vs: int, ve: int, gap: int) -> tuple[np.ndarray, np.ndarray]:
        left_end = max(0, vs - gap)
        right_start = min(n, ve + gap)
        if left_end <= 0 and right_start >= n:
            return np.array([], dtype=float), np.array([], dtype=float)
        if left_end <= 0:
            return ret_train[right_start:], pos_train[right_start:]
        if right_start >= n:
            return ret_train[:left_end], pos_train[:left_end]
        r = np.concatenate([ret_train[:left_end], ret_train[right_start:]], axis=0)
        p = np.concatenate([pos_train[:left_end], pos_train[right_start:]], axis=0)
        return r, p

    # Multi-fold purged/embargoed inner score to reduce overfitting to one tail segment.
    folds = int(max(1, INNER_FOLDS))
    gap = int(max(0, EMBARGO + PURGE))
    scores = []
    for k in range(folds):
        val_end = n - k * INNER_STEP
        val_start = val_end - INNER_STEP
        if val_start < 0:
            break
        # Embargo around validation edges.
        val_start = min(val_start + EMBARGO, val_end - 1)
        if val_end - val_start < 10:
            continue

        tr_r, tr_p = _train_slices(val_start, val_end, gap)
        if len(tr_r) < INNER_MIN:
            continue

        p = _transform_pos(pos_train[val_start:val_end], cfg["cap"], cfg["deadband"], cfg["span"])
        r = ret_train[val_start:val_end]
        pnl = _pnl_from_pos_ret(p, r, COST_BPS)
        sh_val = _sharpe(pnl)

        p_tr = _transform_pos(tr_p, cfg["cap"], cfg["deadband"], cfg["span"])
        pnl_tr = _pnl_from_pos_ret(p_tr, tr_r, COST_BPS)
        sh_tr = _sharpe(pnl_tr)

        # Penalize unstable configs that only spike on one fold.
        gap_pen = 0.20 * abs(sh_val - sh_tr)
        dd_pen = 0.05 * abs(_max_dd(pnl))
        fold_score = sh_val - gap_pen - dd_pen
        scores.append(fold_score)
    if not scores:
        return -1e9
    # Favor high average with lower variance across folds.
    sc = np.asarray(scores, float)
    return float(np.mean(sc) - 0.15 * np.std(sc, ddof=1) if len(sc) > 1 else np.mean(sc))


def run_asset_nested(ret: np.ndarray, pos: np.ndarray) -> tuple[dict, list[str]]:
    n = len(ret)
    if n < MIN_ROWS:
        return {}, []
    if MAX_ROWS > 0 and n > MAX_ROWS:
        ret = ret[-MAX_ROWS:]
        pos = pos[-MAX_ROWS:]
        n = len(ret)

    cfgs = _candidate_grid()
    chosen = []
    oos_chunks = []
    ret_chunks = []
    outer_attempts = 0
    train_ratios = []
    t = max(TRAIN_MIN, 120)
    while t < n:
        test_start = t
        test_end = min(n, t + OUTER_STEP)
        if test_end - test_start < 10:
            break
        outer_attempts += 1

        train_end = max(0, test_start - EMBARGO)
        if train_end < max(INNER_MIN + INNER_STEP, 160):
            t += OUTER_STEP
            continue

        ret_train = ret[:train_end]
        pos_train = pos[:train_end]

        best_cfg = None
        best_score = -1e12
        for cfg in cfgs:
            sc = _inner_score(ret_train, pos_train, cfg)
            if sc > best_score:
                best_score = sc
                best_cfg = cfg

        if best_cfg is None:
            t += OUTER_STEP
            continue

        p_test = _transform_pos(pos[test_start:test_end], best_cfg["cap"], best_cfg["deadband"], best_cfg["span"])
        r_test = ret[test_start:test_end]
        pnl_test = _pnl_from_pos_ret(p_test, r_test, COST_BPS)
        oos_chunks.append(pnl_test)
        ret_chunks.append(r_test)
        chosen.append(f"cap={best_cfg['cap']:.2f}|db={best_cfg['deadband']:.2f}|span={best_cfg['span']}")
        train_ratios.append(float(train_end) / max(1.0, float(n)))

        t += OUTER_STEP

    if not oos_chunks:
        return {}, []

    oos = np.concatenate(oos_chunks)
    ret_oos = np.concatenate(ret_chunks) if ret_chunks else np.array([], dtype=float)
    if oos.size and ret_oos.size == oos.size:
        hit = float((np.sign(oos) == np.sign(ret_oos)).mean())
    else:
        hit = None
    out = {
        "oos_sharpe": _sharpe(oos),
        "oos_maxDD": _max_dd(oos),
        "oos_mean_daily": float(np.mean(oos)) if oos.size else 0.0,
        "oos_n": int(oos.size),
        "hit": hit,
        "outer_folds_used": int(len(chosen)),
        "outer_folds_attempted": int(outer_attempts),
        "outer_fold_utilization": float(len(chosen) / max(1, outer_attempts)),
        "train_ratio_mean": float(np.mean(train_ratios)) if train_ratios else None,
    }
    return out, chosen


if __name__ == "__main__":
    rows = []
    cfg_counter = Counter()

    for asset_dir in sorted(RUNS.iterdir()):
        if not asset_dir.is_dir():
            continue
        loaded = _load_asset_stream(asset_dir)
        if loaded is None:
            continue
        ret, pos = loaded
        stats, chosen = run_asset_nested(ret, pos)
        if not stats:
            continue
        for c in chosen:
            cfg_counter[c] += 1
        rows.append({"asset": asset_dir.name, **stats, "outer_folds": int(len(chosen))})

    df = pd.DataFrame(rows).sort_values("oos_sharpe", ascending=False) if rows else pd.DataFrame()
    if not df.empty:
        df.to_csv(RUNS / "nested_wf_asset_table.csv", index=False)

    summary = {
        "assets": int(len(df)) if not df.empty else 0,
        "avg_oos_sharpe": float(df["oos_sharpe"].mean()) if not df.empty else None,
        "median_oos_sharpe": float(df["oos_sharpe"].median()) if not df.empty else None,
        "avg_oos_maxDD": float(df["oos_maxDD"].mean()) if not df.empty else None,
        "avg_hit": float(df["hit"].mean()) if not df.empty else None,
        "avg_outer_fold_utilization": float(df["outer_fold_utilization"].mean()) if (not df.empty and "outer_fold_utilization" in df.columns) else None,
        "median_outer_fold_utilization": float(df["outer_fold_utilization"].median()) if (not df.empty and "outer_fold_utilization" in df.columns) else None,
        "low_utilization_assets": int(((df["outer_fold_utilization"] < 0.50).sum())) if (not df.empty and "outer_fold_utilization" in df.columns) else 0,
        "avg_train_ratio_mean": float(df["train_ratio_mean"].mean()) if (not df.empty and "train_ratio_mean" in df.columns) else None,
        "top_configs": [{"config": k, "count": int(v)} for k, v in cfg_counter.most_common(8)],
        "params": {
            "train_min": TRAIN_MIN,
            "outer_step": OUTER_STEP,
            "inner_min": INNER_MIN,
            "inner_step": INNER_STEP,
            "inner_folds": INNER_FOLDS,
            "embargo": EMBARGO,
            "purge": PURGE,
            "max_rows": MAX_ROWS,
            "cost_bps": COST_BPS,
            "winsor_pct": WINSOR_PCT,
            "purge_embargo_ratio": float((PURGE + EMBARGO) / max(1, OUTER_STEP)),
        },
    }

    outp = RUNS / "nested_wf_summary.json"
    outp.write_text(json.dumps(summary, indent=2))
    print(f"✅ Wrote {outp}")
    if not df.empty:
        print(f"✅ Wrote {RUNS/'nested_wf_asset_table.csv'}")
        print(
            "Nested WF summary: "
            f"assets={summary['assets']}, "
            f"avg_sharpe={summary['avg_oos_sharpe']:.3f}, "
            f"avg_maxDD={summary['avg_oos_maxDD']:.3f}"
        )
