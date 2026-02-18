#!/usr/bin/env python3
"""
Governor parameter sweep.

Tunes key governor knobs (floor, shock alpha, concentration caps, core governor strengths) and writes:
  - runs_plus/governor_param_sweep.csv
  - runs_plus/governor_params_profile.json

The chosen profile is then applied by build_final_portfolio.py automatically.
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)
PYTHON = str(Path(sys.executable))


def _run(cmd: list[str], env: dict[str, str]) -> None:
    p = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{p.stdout}\n"
            f"stderr:\n{p.stderr}"
        )


def _build_make_daily(env_overrides: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items()})
    _run([PYTHON, str(ROOT / "tools" / "build_final_portfolio.py")], env)
    _run([PYTHON, str(ROOT / "tools" / "make_daily_from_weights.py")], env)


def _run_strict_oos(env_overrides: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items()})
    _run([PYTHON, str(ROOT / "tools" / "run_strict_oos_validation.py")], env)


def _metrics() -> dict:
    rp = RUNS / "daily_returns.csv"
    wp = RUNS / "portfolio_weights_final.csv"
    if not rp.exists():
        raise RuntimeError("Missing runs_plus/daily_returns.csv")
    if not wp.exists():
        raise RuntimeError("Missing runs_plus/portfolio_weights_final.csv")
    r = np.asarray(np.loadtxt(rp, delimiter=","), float).ravel()
    w = np.asarray(np.loadtxt(wp, delimiter=","), float)
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    mu = float(np.nanmean(r))
    sd = float(np.nanstd(r, ddof=1)) if r.size > 1 else 0.0
    sh = float((mu / (sd + 1e-12)) * np.sqrt(252.0))
    hit = float(np.sum(r > 0.0) / max(1, r.size))
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    mdd = float(np.min(eq - peak))
    turnover = float(np.mean(np.sum(np.abs(np.diff(w, axis=0)), axis=1))) if w.shape[0] > 1 else 0.0
    gross = float(np.mean(np.sum(np.abs(w), axis=1))) if w.size else 0.0
    return {
        "sharpe": sh,
        "hit_rate": hit,
        "max_drawdown": mdd,
        "turnover_mean": turnover,
        "gross_mean": gross,
        "n": int(r.size),
    }


def _strict_oos_metrics() -> dict:
    p = RUNS / "strict_oos_validation.json"
    if not p.exists():
        return {"sharpe": 0.0, "hit_rate": 0.0, "max_drawdown": 0.0, "n": 0, "source": "missing"}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"sharpe": 0.0, "hit_rate": 0.0, "max_drawdown": 0.0, "n": 0, "source": "error"}
    mode = str(os.getenv("Q_SWEEP_OOS_MODE", "robust_then_single")).strip().lower()
    net = obj.get("metrics_oos_net", {}) if isinstance(obj, dict) and isinstance(obj.get("metrics_oos_net"), dict) else {}
    robust = obj.get("metrics_oos_robust", {}) if isinstance(obj, dict) and isinstance(obj.get("metrics_oos_robust"), dict) else {}
    if mode == "single":
        src, m = "metrics_oos_net", net
    elif mode == "robust":
        src, m = "metrics_oos_robust", robust
    elif robust:
        src, m = "metrics_oos_robust", robust
    else:
        src, m = "metrics_oos_net", net
    if not isinstance(m, dict):
        return {"sharpe": 0.0, "hit_rate": 0.0, "max_drawdown": 0.0, "n": 0, "source": src}
    return {
        "sharpe": float(m.get("sharpe", 0.0)),
        "hit_rate": float(m.get("hit_rate", 0.0)),
        "max_drawdown": float(m.get("max_drawdown", 0.0)),
        "n": int(m.get("n", 0)),
        "source": src,
    }


def _objective(cur: dict, base: dict) -> tuple[float, dict]:
    cur_sh = float(cur["oos_sharpe"]) if "oos_sharpe" in cur else float(cur.get("sharpe", 0.0))
    base_sh = float(base["oos_sharpe"]) if "oos_sharpe" in base else float(base.get("sharpe", 0.0))
    cur_hit = float(cur["oos_hit_rate"]) if "oos_hit_rate" in cur else float(cur.get("hit_rate", 0.0))
    base_hit = float(base["oos_hit_rate"]) if "oos_hit_rate" in base else float(base.get("hit_rate", 0.0))
    cur_dd = abs(float(cur["oos_max_drawdown"])) if "oos_max_drawdown" in cur else abs(float(cur.get("max_drawdown", 0.0)))
    base_dd = max(
        1e-9,
        abs(float(base["oos_max_drawdown"])) if "oos_max_drawdown" in base else abs(float(base.get("max_drawdown", 0.0))),
    )
    dd_ratio = cur_dd / base_dd
    hit_delta = cur_hit - base_hit
    sharpe_delta = cur_sh - base_sh
    turn_ratio = float(cur["turnover_mean"]) / max(1e-9, float(base["turnover_mean"]))

    min_oos_sharpe = float(np.clip(float(os.getenv("Q_SWEEP_MIN_OOS_SHARPE", "1.00")), -2.0, 10.0))
    min_oos_hit = float(np.clip(float(os.getenv("Q_SWEEP_MIN_OOS_HIT", "0.49")), 0.0, 1.0))
    max_oos_abs_mdd = float(np.clip(float(os.getenv("Q_SWEEP_MAX_OOS_ABS_MDD", "0.10")), 0.001, 2.0))
    min_oos_n = int(np.clip(int(float(os.getenv("Q_SWEEP_MIN_OOS_N", "252"))), 1, 1000000))

    # Strict-OOS-first objective with target-aware penalties and hard vetoes.
    target_gap_sharpe = max(0.0, min_oos_sharpe - cur_sh)
    target_gap_hit = max(0.0, min_oos_hit - cur_hit)
    penalty = 0.012 * max(0.0, dd_ratio - 1.0)
    penalty += 0.006 * max(0.0, turn_ratio - 2.0)
    penalty += 0.010 * max(0.0, -hit_delta)
    penalty += 0.65 * target_gap_sharpe
    penalty += 0.40 * target_gap_hit
    score = cur_sh - penalty

    oos_n = int(cur.get("oos_n", cur.get("n", 0)))
    veto = (
        dd_ratio > 2.5
        or cur_dd > (1.75 * max_oos_abs_mdd)
        or cur_hit < (min_oos_hit - 0.06)
        or cur_sh < (min_oos_sharpe - 0.60)
        or oos_n < min_oos_n
    )
    if veto:
        score -= 1.0
    complexity_penalty, complexity_detail = _complexity_penalty(cur, base)
    score -= complexity_penalty
    detail = {
        "objective_sharpe": float(cur_sh),
        "objective_hit_rate": float(cur_hit),
        "objective_sharpe_delta": float(sharpe_delta),
        "dd_ratio": float(dd_ratio),
        "hit_delta": float(hit_delta),
        "turnover_ratio": float(turn_ratio),
        "target_gap_sharpe": float(target_gap_sharpe),
        "target_gap_hit_rate": float(target_gap_hit),
        "target_max_abs_mdd": float(max_oos_abs_mdd),
        "target_min_oos_n": int(min_oos_n),
        "oos_n": int(oos_n),
        "penalty": float(penalty),
        "complexity_penalty": float(complexity_penalty),
        "veto": bool(veto),
    }
    detail.update(complexity_detail)
    return score, detail


def _complexity_penalty(cur: dict, base: dict) -> tuple[float, dict]:
    lam = float(np.clip(float(os.getenv("Q_SWEEP_COMPLEXITY_PENALTY", "0.08")), 0.0, 5.0))
    strength_keys = [
        "meta_execution_gate_strength",
        "council_gate_strength",
        "meta_mix_leverage_strength",
        "meta_reliability_strength",
        "global_governor_strength",
        "heartbeat_scaler_strength",
        "quality_governor_strength",
        "regime_moe_strength",
        "uncertainty_sizing_strength",
        "vol_target_strength",
        "hit_gate_strength",
    ]
    strength_raw = 0.0
    strength_n = 0
    for k in strength_keys:
        if k in cur:
            try:
                strength_raw += abs(float(cur.get(k, 1.0)) - 1.0)
                strength_n += 1
            except Exception:
                continue
    strength_raw = float(strength_raw / max(1, strength_n))

    struct_raw = 0.0
    struct_raw += abs(float(cur.get("runtime_total_floor", base.get("runtime_total_floor", 0.18))) - float(base.get("runtime_total_floor", 0.18))) / 0.10
    struct_raw += abs(float(cur.get("shock_alpha", base.get("shock_alpha", 0.35))) - float(base.get("shock_alpha", 0.35))) / 0.20
    struct_raw += abs(float(cur.get("rank_sleeve_blend", 0.0))) / 0.15
    struct_raw += abs(float(cur.get("low_vol_sleeve_blend", 0.0))) / 0.15
    struct_raw += abs(float(cur.get("signal_deadzone", 0.0))) / 0.01
    struct_raw = float(struct_raw / 5.0)

    raw = float(strength_raw + 0.50 * struct_raw)
    penalty = float(lam * raw)
    return penalty, {
        "complexity_raw_strength": float(strength_raw),
        "complexity_raw_structure": float(struct_raw),
        "complexity_raw_total": float(raw),
    }


def _profile_from_row(row: dict) -> dict:
    return {
        "runtime_total_floor": float(row["runtime_total_floor"]),
        "shock_alpha": float(row["shock_alpha"]),
        "rank_sleeve_blend": float(row.get("rank_sleeve_blend", 0.0)),
        "low_vol_sleeve_blend": float(row.get("low_vol_sleeve_blend", 0.0)),
        "meta_execution_gate_strength": float(row.get("meta_execution_gate_strength", 1.0)),
        "council_gate_strength": float(row.get("council_gate_strength", 1.0)),
        "meta_mix_leverage_strength": float(row.get("meta_mix_leverage_strength", 1.0)),
        "meta_reliability_strength": float(row.get("meta_reliability_strength", 1.0)),
        "global_governor_strength": float(row.get("global_governor_strength", 1.0)),
        "heartbeat_scaler_strength": float(row.get("heartbeat_scaler_strength", 1.0)),
        "quality_governor_strength": float(row.get("quality_governor_strength", 1.0)),
        "regime_moe_strength": float(row.get("regime_moe_strength", 1.0)),
        "uncertainty_sizing_strength": float(row.get("uncertainty_sizing_strength", 1.0)),
        "vol_target_strength": float(row.get("vol_target_strength", 1.0)),
        "hit_gate_strength": float(row.get("hit_gate_strength", 1.0)),
        "hit_gate_threshold": float(row.get("hit_gate_threshold", 0.50)),
        "signal_deadzone": float(row.get("signal_deadzone", 0.0)),
        "use_concentration_governor": bool(int(row["use_concentration_governor"])),
        "concentration_top1_cap": float(row["concentration_top1_cap"]),
        "concentration_top3_cap": float(row["concentration_top3_cap"]),
        "concentration_max_hhi": float(row["concentration_max_hhi"]),
    }


def _row_from_params(params: dict, metrics: dict, score: float, score_detail: dict) -> dict:
    out = {**params, **metrics}
    out["score"] = float(score)
    out.update(score_detail)
    return out


def _csv_write(rows: list[dict], outp: Path) -> None:
    cols = [
        "stage",
        "runtime_total_floor",
        "shock_alpha",
        "rank_sleeve_blend",
        "low_vol_sleeve_blend",
        "meta_execution_gate_strength",
        "council_gate_strength",
        "meta_mix_leverage_strength",
        "meta_reliability_strength",
        "global_governor_strength",
        "heartbeat_scaler_strength",
        "quality_governor_strength",
        "regime_moe_strength",
        "uncertainty_sizing_strength",
        "vol_target_strength",
        "hit_gate_strength",
        "hit_gate_threshold",
        "signal_deadzone",
        "use_concentration_governor",
        "concentration_top1_cap",
        "concentration_top3_cap",
        "concentration_max_hhi",
        "sharpe",
        "hit_rate",
        "max_drawdown",
        "oos_sharpe",
        "oos_hit_rate",
        "oos_max_drawdown",
        "oos_n",
        "oos_source",
        "turnover_mean",
        "gross_mean",
        "score",
        "objective_sharpe",
        "objective_hit_rate",
        "objective_sharpe_delta",
        "dd_ratio",
        "hit_delta",
        "turnover_ratio",
        "target_gap_sharpe",
        "target_gap_hit_rate",
        "target_max_abs_mdd",
        "target_min_oos_n",
        "penalty",
        "complexity_penalty",
        "complexity_raw_strength",
        "complexity_raw_structure",
        "complexity_raw_total",
        "veto",
    ]
    with outp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def _env_from_params(params: dict) -> dict[str, str]:
    return {
        "Q_RUNTIME_TOTAL_FLOOR": str(params["runtime_total_floor"]),
        "Q_SHOCK_ALPHA": str(params["shock_alpha"]),
        "Q_RANK_SLEEVE_BLEND": str(params.get("rank_sleeve_blend", 0.0)),
        "Q_LOW_VOL_SLEEVE_BLEND": str(params.get("low_vol_sleeve_blend", 0.0)),
        "Q_META_EXECUTION_GATE_STRENGTH": str(params.get("meta_execution_gate_strength", 1.0)),
        "Q_COUNCIL_GATE_STRENGTH": str(params.get("council_gate_strength", 1.0)),
        "Q_META_MIX_LEVERAGE_STRENGTH": str(params.get("meta_mix_leverage_strength", 1.0)),
        "Q_META_RELIABILITY_STRENGTH": str(params.get("meta_reliability_strength", 1.0)),
        "Q_GLOBAL_GOVERNOR_STRENGTH": str(params.get("global_governor_strength", 1.0)),
        "Q_HEARTBEAT_SCALER_STRENGTH": str(params.get("heartbeat_scaler_strength", 1.0)),
        "Q_QUALITY_GOVERNOR_STRENGTH": str(params.get("quality_governor_strength", 1.0)),
        "Q_REGIME_MOE_STRENGTH": str(params.get("regime_moe_strength", 1.0)),
        "Q_UNCERTAINTY_SIZING_STRENGTH": str(params.get("uncertainty_sizing_strength", 1.0)),
        "Q_VOL_TARGET_STRENGTH": str(params.get("vol_target_strength", 1.0)),
        "Q_HIT_GATE_STRENGTH": str(params.get("hit_gate_strength", 1.0)),
        "Q_HIT_GATE_THRESHOLD": str(params.get("hit_gate_threshold", 0.50)),
        "Q_SIGNAL_DEADZONE": str(params.get("signal_deadzone", 0.0)),
        "Q_USE_CONCENTRATION_GOV": str(params["use_concentration_governor"]),
        "Q_CONCENTRATION_TOP1_CAP": str(params["concentration_top1_cap"]),
        "Q_CONCENTRATION_TOP3_CAP": str(params["concentration_top3_cap"]),
        "Q_CONCENTRATION_MAX_HHI": str(params["concentration_max_hhi"]),
    }


def _evaluate_candidate(params: dict, base: dict, rows: list[dict]) -> tuple[float, dict, dict]:
    _build_make_daily(_env_from_params(params))
    _run_strict_oos(_env_from_params(params))
    full = _metrics()
    oos = _strict_oos_metrics()
    m = {
        **full,
        "oos_sharpe": float(oos["sharpe"]),
        "oos_hit_rate": float(oos["hit_rate"]),
        "oos_max_drawdown": float(oos["max_drawdown"]),
        "oos_n": int(oos["n"]),
        "oos_source": str(oos.get("source", "")),
    }
    score, detail = _objective(m, base)
    row = _row_from_params(params, m, score, detail)
    rows.append(row)
    return score, detail, row


def main() -> int:
    if not (RUNS / "asset_returns.csv").exists():
        print("(!) Missing runs_plus/asset_returns.csv. Run tools/rebuild_asset_matrix.py first.")
        return 0

    # Compact staged grid: coarse core knobs -> coarse strengths -> local refinement.
    floors = [0.00, 0.05, 0.10, 0.15]
    shocks = [0.20, 0.35, 0.50, 0.65]
    conc_presets = [
        {"use_concentration_governor": 1, "concentration_top1_cap": 0.16, "concentration_top3_cap": 0.38, "concentration_max_hhi": 0.12},
        {"use_concentration_governor": 1, "concentration_top1_cap": 0.18, "concentration_top3_cap": 0.42, "concentration_max_hhi": 0.14},
        {"use_concentration_governor": 0, "concentration_top1_cap": 0.18, "concentration_top3_cap": 0.42, "concentration_max_hhi": 0.14},
    ]
    meta_execution_gate_strengths = [0.0, 0.5, 1.0]
    council_gate_strengths = [0.85, 1.00]
    meta_mix_leverage_strengths = [0.90, 1.00]
    meta_reliability_strengths = [0.90, 1.00]
    global_governor_strengths = [1.00, 1.15]
    heartbeat_scaler_strengths = [0.0, 0.5, 1.0]
    quality_governor_strengths = [1.00, 1.20]
    rank_sleeve_blends = [0.00, 0.05, 0.10]
    low_vol_sleeve_blends = [0.00, 0.03, 0.06]
    regime_moe_strengths = [0.0, 0.5, 1.0, 1.25]
    uncertainty_sizing_strengths = [0.5, 0.75, 1.00, 1.25]
    vol_target_strengths = [0.0, 0.5, 1.0, 1.25]
    hit_gate_strengths = [0.0, 0.5, 1.0]
    hit_gate_thresholds = [0.49, 0.52, 0.55]
    signal_deadzones = [0.0, 0.0005, 0.0015]

    rows: list[dict] = []
    try:
        # Baseline under current profile.
        _build_make_daily()
        _run_strict_oos()
        full = _metrics()
        oos = _strict_oos_metrics()
        base = {
            **full,
            "oos_sharpe": float(oos["sharpe"]),
            "oos_hit_rate": float(oos["hit_rate"]),
            "oos_max_drawdown": float(oos["max_drawdown"]),
            "oos_n": int(oos["n"]),
        }

        best_score = -1e9
        best_row = None
        # Stage 1: coarse search on core controls.
        for floor, shock, conc in itertools.product(floors, shocks, conc_presets):
            params = {
                "stage": "core",
                "runtime_total_floor": float(floor),
                "shock_alpha": float(shock),
                "rank_sleeve_blend": 0.0,
                "low_vol_sleeve_blend": 0.0,
                "meta_execution_gate_strength": 1.0,
                "council_gate_strength": 1.0,
                "meta_mix_leverage_strength": 1.0,
                "meta_reliability_strength": 1.0,
                "global_governor_strength": 1.0,
                "heartbeat_scaler_strength": 1.0,
                "quality_governor_strength": 1.0,
                "regime_moe_strength": 1.0,
                "uncertainty_sizing_strength": 1.0,
                "vol_target_strength": 1.0,
                "hit_gate_strength": 1.0,
                "hit_gate_threshold": 0.50,
                "signal_deadzone": 0.0,
                "use_concentration_governor": int(conc["use_concentration_governor"]),
                "concentration_top1_cap": float(conc["concentration_top1_cap"]),
                "concentration_top3_cap": float(conc["concentration_top3_cap"]),
                "concentration_max_hhi": float(conc["concentration_max_hhi"]),
            }
            score, _detail, row = _evaluate_candidate(params, base, rows)
            if score > best_score:
                best_score = score
                best_row = row

        if best_row is None:
            print("(!) No sweep candidates evaluated.")
            return 1

        # Stage 2: coordinate strength search (faster than full cartesian product).
        if best_row is not None:
            cur = dict(best_row)
            line_grids = [
                ("meta_execution_gate_strength", meta_execution_gate_strengths),
                ("council_gate_strength", council_gate_strengths),
                ("meta_mix_leverage_strength", meta_mix_leverage_strengths),
                ("meta_reliability_strength", meta_reliability_strengths),
                ("global_governor_strength", global_governor_strengths),
                ("heartbeat_scaler_strength", heartbeat_scaler_strengths),
                ("quality_governor_strength", quality_governor_strengths),
                ("rank_sleeve_blend", rank_sleeve_blends),
                ("low_vol_sleeve_blend", low_vol_sleeve_blends),
                ("regime_moe_strength", regime_moe_strengths),
                ("uncertainty_sizing_strength", uncertainty_sizing_strengths),
                ("vol_target_strength", vol_target_strengths),
                ("hit_gate_strength", hit_gate_strengths),
                ("hit_gate_threshold", hit_gate_thresholds),
                ("signal_deadzone", signal_deadzones),
            ]
            for _pass in range(2):
                improved = False
                for key, values in line_grids:
                    for val in values:
                        params = {
                            "stage": "strengths",
                            "runtime_total_floor": float(cur["runtime_total_floor"]),
                            "shock_alpha": float(cur["shock_alpha"]),
                            "rank_sleeve_blend": float(cur.get("rank_sleeve_blend", 0.0)),
                            "low_vol_sleeve_blend": float(cur.get("low_vol_sleeve_blend", 0.0)),
                            "meta_execution_gate_strength": float(cur["meta_execution_gate_strength"]),
                            "council_gate_strength": float(cur["council_gate_strength"]),
                            "meta_mix_leverage_strength": float(cur["meta_mix_leverage_strength"]),
                            "meta_reliability_strength": float(cur["meta_reliability_strength"]),
                            "global_governor_strength": float(cur["global_governor_strength"]),
                            "heartbeat_scaler_strength": float(cur.get("heartbeat_scaler_strength", 1.0)),
                            "quality_governor_strength": float(cur["quality_governor_strength"]),
                            "regime_moe_strength": float(cur.get("regime_moe_strength", 1.0)),
                            "uncertainty_sizing_strength": float(cur.get("uncertainty_sizing_strength", 1.0)),
                            "vol_target_strength": float(cur.get("vol_target_strength", 1.0)),
                            "hit_gate_strength": float(cur.get("hit_gate_strength", 1.0)),
                            "hit_gate_threshold": float(cur.get("hit_gate_threshold", 0.50)),
                            "signal_deadzone": float(cur.get("signal_deadzone", 0.0)),
                            "use_concentration_governor": int(cur["use_concentration_governor"]),
                            "concentration_top1_cap": float(cur["concentration_top1_cap"]),
                            "concentration_top3_cap": float(cur["concentration_top3_cap"]),
                            "concentration_max_hhi": float(cur["concentration_max_hhi"]),
                        }
                        params[key] = float(val)
                        score, _detail, row = _evaluate_candidate(params, base, rows)
                        if score > best_score:
                            best_score = score
                            best_row = row
                            cur = dict(row)
                            improved = True
                if not improved:
                    break

        # Stage 3: coordinate local refinement around stage-2 best.
        if best_row is not None:
            cur = dict(best_row)
            specs = [
                ("shock_alpha", 0.05, 0.0, 1.0),
                ("rank_sleeve_blend", 0.02, 0.0, 0.60),
                ("low_vol_sleeve_blend", 0.02, 0.0, 0.35),
                ("meta_execution_gate_strength", 0.05, 0.0, 1.4),
                ("council_gate_strength", 0.05, 0.6, 1.4),
                ("meta_mix_leverage_strength", 0.05, 0.7, 1.3),
                ("meta_reliability_strength", 0.05, 0.7, 1.3),
                ("global_governor_strength", 0.05, 0.8, 1.4),
                ("heartbeat_scaler_strength", 0.05, 0.0, 1.4),
                ("quality_governor_strength", 0.05, 0.8, 1.4),
                ("regime_moe_strength", 0.05, 0.0, 2.0),
                ("uncertainty_sizing_strength", 0.05, 0.0, 2.0),
                ("vol_target_strength", 0.05, 0.0, 2.0),
                ("hit_gate_strength", 0.05, 0.0, 2.0),
                ("hit_gate_threshold", 0.01, 0.35, 0.75),
                ("signal_deadzone", 0.0004, 0.0, 0.020),
            ]
            for _pass in range(2):
                improved = False
                for key, step, lo, hi in specs:
                    c = float(cur[key])
                    vals = sorted(
                        {
                            float(np.clip(c - 2.0 * step, lo, hi)),
                            float(np.clip(c - step, lo, hi)),
                            float(np.clip(c, lo, hi)),
                            float(np.clip(c + step, lo, hi)),
                            float(np.clip(c + 2.0 * step, lo, hi)),
                        }
                    )
                    for v in vals:
                        params = {
                            "stage": "refine",
                            "runtime_total_floor": float(cur["runtime_total_floor"]),
                            "shock_alpha": float(cur["shock_alpha"]),
                            "rank_sleeve_blend": float(cur.get("rank_sleeve_blend", 0.0)),
                            "low_vol_sleeve_blend": float(cur.get("low_vol_sleeve_blend", 0.0)),
                            "meta_execution_gate_strength": float(cur["meta_execution_gate_strength"]),
                            "council_gate_strength": float(cur["council_gate_strength"]),
                            "meta_mix_leverage_strength": float(cur["meta_mix_leverage_strength"]),
                            "meta_reliability_strength": float(cur["meta_reliability_strength"]),
                            "global_governor_strength": float(cur["global_governor_strength"]),
                            "heartbeat_scaler_strength": float(cur.get("heartbeat_scaler_strength", 1.0)),
                            "quality_governor_strength": float(cur["quality_governor_strength"]),
                            "regime_moe_strength": float(cur.get("regime_moe_strength", 1.0)),
                            "uncertainty_sizing_strength": float(cur.get("uncertainty_sizing_strength", 1.0)),
                            "vol_target_strength": float(cur.get("vol_target_strength", 1.0)),
                            "hit_gate_strength": float(cur.get("hit_gate_strength", 1.0)),
                            "hit_gate_threshold": float(cur.get("hit_gate_threshold", 0.50)),
                            "signal_deadzone": float(cur.get("signal_deadzone", 0.0)),
                            "use_concentration_governor": int(cur["use_concentration_governor"]),
                            "concentration_top1_cap": float(cur["concentration_top1_cap"]),
                            "concentration_top3_cap": float(cur["concentration_top3_cap"]),
                            "concentration_max_hhi": float(cur["concentration_max_hhi"]),
                        }
                        params[key] = float(v)
                        score, _detail, row = _evaluate_candidate(params, base, rows)
                        if score > best_score:
                            best_score = score
                            best_row = row
                            cur = dict(row)
                            improved = True
                if not improved:
                    break

        sweep_csv = RUNS / "governor_param_sweep.csv"
        _csv_write(rows, sweep_csv)

        profile = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "baseline": base,
            "best": best_row,
            "parameters": _profile_from_row(best_row),
            "search": {
                "runtime_total_floor": floors,
                "shock_alpha": shocks,
                "concentration_presets": conc_presets,
                "rank_sleeve_blend": rank_sleeve_blends,
                "low_vol_sleeve_blend": low_vol_sleeve_blends,
                "meta_execution_gate_strength": meta_execution_gate_strengths,
                "council_gate_strength": council_gate_strengths,
                "meta_mix_leverage_strength": meta_mix_leverage_strengths,
                "meta_reliability_strength": meta_reliability_strengths,
                "global_governor_strength": global_governor_strengths,
                "heartbeat_scaler_strength": heartbeat_scaler_strengths,
                "quality_governor_strength": quality_governor_strengths,
                "regime_moe_strength": regime_moe_strengths,
                "uncertainty_sizing_strength": uncertainty_sizing_strengths,
                "vol_target_strength": vol_target_strengths,
                "hit_gate_strength": hit_gate_strengths,
                "hit_gate_threshold": hit_gate_thresholds,
                "signal_deadzone": signal_deadzones,
                "num_candidates": len(rows),
            },
        }
        out_json = RUNS / "governor_params_profile.json"
        out_json.write_text(json.dumps(profile, indent=2), encoding="utf-8")

        # Apply chosen profile for downstream runs.
        _build_make_daily()
        _run_strict_oos()
        applied = _metrics()
        applied_oos = _strict_oos_metrics()

        print(f"✅ Wrote {sweep_csv}")
        print(f"✅ Wrote {out_json}")
        print(
            "Baseline:",
            f"Sharpe={base['sharpe']:.3f}",
            f"Hit={base['hit_rate']:.3f}",
            f"MaxDD={base['max_drawdown']:.3f}",
            f"OOS_Sharpe={base['oos_sharpe']:.3f}",
            f"OOS_Hit={base['oos_hit_rate']:.3f}",
            f"OOS_MaxDD={base['oos_max_drawdown']:.3f}",
            f"OOS_Source={base.get('oos_source','')}",
        )
        print(
            "Best:",
            f"Sharpe={best_row['sharpe']:.3f}",
            f"Hit={best_row['hit_rate']:.3f}",
            f"MaxDD={best_row['max_drawdown']:.3f}",
            f"OOS_Sharpe={best_row['oos_sharpe']:.3f}",
            f"OOS_Hit={best_row['oos_hit_rate']:.3f}",
            f"OOS_MaxDD={best_row['oos_max_drawdown']:.3f}",
            f"OOS_Source={best_row.get('oos_source','')}",
            f"Score={best_row['score']:.3f}",
        )
        print(
            "Applied profile:",
            f"Sharpe={applied['sharpe']:.3f}",
            f"Hit={applied['hit_rate']:.3f}",
            f"MaxDD={applied['max_drawdown']:.3f}",
            f"OOS_Sharpe={applied_oos['sharpe']:.3f}",
            f"OOS_Hit={applied_oos['hit_rate']:.3f}",
            f"OOS_MaxDD={applied_oos['max_drawdown']:.3f}",
            f"OOS_Source={applied_oos.get('source','')}",
        )
        return 0
    finally:
        # Keep baseline/current profile output coherent even on sweep errors.
        try:
            _build_make_daily()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
