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
    sd = float(np.nanstd(r) + 1e-12)
    sh = float((mu / sd) * np.sqrt(252.0))
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


def _objective(cur: dict, base: dict) -> tuple[float, dict]:
    base_dd = max(1e-9, abs(float(base["max_drawdown"])))
    cur_dd = abs(float(cur["max_drawdown"]))
    dd_ratio = cur_dd / base_dd
    hit_delta = float(cur["hit_rate"]) - float(base["hit_rate"])
    turn_ratio = float(cur["turnover_mean"]) / max(1e-9, float(base["turnover_mean"]))

    # Sharpe-first objective with gentle risk penalties and hard vetoes.
    penalty = 0.012 * max(0.0, dd_ratio - 1.0)
    penalty += 0.006 * max(0.0, turn_ratio - 2.0)
    penalty += 0.010 * max(0.0, -hit_delta)
    score = float(cur["sharpe"]) - penalty

    veto = dd_ratio > 2.0 or cur_dd > 0.22 or hit_delta < -0.03
    if veto:
        score -= 1.0
    detail = {
        "dd_ratio": float(dd_ratio),
        "hit_delta": float(hit_delta),
        "turnover_ratio": float(turn_ratio),
        "penalty": float(penalty),
        "veto": bool(veto),
    }
    return score, detail


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
        "use_concentration_governor",
        "concentration_top1_cap",
        "concentration_top3_cap",
        "concentration_max_hhi",
        "sharpe",
        "hit_rate",
        "max_drawdown",
        "turnover_mean",
        "gross_mean",
        "score",
        "dd_ratio",
        "hit_delta",
        "turnover_ratio",
        "penalty",
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
        "Q_USE_CONCENTRATION_GOV": str(params["use_concentration_governor"]),
        "Q_CONCENTRATION_TOP1_CAP": str(params["concentration_top1_cap"]),
        "Q_CONCENTRATION_TOP3_CAP": str(params["concentration_top3_cap"]),
        "Q_CONCENTRATION_MAX_HHI": str(params["concentration_max_hhi"]),
    }


def _evaluate_candidate(params: dict, base: dict, rows: list[dict]) -> tuple[float, dict, dict]:
    _build_make_daily(_env_from_params(params))
    m = _metrics()
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

    rows: list[dict] = []
    try:
        # Baseline under current profile.
        _build_make_daily()
        base = _metrics()

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
                "num_candidates": len(rows),
            },
        }
        out_json = RUNS / "governor_params_profile.json"
        out_json.write_text(json.dumps(profile, indent=2), encoding="utf-8")

        # Apply chosen profile for downstream runs.
        _build_make_daily()
        _run_strict_oos()
        applied = _metrics()

        print(f"✅ Wrote {sweep_csv}")
        print(f"✅ Wrote {out_json}")
        print(
            "Baseline:",
            f"Sharpe={base['sharpe']:.3f}",
            f"Hit={base['hit_rate']:.3f}",
            f"MaxDD={base['max_drawdown']:.3f}",
        )
        print(
            "Best:",
            f"Sharpe={best_row['sharpe']:.3f}",
            f"Hit={best_row['hit_rate']:.3f}",
            f"MaxDD={best_row['max_drawdown']:.3f}",
            f"Score={best_row['score']:.3f}",
        )
        print(
            "Applied profile:",
            f"Sharpe={applied['sharpe']:.3f}",
            f"Hit={applied['hit_rate']:.3f}",
            f"MaxDD={applied['max_drawdown']:.3f}",
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
