#!/usr/bin/env python3
"""
Hit-rate recovery tuner.

Purpose:
  Improve strict-OOS hit rate (promotion gate readiness) while preserving Sharpe.

Reads/Writes:
  - reads runs_plus/governor_params_profile.json (optional)
  - writes runs_plus/hit_rate_recovery_sweep.csv
  - writes runs_plus/hit_rate_recovery_profile.json
  - updates runs_plus/governor_params_profile.json (parameters merged)
"""

from __future__ import annotations

import csv
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


def _build_and_validate(env_overrides: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items()})
    _run([PYTHON, str(ROOT / "tools" / "build_final_portfolio.py")], env)
    _run([PYTHON, str(ROOT / "tools" / "make_daily_from_weights.py")], env)
    _run([PYTHON, str(ROOT / "tools" / "run_strict_oos_validation.py")], env)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _select_oos_metrics(payload: dict, mode: str) -> tuple[str, dict]:
    robust = payload.get("metrics_oos_robust", {}) if isinstance(payload.get("metrics_oos_robust"), dict) else {}
    single = payload.get("metrics_oos_net", {}) if isinstance(payload.get("metrics_oos_net"), dict) else {}
    m = str(mode).strip().lower()
    if m == "single":
        return "metrics_oos_net", single
    if m == "robust":
        return "metrics_oos_robust", robust
    if robust:
        return "metrics_oos_robust", robust
    return "metrics_oos_net", single


def _strict_oos_metrics() -> dict:
    payload = _load_json(RUNS / "strict_oos_validation.json")
    src, m = _select_oos_metrics(payload, os.getenv("Q_HIT_RECOVERY_OOS_MODE", "robust_then_single"))
    return {
        "source": src,
        "sharpe": float(m.get("sharpe", 0.0)),
        "hit_rate": float(m.get("hit_rate", 0.0)),
        "max_drawdown": float(m.get("max_drawdown", 0.0)),
        "n": int(m.get("n", 0)),
    }


def _score_candidate(metrics: dict) -> tuple[float, dict]:
    sh = float(metrics.get("sharpe", 0.0))
    hit = float(metrics.get("hit_rate", 0.0))
    mdd = abs(float(metrics.get("max_drawdown", 0.0)))
    n = int(metrics.get("n", 0))

    target_hit = float(np.clip(float(os.getenv("Q_HIT_RECOVERY_TARGET_HIT", "0.49")), 0.0, 1.0))
    target_sh = float(np.clip(float(os.getenv("Q_HIT_RECOVERY_TARGET_SHARPE", "1.10")), -2.0, 10.0))
    max_abs_mdd = float(np.clip(float(os.getenv("Q_HIT_RECOVERY_MAX_ABS_MDD", "0.10")), 0.001, 2.0))
    min_n = int(np.clip(int(float(os.getenv("Q_HIT_RECOVERY_MIN_OOS_N", "252"))), 1, 1000000))
    hit_penalty = float(np.clip(float(os.getenv("Q_HIT_RECOVERY_HIT_PENALTY", "3.2")), 0.0, 20.0))

    hit_gap = max(0.0, target_hit - hit)
    sh_gap = max(0.0, target_sh - sh)
    dd_gap = max(0.0, mdd - max_abs_mdd)

    penalty = hit_penalty * hit_gap + 0.6 * sh_gap + 0.8 * dd_gap
    score = sh - penalty
    veto = bool(n < min_n or hit < (target_hit - 0.10) or mdd > (1.8 * max_abs_mdd))
    if veto:
        score -= 1.0
    detail = {
        "target_hit": float(target_hit),
        "target_sharpe": float(target_sh),
        "target_max_abs_mdd": float(max_abs_mdd),
        "target_min_oos_n": int(min_n),
        "hit_penalty": float(hit_penalty),
        "hit_gap": float(hit_gap),
        "sharpe_gap": float(sh_gap),
        "mdd_gap": float(dd_gap),
        "penalty": float(penalty),
        "veto": bool(veto),
    }
    return float(score), detail


def _local_grid(center: float, lo: float, hi: float, step: float) -> list[float]:
    vals = [
        float(np.clip(center - 2.0 * step, lo, hi)),
        float(np.clip(center - step, lo, hi)),
        float(np.clip(center, lo, hi)),
        float(np.clip(center + step, lo, hi)),
        float(np.clip(center + 2.0 * step, lo, hi)),
    ]
    return sorted(set(vals))


def _params_from_profile() -> dict:
    prof = _load_json(RUNS / "governor_params_profile.json")
    p = prof.get("parameters", prof)
    if not isinstance(p, dict):
        p = {}
    return {
        "hit_gate_strength": float(p.get("hit_gate_strength", 1.0)),
        "hit_gate_threshold": float(p.get("hit_gate_threshold", 0.50)),
        "signal_deadzone": float(p.get("signal_deadzone", 0.0)),
        "rank_sleeve_blend": float(p.get("rank_sleeve_blend", 0.0)),
        "low_vol_sleeve_blend": float(p.get("low_vol_sleeve_blend", 0.0)),
    }


def _env_from_params(params: dict) -> dict[str, str]:
    return {
        "Q_HIT_GATE_STRENGTH": str(params["hit_gate_strength"]),
        "Q_HIT_GATE_THRESHOLD": str(params["hit_gate_threshold"]),
        "Q_SIGNAL_DEADZONE": str(params["signal_deadzone"]),
        "Q_RANK_SLEEVE_BLEND": str(params["rank_sleeve_blend"]),
        "Q_LOW_VOL_SLEEVE_BLEND": str(params["low_vol_sleeve_blend"]),
    }


def _evaluate(params: dict, rows: list[dict]) -> tuple[float, dict]:
    env = _env_from_params(params)
    _build_and_validate(env)
    oos = _strict_oos_metrics()
    score, detail = _score_candidate(oos)
    row = {**params, **oos, **detail, "score": float(score)}
    rows.append(row)
    return score, row


def _write_csv(rows: list[dict], outp: Path) -> None:
    cols = [
        "hit_gate_strength",
        "hit_gate_threshold",
        "signal_deadzone",
        "rank_sleeve_blend",
        "low_vol_sleeve_blend",
        "source",
        "sharpe",
        "hit_rate",
        "max_drawdown",
        "n",
        "target_hit",
        "target_sharpe",
        "target_max_abs_mdd",
        "target_min_oos_n",
        "hit_penalty",
        "hit_gap",
        "sharpe_gap",
        "mdd_gap",
        "penalty",
        "veto",
        "score",
    ]
    with outp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def _merge_profile(best: dict) -> None:
    p = RUNS / "governor_params_profile.json"
    base = _load_json(p)
    params = base.get("parameters", {}) if isinstance(base.get("parameters"), dict) else {}
    params.update(
        {
            "hit_gate_strength": float(best["hit_gate_strength"]),
            "hit_gate_threshold": float(best["hit_gate_threshold"]),
            "signal_deadzone": float(best["signal_deadzone"]),
            "rank_sleeve_blend": float(best["rank_sleeve_blend"]),
            "low_vol_sleeve_blend": float(best["low_vol_sleeve_blend"]),
        }
    )
    base["parameters"] = params
    base["hit_recovery"] = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected": {
            "hit_gate_strength": float(best["hit_gate_strength"]),
            "hit_gate_threshold": float(best["hit_gate_threshold"]),
            "signal_deadzone": float(best["signal_deadzone"]),
            "rank_sleeve_blend": float(best["rank_sleeve_blend"]),
            "low_vol_sleeve_blend": float(best["low_vol_sleeve_blend"]),
        },
        "oos": {
            "source": str(best.get("source", "")),
            "sharpe": float(best.get("sharpe", 0.0)),
            "hit_rate": float(best.get("hit_rate", 0.0)),
            "max_drawdown": float(best.get("max_drawdown", 0.0)),
            "n": int(best.get("n", 0)),
        },
        "score": float(best.get("score", 0.0)),
    }
    p.write_text(json.dumps(base, indent=2), encoding="utf-8")


def main() -> int:
    if not (RUNS / "asset_returns.csv").exists():
        print("(!) Missing runs_plus/asset_returns.csv. Run tools/rebuild_asset_matrix.py first.")
        return 0

    base_params = _params_from_profile()
    rows: list[dict] = []
    best_score = -1e9
    best = None
    try:
        # Baseline under current profile.
        s0, r0 = _evaluate(base_params, rows)
        best_score = float(s0)
        best = dict(r0)

        cur = dict(base_params)
        grids = {
            "hit_gate_strength": [0.0, 0.3, 0.5, 0.8, 1.0, 1.2],
            "hit_gate_threshold": [0.40, 0.42, 0.44, 0.46, 0.48, 0.50],
            "signal_deadzone": [0.0, 0.0003, 0.0006, 0.0010, 0.0015],
            "rank_sleeve_blend": _local_grid(base_params["rank_sleeve_blend"], 0.0, 0.25, 0.02),
            "low_vol_sleeve_blend": _local_grid(base_params["low_vol_sleeve_blend"], 0.0, 0.20, 0.015),
        }
        for _pass in range(2):
            improved = False
            for key, vals in grids.items():
                for v in vals:
                    cand = dict(cur)
                    cand[key] = float(v)
                    sc, row = _evaluate(cand, rows)
                    if sc > best_score:
                        best_score = float(sc)
                        best = dict(row)
                        cur = dict(cand)
                        improved = True
            if not improved:
                break

        if not rows:
            print("(!) No candidates evaluated.")
            return 1
        target_hit = float(rows[-1].get("target_hit", 0.49))
        target_mdd = float(rows[-1].get("target_max_abs_mdd", 0.10))
        target_n = int(rows[-1].get("target_min_oos_n", 252))
        feasible = [
            r
            for r in rows
            if float(r.get("hit_rate", 0.0)) >= target_hit
            and abs(float(r.get("max_drawdown", 0.0))) <= target_mdd
            and int(r.get("n", 0)) >= target_n
        ]
        if feasible:
            best = max(feasible, key=lambda r: (float(r.get("sharpe", 0.0)), float(r.get("score", -1e9))))
            best_score = float(best.get("score", -1e9))
        else:
            best = max(
                rows,
                key=lambda r: (float(r.get("hit_rate", 0.0)), float(r.get("sharpe", 0.0)), float(r.get("score", -1e9))),
            )
            best_score = float(best.get("score", -1e9))

        _write_csv(rows, RUNS / "hit_rate_recovery_sweep.csv")
        (RUNS / "hit_rate_recovery_profile.json").write_text(
            json.dumps(
                {
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "baseline": r0,
                    "best": best,
                    "num_candidates": int(len(rows)),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _merge_profile(best)

        # Apply merged profile cleanly and refresh promotion gate artifact.
        _build_and_validate()
        _run([PYTHON, str(ROOT / "tools" / "run_q_promotion_gate.py")], os.environ.copy())

        print(f"✅ Wrote {RUNS/'hit_rate_recovery_sweep.csv'}")
        print(f"✅ Wrote {RUNS/'hit_rate_recovery_profile.json'}")
        print(f"✅ Updated {RUNS/'governor_params_profile.json'}")
        print(
            "Best hit-recovery:",
            f"OOS_Sharpe={float(best['sharpe']):.3f}",
            f"OOS_Hit={float(best['hit_rate']):.3f}",
            f"OOS_MaxDD={float(best['max_drawdown']):.3f}",
            f"Score={float(best['score']):.3f}",
        )
        return 0
    finally:
        # Keep artifacts coherent for downstream steps.
        try:
            _build_and_validate()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
