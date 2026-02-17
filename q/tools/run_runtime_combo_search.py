#!/usr/bin/env python3
"""
Runtime combo search for stable, gate-compliant Sharpe improvement.

Evaluates a small grid of runtime floors and governor disable combinations.
Only configurations that pass promotion, cost stress, and health hard alerts
are considered valid.

Writes:
  - runs_plus/runtime_combo_search.json
  - runs_plus/runtime_profile_selected.json
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)
PY = str(Path(sys.executable))


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _parse_csv_floats(raw: str, lo: float, hi: float) -> list[float]:
    vals = []
    for tok in str(raw).split(","):
        t = str(tok).strip()
        if not t:
            continue
        try:
            vals.append(float(np.clip(float(t), lo, hi)))
        except Exception:
            continue
    out = sorted(set(round(v, 6) for v in vals))
    return [float(v) for v in out]


def _parse_csv_tokens(raw: str) -> list[str]:
    out = []
    seen = set()
    for tok in str(raw).split(","):
        t = str(tok).strip().lower()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _run(cmd: list[str], env: dict[str, str]) -> int:
    p = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return int(p.returncode)


def _eval_combo(env_overrides: dict[str, str]) -> dict:
    env = os.environ.copy()
    env.update(env_overrides)

    cmds = [
        [PY, str(ROOT / "tools" / "build_final_portfolio.py")],
        [PY, str(ROOT / "tools" / "make_daily_from_weights.py")],
        [PY, str(ROOT / "tools" / "run_meta_execution_gate.py")],
        [PY, str(ROOT / "tools" / "build_final_portfolio.py")],
        [PY, str(ROOT / "tools" / "make_daily_from_weights.py")],
        [PY, str(ROOT / "tools" / "run_strict_oos_validation.py")],
        [PY, str(ROOT / "tools" / "run_cost_stress_validation.py")],
        [PY, str(ROOT / "tools" / "run_q_promotion_gate.py")],
        [PY, str(ROOT / "tools" / "run_system_health.py")],
        [PY, str(ROOT / "tools" / "run_health_alerts.py")],
    ]
    rc = []
    for c in cmds:
        code = _run(c, env)
        rc.append({"step": Path(c[-1]).name, "code": int(code)})
        if code != 0:
            break

    strict = _load_json(RUNS / "strict_oos_validation.json")
    robust = strict.get("metrics_oos_robust", {}) if isinstance(strict.get("metrics_oos_robust"), dict) else {}
    promo = _load_json(RUNS / "q_promotion_gate.json")
    stress = _load_json(RUNS / "cost_stress_validation.json")
    health = _load_json(RUNS / "health_alerts.json")

    return {
        "robust_sharpe": float(robust.get("sharpe", 0.0)),
        "robust_hit_rate": float(robust.get("hit_rate", 0.0)),
        "robust_max_drawdown": float(robust.get("max_drawdown", 0.0)),
        "promotion_ok": bool(promo.get("ok", False)),
        "cost_stress_ok": bool(stress.get("ok", False)),
        "health_ok": bool(health.get("ok", False)),
        "health_alerts_hard": int(health.get("alerts_hard", 999)),
        "rc": rc,
    }


def _base_runtime_env() -> dict[str, str]:
    prof = _load_json(RUNS / "governor_params_profile.json")
    params = prof.get("parameters", prof)
    if not isinstance(params, dict):
        params = {}
    return {
        "Q_DISABLE_REPORT_CARDS": "1",
        "Q_PROMOTION_REQUIRE_COST_STRESS": "1",
        "TURNOVER_MAX_STEP": str(params.get("turnover_max_step", 0.30)),
        "TURNOVER_BUDGET_WINDOW": str(params.get("turnover_budget_window", 13)),
        "TURNOVER_BUDGET_LIMIT": str(params.get("turnover_budget_limit", 0.70)),
        "Q_META_EXEC_MIN_PROB": str(params.get("meta_exec_min_prob", 0.57)),
        "Q_META_EXEC_FLOOR": str(params.get("meta_exec_floor", 0.20)),
        "Q_META_EXEC_SLOPE": str(params.get("meta_exec_slope", 12.0)),
        "Q_VOL_TARGET_STRENGTH": str(params.get("vol_target_strength", 0.45)),
        "Q_VOL_TARGET_ANNUAL": str(params.get("vol_target_annual", 0.10)),
        "Q_VOL_TARGET_LOOKBACK": str(params.get("vol_target_lookback", 63)),
        "Q_VOL_TARGET_MIN_SCALAR": str(params.get("vol_target_min_scalar", 0.40)),
        "Q_VOL_TARGET_MAX_SCALAR": str(params.get("vol_target_max_scalar", 1.40)),
        "Q_VOL_TARGET_SMOOTH_ALPHA": str(params.get("vol_target_smooth_alpha", 0.10)),
        "Q_CASH_YIELD_ANNUAL": str(params.get("cash_yield_annual", 0.01)),
        "Q_CASH_EXPOSURE_TARGET": str(params.get("cash_exposure_target", 1.0)),
    }


def main() -> int:
    floors = _parse_csv_floats(os.getenv("Q_RUNTIME_SEARCH_FLOORS", "0.18,0.20,0.22"), 0.0, 1.0)
    flags = _parse_csv_tokens(
        os.getenv(
            "Q_RUNTIME_SEARCH_FLAGS",
            "uncertainty_sizing,global_governor,quality_governor,heartbeat_scaler",
        )
    )
    if not floors:
        floors = [0.18]

    rows = []
    base_env = _base_runtime_env()
    combos = list(itertools.product([0, 1], repeat=len(flags)))
    total = len(floors) * len(combos)
    i = 0
    for floor in floors:
        for bits in combos:
            i += 1
            disabled = [f for f, b in zip(flags, bits) if b]
            env = dict(base_env)
            env["Q_RUNTIME_TOTAL_FLOOR"] = str(floor)
            env["Q_DISABLE_GOVERNORS"] = ",".join(disabled)
            out = _eval_combo(env)
            row = {
                "runtime_total_floor": float(floor),
                "disable_governors": disabled,
                **out,
            }
            rows.append(row)
            if i % 8 == 0 or i == total:
                print(f"… evaluated {i}/{total} combos")

    def _valid(r: dict) -> bool:
        rc_ok = all(int(x.get("code", 1)) == 0 for x in (r.get("rc") or []))
        return (
            rc_ok
            and bool(r.get("promotion_ok", False))
            and bool(r.get("cost_stress_ok", False))
            and bool(r.get("health_ok", False))
            and int(r.get("health_alerts_hard", 999)) == 0
        )

    valid = [r for r in rows if _valid(r)]
    valid_sorted = sorted(
        valid,
        key=lambda r: (
            float(r.get("robust_sharpe", 0.0)),
            float(r.get("robust_hit_rate", 0.0)),
            -abs(float(r.get("robust_max_drawdown", 0.0))),
        ),
        reverse=True,
    )

    selected = valid_sorted[0] if valid_sorted else None
    out = {
        "floors": floors,
        "flags": flags,
        "rows_total": len(rows),
        "rows_valid": len(valid_sorted),
        "top_valid": valid_sorted[:20],
        "selected": selected,
    }
    (RUNS / "runtime_combo_search.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    if selected:
        sel = {
            "runtime_total_floor": float(selected.get("runtime_total_floor", 0.18)),
            "disable_governors": list(selected.get("disable_governors", [])),
            "robust_sharpe": float(selected.get("robust_sharpe", 0.0)),
            "robust_hit_rate": float(selected.get("robust_hit_rate", 0.0)),
            "robust_max_drawdown": float(selected.get("robust_max_drawdown", 0.0)),
            "promotion_ok": bool(selected.get("promotion_ok", False)),
            "cost_stress_ok": bool(selected.get("cost_stress_ok", False)),
            "health_ok": bool(selected.get("health_ok", False)),
        }
        (RUNS / "runtime_profile_selected.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
        print(f"✅ Selected runtime profile: floor={sel['runtime_total_floor']} disable={sel['disable_governors']}")
    else:
        print("(!) No valid runtime profile found in search grid.")

    print(f"✅ Wrote {RUNS/'runtime_combo_search.json'}")
    if selected:
        print(f"✅ Wrote {RUNS/'runtime_profile_selected.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
