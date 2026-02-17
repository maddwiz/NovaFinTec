#!/usr/bin/env python3
"""
Runtime combo search for stable, gate-compliant Sharpe improvement.

Evaluates a small grid of runtime floors and governor disable combinations.
Only configurations that pass promotion, cost stress, and health hard alerts
are considered valid.

Writes:
  - runs_plus/runtime_combo_search.json
  - runs_plus/runtime_profile_selected.json
  - runs_plus/runtime_profile_stable.json
  - runs_plus/runtime_profile_active.json
  - runs_plus/runtime_profile_challenger.json (when canary armed)
  - runs_plus/runtime_profile_canary_state.json
  - runs_plus/runtime_profile_promotion_status.json
"""

from __future__ import annotations

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


def _parse_csv_bools(raw: str) -> list[int]:
    out = []
    seen = set()
    for tok in str(raw).split(","):
        t = str(tok).strip().lower()
        if not t:
            continue
        if t in {"1", "true", "yes", "on"}:
            v = 1
        elif t in {"0", "false", "no", "off"}:
            v = 0
        else:
            continue
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _infer_asset_class(sym: str) -> str:
    s = str(sym or "").upper().replace("-", "").replace("_", "").replace("/", "")
    if not s:
        return "EQ"
    if s in {"VIX", "VIX9D", "VIX3M", "VIXCLS", "UVXY", "VXX"}:
        return "VOL"
    if s in {"LQD", "HYG", "JNK", "BND", "AGG", "MBB", "HYGTR", "LQDTR"} or s.endswith("_TR"):
        return "CREDIT"
    if s in {"TLT", "IEF", "SHY", "VGSH", "DGS2", "DGS3MO", "DGS5", "DGS10", "DGS30", "TY", "ZN", "ZB"}:
        return "RATES"
    if s in {"GLD", "SLV", "USO", "UNG", "CORN", "WEAT", "CPER", "DBC", "DBA", "XLE", "XOP"}:
        return "COMMOD"
    if len(s) == 6 and s[:3] in {"USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD"} and s[3:] in {
        "USD",
        "EUR",
        "JPY",
        "GBP",
        "CHF",
        "CAD",
        "AUD",
        "NZD",
    }:
        return "FX"
    if s.startswith("BTC") or s.startswith("ETH") or s in {"IBIT", "FBTC", "BITO"}:
        return "CRYPTO"
    return "EQ"


def _discover_asset_names() -> list[str]:
    p = RUNS / "asset_names.csv"
    if p.exists():
        try:
            import csv

            with p.open("r", encoding="utf-8", newline="") as fh:
                rows = list(csv.reader(fh))
            if rows and rows[0]:
                names = [str(r[0]).strip().upper() for r in rows[1:] if r]
                names = [x for x in names if x]
                if names:
                    return names
        except Exception:
            pass
    d = ROOT / "data"
    if d.exists():
        try:
            return sorted({x.stem.upper() for x in d.glob("*.csv") if x.is_file()})
        except Exception:
            return []
    return []


def _default_class_enable_grid() -> list[int]:
    raw = os.getenv("Q_RUNTIME_SEARCH_CLASS_ENABLES", "").strip()
    if raw:
        vals = _parse_csv_bools(raw)
        return vals if vals else [0]
    names = _discover_asset_names()
    classes = { _infer_asset_class(x) for x in names if str(x).strip() }
    min_classes = int(np.clip(int(float(os.getenv("Q_RUNTIME_SEARCH_MIN_CLASSES_FOR_DIVERSIFICATION", "3"))), 1, 20))
    if len(classes) >= min_classes:
        return [0, 1]
    return [0]


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

    use_asset_class = str(env.get("Q_ENABLE_ASSET_CLASS_DIVERSIFICATION", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    cmds = []
    if use_asset_class:
        cmds.append([PY, str(ROOT / "tools" / "run_asset_class_diversification.py")])
    cmds.extend(
        [
        [PY, str(ROOT / "tools" / "run_capacity_impact_guard.py")],
        [PY, str(ROOT / "tools" / "run_macro_proxy_guard.py")],
        [PY, str(ROOT / "tools" / "run_uncertainty_sizing.py")],
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
    )
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
    cinfo = _load_json(RUNS / "daily_costs_info.json")

    return {
        "robust_sharpe": float(robust.get("sharpe", 0.0)),
        "robust_hit_rate": float(robust.get("hit_rate", 0.0)),
        "robust_max_drawdown": float(robust.get("max_drawdown", 0.0)),
        "ann_cost_estimate": float(cinfo.get("ann_cost_estimate", 0.0)),
        "mean_turnover": float(cinfo.get("mean_turnover", 0.0)),
        "mean_effective_cost_bps": float(cinfo.get("mean_effective_cost_bps", 0.0)),
        "promotion_ok": bool(promo.get("ok", False)),
        "cost_stress_ok": bool(stress.get("ok", False)),
        "health_ok": bool(health.get("ok", False)),
        "health_alerts_hard": int(health.get("alerts_hard", 999)),
        "rc": rc,
    }


def _score_row(row: dict) -> float:
    sh = float(row.get("robust_sharpe", 0.0))
    hit = float(row.get("robust_hit_rate", 0.0))
    mdd = abs(float(row.get("robust_max_drawdown", 0.0)))
    target_hit = float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_TARGET_HIT", "0.49")), 0.0, 1.0))
    hit_w = float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_HIT_WEIGHT", "0.75")), 0.0, 10.0))
    mdd_ref = float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_MDD_REF", "0.04")), 0.001, 1.0))
    mdd_w = float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_MDD_PENALTY", "4.0")), 0.0, 25.0))
    cost_ref = float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_COST_REF_ANNUAL", "0.02")), 0.0, 1.0))
    cost_w = float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_COST_PENALTY", "3.0")), 0.0, 100.0))
    turn_ref = float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_TURNOVER_REF_DAILY", "0.06")), 0.0, 5.0))
    turn_w = float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_TURNOVER_PENALTY", "1.5")), 0.0, 100.0))
    ann_cost = max(0.0, float(row.get("ann_cost_estimate", 0.0)))
    mean_turn = max(0.0, float(row.get("mean_turnover", 0.0)))
    over_mdd = max(0.0, mdd - mdd_ref)
    over_cost = max(0.0, ann_cost - cost_ref)
    over_turn = max(0.0, mean_turn - turn_ref)
    return float(
        sh
        + hit_w * (hit - target_hit)
        - mdd_w * over_mdd
        - cost_w * over_cost
        - turn_w * over_turn
    )


def _write_progress(*, evaluated: int, total: int, best: dict | None) -> None:
    payload = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluated": int(evaluated),
        "total": int(total),
        "best_so_far": best or {},
    }
    (RUNS / "runtime_combo_search_progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _profile_signature(profile: dict) -> str:
    floor = float(profile.get("runtime_total_floor", 0.0))
    disabled = sorted({str(x).strip().lower() for x in (profile.get("disable_governors") or []) if str(x).strip()})
    use_asset_class = int(bool(profile.get("enable_asset_class_diversification", False)))
    macro_strength = float(profile.get("macro_proxy_strength", 0.0))
    capacity_strength = float(profile.get("capacity_impact_strength", 0.0))
    macro_blend = float(profile.get("uncertainty_macro_shock_blend", 0.0))
    return (
        f"floor={floor:.6f}|disable={','.join(disabled)}"
        f"|asset_class={use_asset_class}"
        f"|macro={macro_strength:.6f}"
        f"|capacity={capacity_strength:.6f}"
        f"|macro_blend={macro_blend:.6f}"
    )


def _profile_payload(row: dict) -> dict:
    return {
        "runtime_total_floor": float(row.get("runtime_total_floor", 0.18)),
        "disable_governors": list(row.get("disable_governors", [])),
        "enable_asset_class_diversification": bool(row.get("enable_asset_class_diversification", False)),
        "macro_proxy_strength": float(row.get("macro_proxy_strength", 0.0)),
        "capacity_impact_strength": float(row.get("capacity_impact_strength", 0.0)),
        "uncertainty_macro_shock_blend": float(row.get("uncertainty_macro_shock_blend", 0.0)),
        "robust_sharpe": float(row.get("robust_sharpe", 0.0)),
        "robust_hit_rate": float(row.get("robust_hit_rate", 0.0)),
        "robust_max_drawdown": float(row.get("robust_max_drawdown", 0.0)),
        "ann_cost_estimate": float(row.get("ann_cost_estimate", 0.0)),
        "mean_turnover": float(row.get("mean_turnover", 0.0)),
        "mean_effective_cost_bps": float(row.get("mean_effective_cost_bps", 0.0)),
        "score": float(row.get("score", 0.0)),
        "promotion_ok": bool(row.get("promotion_ok", False)),
        "cost_stress_ok": bool(row.get("cost_stress_ok", False)),
        "health_ok": bool(row.get("health_ok", False)),
    }


def _write_profile(name: str, payload: dict) -> None:
    (RUNS / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _remove_profile(name: str) -> None:
    p = RUNS / name
    if p.exists():
        p.unlink()


def _canary_qualifies(stable: dict, candidate: dict) -> tuple[bool, list[str]]:
    reasons = []
    min_sh_delta = float(np.clip(float(os.getenv("Q_RUNTIME_CANARY_MIN_SHARPE_DELTA", "0.02")), 0.0, 2.0))
    min_score_delta = float(np.clip(float(os.getenv("Q_RUNTIME_CANARY_MIN_SCORE_DELTA", "0.015")), 0.0, 2.0))
    max_hit_drop = float(np.clip(float(os.getenv("Q_RUNTIME_CANARY_MAX_HIT_DROP", "0.0025")), 0.0, 0.10))
    max_mdd_worsen = float(np.clip(float(os.getenv("Q_RUNTIME_CANARY_MAX_ABS_MDD_WORSEN", "0.005")), 0.0, 0.50))
    max_ann_cost_worsen = float(np.clip(float(os.getenv("Q_RUNTIME_CANARY_MAX_ANN_COST_WORSEN", "0.004")), 0.0, 1.0))
    max_turnover_worsen = float(np.clip(float(os.getenv("Q_RUNTIME_CANARY_MAX_TURNOVER_WORSEN", "0.010")), 0.0, 5.0))

    st_sh = float(stable.get("robust_sharpe", 0.0))
    st_hit = float(stable.get("robust_hit_rate", 0.0))
    st_mdd = abs(float(stable.get("robust_max_drawdown", 0.0)))
    st_score = float(stable.get("score", st_sh))
    def _opt_nonneg(src: dict, key: str) -> float | None:
        if key not in src:
            return None
        try:
            v = float(src.get(key))
        except Exception:
            return None
        if not np.isfinite(v):
            return None
        return max(0.0, v)

    st_ann_cost = _opt_nonneg(stable, "ann_cost_estimate")
    st_turn = _opt_nonneg(stable, "mean_turnover")
    ca_sh = float(candidate.get("robust_sharpe", 0.0))
    ca_hit = float(candidate.get("robust_hit_rate", 0.0))
    ca_mdd = abs(float(candidate.get("robust_max_drawdown", 0.0)))
    ca_score = float(candidate.get("score", ca_sh))
    ca_ann_cost = _opt_nonneg(candidate, "ann_cost_estimate")
    ca_turn = _opt_nonneg(candidate, "mean_turnover")
    sh_delta = float(ca_sh - st_sh)
    score_delta = float(ca_score - st_score)

    if (sh_delta < min_sh_delta) and (score_delta < min_score_delta):
        reasons.append(
            f"delta_below_thresholds (sh={sh_delta:.3f}<{min_sh_delta:.3f}, score={score_delta:.3f}<{min_score_delta:.3f})"
        )
    if ca_hit < (st_hit - max_hit_drop):
        reasons.append(f"hit_drop>{max_hit_drop:.4f} ({st_hit - ca_hit:.4f})")
    if ca_mdd > (st_mdd + max_mdd_worsen):
        reasons.append(f"mdd_worsen>{max_mdd_worsen:.3f} ({ca_mdd - st_mdd:.3f})")
    if (st_ann_cost is not None) and (ca_ann_cost is not None) and (ca_ann_cost > (st_ann_cost + max_ann_cost_worsen)):
        reasons.append(f"ann_cost_worsen>{max_ann_cost_worsen:.4f} ({ca_ann_cost - st_ann_cost:.4f})")
    if (st_turn is not None) and (ca_turn is not None) and (ca_turn > (st_turn + max_turnover_worsen)):
        reasons.append(f"turnover_worsen>{max_turnover_worsen:.4f} ({ca_turn - st_turn:.4f})")
    if not bool(candidate.get("promotion_ok", False)):
        reasons.append("promotion_not_ok")
    if not bool(candidate.get("cost_stress_ok", False)):
        reasons.append("cost_stress_not_ok")
    if not bool(candidate.get("health_ok", False)):
        reasons.append("health_not_ok")
    return len(reasons) == 0, reasons


def _base_runtime_env() -> dict[str, str]:
    prof = _load_json(RUNS / "governor_params_profile.json")
    params = prof.get("parameters", prof)
    if not isinstance(params, dict):
        params = {}
    base = {
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
    fc = _load_json(RUNS / "friction_calibration.json")
    rec = fc.get("recommendation", {}) if isinstance(fc.get("recommendation"), dict) else {}
    if bool(fc.get("ok", rec.get("ok", False))):
        try:
            base["Q_COST_BASE_BPS"] = str(float(rec.get("recommended_cost_base_bps")))
        except Exception:
            pass
        try:
            base["Q_COST_VOL_SCALED_BPS"] = str(float(rec.get("recommended_cost_vol_scaled_bps", 0.0)))
        except Exception:
            pass
    return base


def _refresh_friction_calibration(base_env: dict[str, str]) -> None:
    env = os.environ.copy()
    env.update(base_env)
    _run([PY, str(ROOT / "tools" / "run_calibrate_friction_from_aion.py")], env)


def main() -> int:
    floors = _parse_csv_floats(os.getenv("Q_RUNTIME_SEARCH_FLOORS", "0.18,0.20,0.22"), 0.0, 1.0)
    flags = _parse_csv_tokens(
        os.getenv(
            "Q_RUNTIME_SEARCH_FLAGS",
            "uncertainty_sizing,global_governor,quality_governor,heartbeat_scaler",
        )
    )
    class_enables = _default_class_enable_grid()
    macro_strengths = _parse_csv_floats(os.getenv("Q_RUNTIME_SEARCH_MACRO_STRENGTHS", "0.0"), 0.0, 2.0) or [0.0]
    capacity_strengths = _parse_csv_floats(os.getenv("Q_RUNTIME_SEARCH_CAPACITY_STRENGTHS", "0.0"), 0.0, 2.0) or [0.0]
    macro_blends = _parse_csv_floats(os.getenv("Q_RUNTIME_SEARCH_MACRO_BLENDS", "0.0"), 0.0, 1.0) or [0.0]
    max_combos = int(np.clip(int(float(os.getenv("Q_RUNTIME_SEARCH_MAX_COMBOS", "128"))), 8, 5000))
    if not floors:
        floors = [0.18]

    rows = []
    base_env = _base_runtime_env()
    _refresh_friction_calibration(base_env)
    base_env = _base_runtime_env()
    bit_combos = list(itertools.product([0, 1], repeat=len(flags)))
    grid = list(
        itertools.product(
            floors,
            bit_combos,
            class_enables,
            macro_strengths,
            capacity_strengths,
            macro_blends,
        )
    )
    full_total = len(grid)
    if full_total > max_combos:
        idx = np.linspace(0, full_total - 1, num=max_combos, dtype=int)
        keep = []
        seen = set()
        for i_raw in idx.tolist():
            i_int = int(i_raw)
            if i_int in seen:
                continue
            seen.add(i_int)
            keep.append(grid[i_int])
        grid = keep
    total = len(grid)
    i = 0
    best_so_far = None
    best_so_far_score = -1e9
    for floor, bits, use_asset_class, macro_strength, capacity_strength, macro_blend in grid:
        i += 1
        disabled = [f for f, b in zip(flags, bits) if b]
        env = dict(base_env)
        env["Q_RUNTIME_TOTAL_FLOOR"] = str(floor)
        env["Q_DISABLE_GOVERNORS"] = ",".join(disabled)
        env["Q_ENABLE_ASSET_CLASS_DIVERSIFICATION"] = "1" if int(use_asset_class) == 1 else "0"
        env["Q_MACRO_PROXY_STRENGTH"] = str(float(macro_strength))
        env["Q_CAPACITY_IMPACT_STRENGTH"] = str(float(capacity_strength))
        env["Q_UNCERTAINTY_MACRO_SHOCK_BLEND"] = str(float(macro_blend))
        out = _eval_combo(env)
        row = {
            "runtime_total_floor": float(floor),
            "disable_governors": disabled,
            "enable_asset_class_diversification": bool(int(use_asset_class) == 1),
            "macro_proxy_strength": float(macro_strength),
            "capacity_impact_strength": float(capacity_strength),
            "uncertainty_macro_shock_blend": float(macro_blend),
            **out,
        }
        row["score"] = _score_row(row)
        rows.append(row)
        if (
            bool(row.get("promotion_ok", False))
            and bool(row.get("cost_stress_ok", False))
            and bool(row.get("health_ok", False))
            and int(row.get("health_alerts_hard", 999)) == 0
            and all(int(x.get("code", 1)) == 0 for x in (row.get("rc") or []))
            and float(row["score"]) > float(best_so_far_score)
        ):
            best_so_far_score = float(row["score"])
            best_so_far = {
                "runtime_total_floor": float(row["runtime_total_floor"]),
                "disable_governors": list(row["disable_governors"]),
                "enable_asset_class_diversification": bool(row["enable_asset_class_diversification"]),
                "macro_proxy_strength": float(row["macro_proxy_strength"]),
                "capacity_impact_strength": float(row["capacity_impact_strength"]),
                "uncertainty_macro_shock_blend": float(row["uncertainty_macro_shock_blend"]),
                "robust_sharpe": float(row["robust_sharpe"]),
                "robust_hit_rate": float(row["robust_hit_rate"]),
                "robust_max_drawdown": float(row["robust_max_drawdown"]),
                "score": float(row["score"]),
            }
        if i % 8 == 0 or i == total:
            print(f"… evaluated {i}/{total} combos")
            _write_progress(evaluated=i, total=total, best=best_so_far)

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
            float(r.get("score", -1e9)),
            float(r.get("robust_sharpe", 0.0)),
            float(r.get("robust_hit_rate", 0.0)),
        ),
        reverse=True,
    )

    selected = valid_sorted[0] if valid_sorted else None
    out = {
        "floors": floors,
        "flags": flags,
        "class_enables": class_enables,
        "macro_strengths": macro_strengths,
        "capacity_strengths": capacity_strengths,
        "macro_blends": macro_blends,
        "full_grid_total": int(full_total),
        "max_combos": int(max_combos),
        "rows_total": len(rows),
        "rows_valid": len(valid_sorted),
        "score_formula": {
            "score": "sharpe + hit_weight*(hit-target_hit) - mdd_penalty*max(0, abs(max_drawdown)-mdd_ref) - cost_penalty*max(0, ann_cost_estimate-cost_ref_annual) - turnover_penalty*max(0, mean_turnover-turnover_ref_daily)",
            "target_hit": float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_TARGET_HIT", "0.49")), 0.0, 1.0)),
            "hit_weight": float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_HIT_WEIGHT", "0.75")), 0.0, 10.0)),
            "mdd_ref": float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_MDD_REF", "0.04")), 0.001, 1.0)),
            "mdd_penalty": float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_MDD_PENALTY", "4.0")), 0.0, 25.0)),
            "cost_ref_annual": float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_COST_REF_ANNUAL", "0.02")), 0.0, 1.0)),
            "cost_penalty": float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_COST_PENALTY", "3.0")), 0.0, 100.0)),
            "turnover_ref_daily": float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_TURNOVER_REF_DAILY", "0.06")), 0.0, 5.0)),
            "turnover_penalty": float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_TURNOVER_PENALTY", "1.5")), 0.0, 100.0)),
        },
        "top_valid": valid_sorted[:20],
        "selected": selected,
    }
    (RUNS / "runtime_combo_search.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    state_path = RUNS / "runtime_profile_canary_state.json"
    status_path = RUNS / "runtime_profile_promotion_status.json"
    required_passes = int(np.clip(int(float(os.getenv("Q_RUNTIME_CANARY_REQUIRED_PASSES", "3"))), 1, 20))
    rollback_fails = int(np.clip(int(float(os.getenv("Q_RUNTIME_CANARY_ROLLBACK_FAILS", "2"))), 1, 20))

    stable = _load_json(RUNS / "runtime_profile_stable.json")
    active = _load_json(RUNS / "runtime_profile_active.json")
    state = _load_json(state_path)
    if not isinstance(state, dict):
        state = {}
    status = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "action": "none",
        "required_passes": int(required_passes),
        "rollback_fails": int(rollback_fails),
        "reasons": [],
    }

    if selected:
        sel = _profile_payload(selected)
        _write_profile("runtime_profile_selected.json", sel)
        print(f"✅ Selected runtime profile: floor={sel['runtime_total_floor']} disable={sel['disable_governors']}")

        if not stable:
            _write_profile("runtime_profile_stable.json", sel)
            _write_profile("runtime_profile_active.json", sel)
            _remove_profile("runtime_profile_challenger.json")
            state = {"challenger_signature": "", "passes": 0, "fails": 0}
            status["action"] = "bootstrap_stable"
        else:
            st_sig = _profile_signature(stable)
            sel_sig = _profile_signature(sel)
            prev_sig = str(state.get("challenger_signature", ""))
            passes = int(state.get("passes", 0))
            fails = int(state.get("fails", 0))
            if sel_sig == st_sig:
                _remove_profile("runtime_profile_challenger.json")
                _write_profile("runtime_profile_active.json", stable)
                state = {"challenger_signature": "", "passes": 0, "fails": 0}
                status["action"] = "stable_retained"
            else:
                ok, reasons = _canary_qualifies(stable, sel)
                if prev_sig != sel_sig:
                    passes = 0
                    fails = 0
                if ok:
                    passes += 1
                    fails = 0
                    _write_profile("runtime_profile_challenger.json", sel)
                    _write_profile("runtime_profile_active.json", stable)
                    status["action"] = "canary_armed"
                    if passes >= required_passes:
                        _write_profile("runtime_profile_stable.json", sel)
                        _write_profile("runtime_profile_active.json", sel)
                        _remove_profile("runtime_profile_challenger.json")
                        status["action"] = "promoted"
                        state = {"challenger_signature": "", "passes": 0, "fails": 0}
                    else:
                        state = {"challenger_signature": sel_sig, "passes": passes, "fails": fails}
                else:
                    fails += 1
                    passes = 0
                    _write_profile("runtime_profile_challenger.json", sel)
                    _write_profile("runtime_profile_active.json", stable)
                    status["action"] = "canary_failed"
                    status["reasons"] = reasons
                    if fails >= rollback_fails:
                        _remove_profile("runtime_profile_challenger.json")
                        status["action"] = "challenger_rollback"
                        state = {"challenger_signature": "", "passes": 0, "fails": 0}
                    else:
                        state = {"challenger_signature": sel_sig, "passes": passes, "fails": fails}
    else:
        print("(!) No valid runtime profile found in search grid.")
        if stable:
            _write_profile("runtime_profile_active.json", stable)
        status["action"] = "no_valid_profile"

    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    status["state"] = state
    status["stable"] = _load_json(RUNS / "runtime_profile_stable.json")
    status["active"] = _load_json(RUNS / "runtime_profile_active.json")
    status["challenger"] = _load_json(RUNS / "runtime_profile_challenger.json")
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    _write_progress(evaluated=total, total=total, best=best_so_far)
    print(f"✅ Wrote {RUNS/'runtime_combo_search.json'}")
    if selected:
        print(f"✅ Wrote {RUNS/'runtime_profile_selected.json'}")
    print(f"✅ Wrote {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
