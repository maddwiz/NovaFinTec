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


def _load_series(paths: list[Path]) -> np.ndarray | None:
    for p in paths:
        if not p.exists():
            continue
        try:
            a = np.loadtxt(p, delimiter=",")
        except Exception:
            try:
                a = np.loadtxt(p, delimiter=",", skiprows=1)
            except Exception:
                continue
        a = np.asarray(a, float)
        if a.ndim == 2 and a.shape[1] >= 1:
            a = a[:, -1]
        a = np.nan_to_num(a.ravel(), nan=0.0, posinf=0.0, neginf=0.0)
        if a.size > 0:
            return a
    return None


def _returns_metrics(r: np.ndarray) -> dict:
    x = np.asarray(r, float).ravel()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    n = int(x.size)
    if n <= 0:
        return {"sharpe": 0.0, "hit_rate": 0.0, "max_drawdown": 0.0, "n": 0}
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    sharpe = float((mu / (sd + 1e-12)) * np.sqrt(252.0)) if sd > 0 else 0.0
    hit_rate = float(np.mean(x > 0.0))
    eq = np.cumprod(1.0 + np.clip(x, -0.95, 10.0))
    peak = np.maximum.accumulate(eq)
    dd = eq / (peak + 1e-12) - 1.0
    mdd = float(np.min(dd)) if dd.size else 0.0
    return {"sharpe": sharpe, "hit_rate": hit_rate, "max_drawdown": mdd, "n": n}


def _governor_train_validation_split(total_rows: int, train_frac: float, holdout_min: int) -> tuple[int, int]:
    n = int(max(0, total_rows))
    if n <= 1:
        return 0, n
    frac = float(np.clip(float(train_frac), 0.50, 0.95))
    min_hold = int(max(20, int(holdout_min)))
    frac_hold = int(max(1, round(n * (1.0 - frac))))
    hold = max(min_hold, frac_hold)
    if hold >= n:
        hold = max(1, n // 3)
        hold = min(hold, n - 1)
    train = max(1, n - hold)
    hold = n - train
    return int(train), int(hold)


def _load_daily_returns_for_governor_eval() -> np.ndarray | None:
    return _load_series(
        [
            RUNS / "daily_returns.csv",
            ROOT / "daily_returns.csv",
            ROOT / "portfolio_daily_returns.csv",
        ]
    )


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
        [PY, str(ROOT / "tools" / "run_external_holdout_validation.py")],
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
    latest = strict.get("metrics_oos_latest", {}) if isinstance(strict.get("metrics_oos_latest"), dict) else {}
    promo = _load_json(RUNS / "q_promotion_gate.json")
    stress = _load_json(RUNS / "cost_stress_validation.json")
    health = _load_json(RUNS / "health_alerts.json")
    cinfo = _load_json(RUNS / "daily_costs_info.json")
    ext = _load_json(RUNS / "external_holdout_validation.json")
    extm = ext.get("metrics_external_holdout_net", {}) if isinstance(ext.get("metrics_external_holdout_net"), dict) else {}

    strict_robust_sh = float(robust.get("sharpe", 0.0))
    strict_robust_hit = float(robust.get("hit_rate", 0.0))
    strict_robust_mdd = float(robust.get("max_drawdown", 0.0))

    train_frac = float(np.clip(float(env.get("Q_RUNTIME_SEARCH_GOVERNOR_TRAIN_FRAC", "0.75")), 0.50, 0.95))
    holdout_min = int(np.clip(int(float(env.get("Q_RUNTIME_SEARCH_GOVERNOR_HOLDOUT_MIN", "252"))), 20, 5000))
    governor_validation_min_rows = int(
        np.clip(
            int(float(env.get("Q_RUNTIME_SEARCH_GOVERNOR_VALIDATION_MIN_ROWS", str(holdout_min)))),
            20,
            5000,
        )
    )
    gov_train_rows = 0
    gov_val_rows = 0
    gov_train_metrics = {"sharpe": 0.0, "hit_rate": 0.0, "max_drawdown": 0.0, "n": 0}
    gov_val_metrics = {"sharpe": 0.0, "hit_rate": 0.0, "max_drawdown": 0.0, "n": 0}

    r = _load_daily_returns_for_governor_eval()
    if r is not None and len(r) >= 3:
        gov_train_rows, gov_val_rows = _governor_train_validation_split(
            total_rows=len(r),
            train_frac=train_frac,
            holdout_min=holdout_min,
        )
        train_r = np.asarray(r[:gov_train_rows], float)
        val_r = np.asarray(r[gov_train_rows : gov_train_rows + gov_val_rows], float)
        gov_train_metrics = _returns_metrics(train_r)
        gov_val_metrics = _returns_metrics(val_r)

    robust_sharpe = float(gov_val_metrics.get("sharpe", strict_robust_sh))
    robust_hit_rate = float(gov_val_metrics.get("hit_rate", strict_robust_hit))
    robust_max_drawdown = float(gov_val_metrics.get("max_drawdown", strict_robust_mdd))
    if int(gov_val_metrics.get("n", 0)) <= 0:
        robust_sharpe = strict_robust_sh
        robust_hit_rate = strict_robust_hit
        robust_max_drawdown = strict_robust_mdd
    governor_validation_ok = bool(int(gov_val_metrics.get("n", 0)) >= int(governor_validation_min_rows))

    return {
        "robust_sharpe": robust_sharpe,
        "robust_hit_rate": robust_hit_rate,
        "robust_max_drawdown": robust_max_drawdown,
        "strict_robust_sharpe": strict_robust_sh,
        "strict_robust_hit_rate": strict_robust_hit,
        "strict_robust_max_drawdown": strict_robust_mdd,
        "governor_train_rows": int(gov_train_rows),
        "governor_validation_rows": int(gov_val_rows),
        "governor_validation_min_rows": int(governor_validation_min_rows),
        "governor_validation_ok": bool(governor_validation_ok),
        "governor_train_sharpe": float(gov_train_metrics.get("sharpe", 0.0)),
        "governor_train_hit_rate": float(gov_train_metrics.get("hit_rate", 0.0)),
        "governor_train_max_drawdown": float(gov_train_metrics.get("max_drawdown", 0.0)),
        "governor_validation_sharpe": float(gov_val_metrics.get("sharpe", 0.0)),
        "governor_validation_hit_rate": float(gov_val_metrics.get("hit_rate", 0.0)),
        "governor_validation_max_drawdown": float(gov_val_metrics.get("max_drawdown", 0.0)),
        "latest_oos_sharpe": float(latest.get("sharpe", 0.0)),
        "latest_oos_hit_rate": float(latest.get("hit_rate", 0.0)),
        "latest_oos_max_drawdown": float(latest.get("max_drawdown", 0.0)),
        "latest_oos_n": int(latest.get("n", 0)),
        "ann_cost_estimate": float(cinfo.get("ann_cost_estimate", 0.0)),
        "mean_turnover": float(cinfo.get("mean_turnover", 0.0)),
        "mean_effective_cost_bps": float(cinfo.get("mean_effective_cost_bps", 0.0)),
        "external_holdout_ok": bool(ext.get("ok", False)),
        "external_holdout_sharpe": float(extm.get("sharpe", 0.0)),
        "external_holdout_hit_rate": float(extm.get("hit_rate", 0.0)),
        "external_holdout_max_drawdown": float(extm.get("max_drawdown", 0.0)),
        "external_holdout_n": int(extm.get("n", ext.get("rows", 0))),
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
    latest_sh = float(row.get("latest_oos_sharpe", 0.0))
    latest_hit = float(row.get("latest_oos_hit_rate", 0.0))
    latest_mdd = abs(float(row.get("latest_oos_max_drawdown", 0.0)))
    latest_n = int(row.get("latest_oos_n", 0))
    latest_sh_ref = float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_LATEST_SHARPE_REF", "0.90")), -2.0, 10.0))
    latest_hit_ref = float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_LATEST_HIT_REF", "0.48")), 0.0, 1.0))
    latest_mdd_ref = float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_LATEST_MDD_REF", "0.12")), 0.001, 2.0))
    latest_sh_w = float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_LATEST_SHARPE_WEIGHT", "0.25")), 0.0, 10.0))
    latest_hit_w = float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_LATEST_HIT_WEIGHT", "0.25")), 0.0, 10.0))
    latest_mdd_pen = float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_LATEST_MDD_PENALTY", "1.5")), 0.0, 25.0))
    latest_min_n = int(np.clip(int(float(os.getenv("Q_RUNTIME_SEARCH_LATEST_MIN_N", "126"))), 1, 1000000))
    latest_n_pen = float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_LATEST_N_PENALTY", "0.50")), 0.0, 25.0))
    latest_over_mdd = max(0.0, latest_mdd - latest_mdd_ref)
    latest_n_shortfall = max(0.0, (latest_min_n - latest_n) / max(1.0, float(latest_min_n)))
    base_score = float(
        sh
        + hit_w * (hit - target_hit)
        + latest_sh_w * (latest_sh - latest_sh_ref)
        + latest_hit_w * (latest_hit - latest_hit_ref)
        - latest_mdd_pen * latest_over_mdd
        - latest_n_pen * latest_n_shortfall
        - mdd_w * over_mdd
        - cost_w * over_cost
        - turn_w * over_turn
    )
    complexity_penalty, _ = _runtime_complexity_penalty(row)
    return float(base_score - complexity_penalty)


def _runtime_complexity_penalty(row: dict) -> tuple[float, dict]:
    base_w = float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_COMPLEXITY_PENALTY", "0.08")), 0.0, 2.0))
    strength_w = float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_COMPLEXITY_STRENGTH_PENALTY", "0.04")), 0.0, 2.0))
    flags_total = int(max(0, int(row.get("search_flag_count", 0))))
    disabled = row.get("disable_governors", [])
    if not isinstance(disabled, list):
        disabled = []
    disabled_n = len([x for x in disabled if str(x).strip()])
    enabled_n = max(0, flags_total - disabled_n)

    optional_n = 0
    optional_n += 1 if bool(row.get("enable_asset_class_diversification", False)) else 0
    macro_strength = abs(float(row.get("macro_proxy_strength", 0.0)))
    capacity_strength = abs(float(row.get("capacity_impact_strength", 0.0)))
    blend_strength = abs(float(row.get("uncertainty_macro_shock_blend", 0.0)))
    optional_n += 1 if macro_strength > 1e-9 else 0
    optional_n += 1 if capacity_strength > 1e-9 else 0
    optional_n += 1 if blend_strength > 1e-9 else 0
    denom = max(1, flags_total + 4)
    active_frac = float((enabled_n + optional_n) / float(denom))
    strength_sum = float(macro_strength + capacity_strength + blend_strength)
    penalty = float(base_w * active_frac + strength_w * strength_sum)
    return penalty, {
        "complexity_penalty": float(penalty),
        "complexity_active_fraction": float(active_frac),
        "complexity_strength_sum": float(strength_sum),
        "complexity_enabled_flags": int(enabled_n),
        "complexity_optional_active": int(optional_n),
        "search_flag_count": int(flags_total),
    }


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
        "governor_train_rows": int(row.get("governor_train_rows", 0)),
        "governor_validation_rows": int(row.get("governor_validation_rows", 0)),
        "governor_validation_min_rows": int(row.get("governor_validation_min_rows", 0)),
        "governor_validation_ok": bool(row.get("governor_validation_ok", True)),
        "governor_train_sharpe": float(row.get("governor_train_sharpe", 0.0)),
        "governor_train_hit_rate": float(row.get("governor_train_hit_rate", 0.0)),
        "governor_train_max_drawdown": float(row.get("governor_train_max_drawdown", 0.0)),
        "governor_validation_sharpe": float(row.get("governor_validation_sharpe", 0.0)),
        "governor_validation_hit_rate": float(row.get("governor_validation_hit_rate", 0.0)),
        "governor_validation_max_drawdown": float(row.get("governor_validation_max_drawdown", 0.0)),
        "robust_sharpe": float(row.get("robust_sharpe", 0.0)),
        "robust_hit_rate": float(row.get("robust_hit_rate", 0.0)),
        "robust_max_drawdown": float(row.get("robust_max_drawdown", 0.0)),
        "latest_oos_sharpe": float(row.get("latest_oos_sharpe", 0.0)),
        "latest_oos_hit_rate": float(row.get("latest_oos_hit_rate", 0.0)),
        "latest_oos_max_drawdown": float(row.get("latest_oos_max_drawdown", 0.0)),
        "latest_oos_n": int(row.get("latest_oos_n", 0)),
        "ann_cost_estimate": float(row.get("ann_cost_estimate", 0.0)),
        "mean_turnover": float(row.get("mean_turnover", 0.0)),
        "mean_effective_cost_bps": float(row.get("mean_effective_cost_bps", 0.0)),
        "external_holdout_ok": bool(row.get("external_holdout_ok", False)),
        "external_holdout_sharpe": float(row.get("external_holdout_sharpe", 0.0)),
        "external_holdout_hit_rate": float(row.get("external_holdout_hit_rate", 0.0)),
        "external_holdout_max_drawdown": float(row.get("external_holdout_max_drawdown", 0.0)),
        "external_holdout_n": int(row.get("external_holdout_n", 0)),
        "complexity_penalty": float(row.get("complexity_penalty", 0.0)),
        "complexity_active_fraction": float(row.get("complexity_active_fraction", 0.0)),
        "complexity_strength_sum": float(row.get("complexity_strength_sum", 0.0)),
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
    latest_min_n = int(np.clip(int(float(os.getenv("Q_RUNTIME_CANARY_LATEST_MIN_N", "126"))), 1, 1000000))
    max_latest_sharpe_drop = float(
        np.clip(float(os.getenv("Q_RUNTIME_CANARY_MAX_LATEST_SHARPE_DROP", "0.08")), 0.0, 5.0)
    )
    max_latest_hit_drop = float(np.clip(float(os.getenv("Q_RUNTIME_CANARY_MAX_LATEST_HIT_DROP", "0.0075")), 0.0, 0.25))
    max_latest_mdd_worsen = float(
        np.clip(float(os.getenv("Q_RUNTIME_CANARY_MAX_LATEST_ABS_MDD_WORSEN", "0.015")), 0.0, 1.0)
    )

    st_sh = float(stable.get("robust_sharpe", 0.0))
    st_hit = float(stable.get("robust_hit_rate", 0.0))
    st_mdd = abs(float(stable.get("robust_max_drawdown", 0.0)))
    st_score = float(stable.get("score", st_sh))

    def _opt_float(src: dict, key: str) -> float | None:
        if key not in src:
            return None
        try:
            v = float(src.get(key))
        except Exception:
            return None
        if not np.isfinite(v):
            return None
        return v

    def _opt_nonneg(src: dict, key: str) -> float | None:
        v = _opt_float(src, key)
        if v is None:
            return None
        return max(0.0, v)

    st_ann_cost = _opt_nonneg(stable, "ann_cost_estimate")
    st_turn = _opt_nonneg(stable, "mean_turnover")
    st_latest_sh = _opt_float(stable, "latest_oos_sharpe")
    st_latest_hit = _opt_float(stable, "latest_oos_hit_rate")
    st_latest_mdd = _opt_float(stable, "latest_oos_max_drawdown")
    st_latest_n = _opt_nonneg(stable, "latest_oos_n")
    ca_sh = float(candidate.get("robust_sharpe", 0.0))
    ca_hit = float(candidate.get("robust_hit_rate", 0.0))
    ca_mdd = abs(float(candidate.get("robust_max_drawdown", 0.0)))
    ca_score = float(candidate.get("score", ca_sh))
    ca_ann_cost = _opt_nonneg(candidate, "ann_cost_estimate")
    ca_turn = _opt_nonneg(candidate, "mean_turnover")
    ca_latest_sh = _opt_float(candidate, "latest_oos_sharpe")
    ca_latest_hit = _opt_float(candidate, "latest_oos_hit_rate")
    ca_latest_mdd = _opt_float(candidate, "latest_oos_max_drawdown")
    ca_latest_n = _opt_nonneg(candidate, "latest_oos_n")
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
    have_latest = (
        (st_latest_n is not None)
        and (ca_latest_n is not None)
        and (int(st_latest_n) >= latest_min_n)
        and (int(ca_latest_n) >= latest_min_n)
    )
    if have_latest and (st_latest_sh is not None) and (ca_latest_sh is not None):
        if (st_latest_sh - ca_latest_sh) > max_latest_sharpe_drop:
            reasons.append(f"latest_sharpe_drop>{max_latest_sharpe_drop:.3f} ({st_latest_sh - ca_latest_sh:.3f})")
    if have_latest and (st_latest_hit is not None) and (ca_latest_hit is not None):
        if ca_latest_hit < (st_latest_hit - max_latest_hit_drop):
            reasons.append(f"latest_hit_drop>{max_latest_hit_drop:.4f} ({st_latest_hit - ca_latest_hit:.4f})")
    if have_latest and (st_latest_mdd is not None) and (ca_latest_mdd is not None):
        if abs(ca_latest_mdd) > (abs(st_latest_mdd) + max_latest_mdd_worsen):
            reasons.append(f"latest_mdd_worsen>{max_latest_mdd_worsen:.3f} ({abs(ca_latest_mdd) - abs(st_latest_mdd):.3f})")
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
            "search_flag_count": int(len(flags)),
            "enable_asset_class_diversification": bool(int(use_asset_class) == 1),
            "macro_proxy_strength": float(macro_strength),
            "capacity_impact_strength": float(capacity_strength),
            "uncertainty_macro_shock_blend": float(macro_blend),
            **out,
        }
        row["score"] = _score_row(row)
        _, cdetail = _runtime_complexity_penalty(row)
        row.update(cdetail)
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
            best_so_far = _profile_payload(row)
        if i % 8 == 0 or i == total:
            print(f"… evaluated {i}/{total} combos")
            _write_progress(evaluated=i, total=total, best=best_so_far)

    require_gov_val = str(os.getenv("Q_RUNTIME_SEARCH_REQUIRE_GOVERNOR_VALIDATION", "1")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    def _valid(r: dict) -> bool:
        rc_ok = all(int(x.get("code", 1)) == 0 for x in (r.get("rc") or []))
        return (
            rc_ok
            and bool(r.get("promotion_ok", False))
            and bool(r.get("cost_stress_ok", False))
            and bool(r.get("health_ok", False))
            and int(r.get("health_alerts_hard", 999)) == 0
            and ((not require_gov_val) or bool(r.get("governor_validation_ok", True)))
        )

    def _relaxed_valid(r: dict) -> bool:
        rc_ok = all(int(x.get("code", 1)) == 0 for x in (r.get("rc") or []))
        return (
            rc_ok
            and bool(r.get("cost_stress_ok", False))
            and bool(r.get("health_ok", False))
            and int(r.get("health_alerts_hard", 999)) == 0
            and ((not require_gov_val) or bool(r.get("governor_validation_ok", True)))
        )

    valid = [r for r in rows if _valid(r)]
    relaxed_valid = [r for r in rows if _relaxed_valid(r)]
    valid_sorted = sorted(
        valid,
        key=lambda r: (
            float(r.get("score", -1e9)),
            float(r.get("robust_sharpe", 0.0)),
            float(r.get("robust_hit_rate", 0.0)),
        ),
        reverse=True,
    )
    relaxed_valid_sorted = sorted(
        relaxed_valid,
        key=lambda r: (
            float(r.get("score", -1e9)),
            float(r.get("robust_sharpe", 0.0)),
            float(r.get("robust_hit_rate", 0.0)),
        ),
        reverse=True,
    )

    selected = valid_sorted[0] if valid_sorted else None
    selected_relaxed = relaxed_valid_sorted[0] if relaxed_valid_sorted else None
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
        "rows_relaxed_valid": len(relaxed_valid_sorted),
        "score_formula": {
            "score": "sharpe + hit_weight*(hit-target_hit) + latest_terms - risk/cost penalties - complexity_penalty*(active_fraction) - complexity_strength_penalty*(macro+capacity+blend)",
            "target_hit": float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_TARGET_HIT", "0.49")), 0.0, 1.0)),
            "hit_weight": float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_HIT_WEIGHT", "0.75")), 0.0, 10.0)),
            "mdd_ref": float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_MDD_REF", "0.04")), 0.001, 1.0)),
            "mdd_penalty": float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_MDD_PENALTY", "4.0")), 0.0, 25.0)),
            "cost_ref_annual": float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_COST_REF_ANNUAL", "0.02")), 0.0, 1.0)),
            "cost_penalty": float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_COST_PENALTY", "3.0")), 0.0, 100.0)),
            "turnover_ref_daily": float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_TURNOVER_REF_DAILY", "0.06")), 0.0, 5.0)),
            "turnover_penalty": float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_TURNOVER_PENALTY", "1.5")), 0.0, 100.0)),
            "complexity_penalty": float(np.clip(float(os.getenv("Q_RUNTIME_SEARCH_COMPLEXITY_PENALTY", "0.08")), 0.0, 2.0)),
            "complexity_strength_penalty": float(
                np.clip(float(os.getenv("Q_RUNTIME_SEARCH_COMPLEXITY_STRENGTH_PENALTY", "0.04")), 0.0, 2.0)
            ),
        },
        "top_valid": valid_sorted[:20],
        "top_relaxed": relaxed_valid_sorted[:20],
        "selected": selected,
        "selected_relaxed": selected_relaxed,
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
        if selected_relaxed:
            sel_relaxed = _profile_payload(selected_relaxed)
            (RUNS / "runtime_profile_selected_relaxed.json").write_text(
                json.dumps(sel_relaxed, indent=2),
                encoding="utf-8",
            )
            print("(!) No promotable runtime profile found; wrote relaxed best candidate for tuning.")
            if stable:
                _write_profile("runtime_profile_active.json", stable)
            status["action"] = "no_promotable_profile"
            status["reasons"] = ["all_candidates_failed_promotion_gate"]
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
