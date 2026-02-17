#!/usr/bin/env python3
# Final portfolio assembler:
# Picks best base weights then applies (if available):
#   cluster caps → adaptive caps → drawdown scaler → turnover governor
#   → council gate → council/meta leverage → heartbeat/legacy/hive/global/quality/novaspine governors
# Outputs:
#   runs_plus/portfolio_weights_final.csv
# Appends a card to report_*.

import json
import os
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"; RUNS.mkdir(exist_ok=True)

import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from qmods.concentration_governor import govern_matrix
from qmods.guardrails_bundle import apply_turnover_budget_governor

TRACE_STEPS = [
    "asset_class_diversification",
    "rank_sleeve_blend",
    "low_vol_sleeve_blend",
    "turnover_governor",
    "meta_execution_gate",
    "execution_hit_gate",
    "council_gate",
    "meta_mix_leverage",
    "meta_mix_reliability",
    "heartbeat_scaler",
    "legacy_scaler",
    "dna_stress_governor",
    "symbolic_governor",
    "dream_coherence",
    "reflex_health_governor",
    "hive_diversification",
    "hive_persistence",
    "global_governor",
    "quality_governor",
    "regime_fracture_governor",
    "regime_moe_governor",
    "uncertainty_sizing",
    "capacity_impact_guard",
    "credit_leadlag_overlay",
    "microstructure_overlay",
    "calendar_event_overlay",
    "macro_proxy_guard",
    "novaspine_context_boost",
    "novaspine_hive_boost",
    "shock_mask_guard",
    "vol_target",
    "runtime_floor",
]


def _load_governor_profile() -> dict:
    p = RUNS / "governor_profile.json"
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text())
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


_GOV_PROFILE = _load_governor_profile()


def _load_governor_param_profile() -> dict:
    p = RUNS / "governor_params_profile.json"
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text())
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


_GOV_PARAM_PROFILE = _load_governor_param_profile()


def _governor_params() -> dict:
    obj = _GOV_PARAM_PROFILE if isinstance(_GOV_PARAM_PROFILE, dict) else {}
    params = obj.get("parameters", obj)
    return params if isinstance(params, dict) else {}


def _env_or_profile_float(env_key: str, profile_key: str, default: float, lo: float, hi: float) -> float:
    env_raw = str(os.getenv(env_key, "")).strip()
    if env_raw:
        try:
            return float(np.clip(float(env_raw), lo, hi))
        except Exception:
            pass
    params = _governor_params()
    if profile_key in params:
        try:
            return float(np.clip(float(params.get(profile_key)), lo, hi))
        except Exception:
            pass
    return float(np.clip(float(default), lo, hi))


def _env_or_profile_int(env_key: str, profile_key: str, default: int, lo: int, hi: int) -> int:
    env_raw = str(os.getenv(env_key, "")).strip()
    if env_raw:
        try:
            return int(np.clip(int(float(env_raw)), lo, hi))
        except Exception:
            pass
    params = _governor_params()
    if profile_key in params:
        try:
            return int(np.clip(int(float(params.get(profile_key))), lo, hi))
        except Exception:
            pass
    return int(np.clip(int(default), lo, hi))


def _env_or_profile_bool(env_key: str, profile_key: str, default: bool) -> bool:
    env_raw = str(os.getenv(env_key, "")).strip()
    if env_raw:
        return env_raw.lower() in {"1", "true", "yes", "on"}
    params = _governor_params()
    if profile_key in params:
        raw = params.get(profile_key)
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}
    return bool(default)


def _apply_governor_strength(vec: np.ndarray, strength: float, lo: float = 0.0, hi: float = 2.0) -> np.ndarray:
    """Blend a governor scalar with identity (1.0) by strength.

    strength=0 -> no effect (all ones)
    strength=1 -> full effect (original vec)
    """
    v = np.asarray(vec, float)
    s = float(np.clip(float(strength), 0.0, 2.0))
    out = 1.0 + s * (v - 1.0)
    return np.clip(out, lo, hi)


def _disabled_governors() -> set[str]:
    out = set()
    vals = _GOV_PROFILE.get("disable_governors", [])
    if isinstance(vals, list):
        for token in vals:
            t = str(token).strip().lower()
            if t:
                out.add(t)
    raw = str(os.getenv("Q_DISABLE_GOVERNORS", "")).strip()
    if raw:
        for token in raw.split(","):
            t = str(token).strip().lower()
            if t:
                out.add(t)
    return out


_DISABLED_GOVS = _disabled_governors()


def _gov_enabled(name: str) -> bool:
    return str(name).strip().lower() not in _DISABLED_GOVS

def _runtime_total_floor_default() -> float:
    env_or_param = _env_or_profile_float(
        "Q_RUNTIME_TOTAL_FLOOR",
        "runtime_total_floor",
        np.nan,
        0.0,
        1.0,
    )
    if np.isfinite(env_or_param):
        return float(env_or_param)
    try:
        prof_v = float(_GOV_PROFILE.get("runtime_total_floor", 0.10))
    except Exception:
        prof_v = 0.10
    return float(np.clip(prof_v, 0.0, 1.0))


def _base_weight_candidates() -> list[str]:
    cands = ["runs_plus/weights_regime.csv"]
    use_asset_class = _env_or_profile_bool(
        "Q_ENABLE_ASSET_CLASS_DIVERSIFICATION",
        "enable_asset_class_diversification",
        False,
    )
    if use_asset_class:
        cands.append("runs_plus/weights_asset_class_diversified.csv")
    cands.extend(
        [
            "runs_plus/weights_tail_blend.csv",
            "runs_plus/portfolio_weights.csv",
            "portfolio_weights.csv",
        ]
    )
    return cands


def _auto_turnover_govern(w: np.ndarray):
    """
    Optional fallback turnover throttle when no precomputed turnover-governed
    matrix is present in runs_plus/.
    """
    enabled = str(os.getenv("Q_ENABLE_AUTO_TURNOVER_GOV", "1")).strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return None, None, None

    arr = np.asarray(w, float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return None, None, None

    max_step = _env_or_profile_float("Q_AUTO_TURNOVER_MAX_STEP", "auto_turnover_max_step", 0.30, 0.01, 5.0)
    budget_window = _env_or_profile_int("Q_AUTO_TURNOVER_BUDGET_WINDOW", "auto_turnover_budget_window", 5, 1, 120)
    budget_limit = _env_or_profile_float("Q_AUTO_TURNOVER_BUDGET_LIMIT", "auto_turnover_budget_limit", 1.00, 0.01, 20.0)

    res = apply_turnover_budget_governor(
        arr,
        max_step_turnover=max_step,
        budget_window=budget_window,
        budget_limit=budget_limit,
    )

    np.savetxt(RUNS / "weights_turnover_budget_governed.csv", res.weights, delimiter=",")
    np.savetxt(RUNS / "turnover_before.csv", res.turnover_before, delimiter=",")
    np.savetxt(RUNS / "turnover_after.csv", res.turnover_after, delimiter=",")
    np.savetxt(RUNS / "turnover_budget_rolling_after.csv", res.rolling_turnover_after, delimiter=",")
    (RUNS / "turnover_governor_auto_info.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "source": "build_final_portfolio:auto",
                "max_step_turnover": max_step,
                "budget_window": budget_window,
                "budget_limit": budget_limit,
                "turnover_before_mean": float(np.mean(res.turnover_before)) if res.turnover_before.size else 0.0,
                "turnover_after_mean": float(np.mean(res.turnover_after)) if res.turnover_after.size else 0.0,
                "turnover_after_max": float(np.max(res.turnover_after)) if res.turnover_after.size else 0.0,
            },
            indent=2,
        )
    )

    tscale = np.ones(arr.shape[0], dtype=float)
    if res.scale_applied.size:
        tscale[1 : 1 + len(res.scale_applied)] = np.asarray(res.scale_applied, float).ravel()[: arr.shape[0] - 1]
    return res.weights, "turnover_budget_auto", tscale

def load_mat(rel):
    p = ROOT/rel
    if not p.exists(): return None
    try:
        a = np.loadtxt(p, delimiter=",")
    except:
        a = np.loadtxt(p, delimiter=",", skiprows=1)
    if a.ndim == 1: a = a.reshape(-1,1)
    return a

def load_series(rel):
    p = ROOT/rel
    if not p.exists(): return None
    try:
        a = np.loadtxt(p, delimiter=",")
    except Exception:
        try:
            a = np.loadtxt(p, delimiter=",", skiprows=1)
        except Exception:
            vals = []
            try:
                with p.open("r", encoding="utf-8", errors="ignore") as f:
                    first = True
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = [s.strip() for s in line.split(",")]
                        if first and any(tok.lower() in ("date", "time", "timestamp") for tok in parts):
                            first = False
                            continue
                        first = False
                        try:
                            vals.append(float(parts[-1]))
                        except Exception:
                            continue
            except Exception:
                return None
            return np.asarray(vals, float).ravel() if vals else None
    a = np.asarray(a, float)
    if a.ndim == 2 and a.shape[1] >= 1:
        a = a[:, -1]
    return a.ravel()

def first_mat(paths):
    for rel in paths:
        a = load_mat(rel)
        if a is not None: return a, rel
    return None, None


def compute_vol_target_scalars(
    weights: np.ndarray,
    asset_returns: np.ndarray,
    *,
    target_annual_vol: float = 0.10,
    lookback: int = 63,
    min_scalar: float = 0.40,
    max_scalar: float = 1.80,
    smooth_alpha: float = 0.20,
) -> np.ndarray:
    w = np.asarray(weights, float)
    a = np.asarray(asset_returns, float)
    T = min(w.shape[0], a.shape[0])
    if T <= 0:
        return np.ones(0, dtype=float)
    r = np.sum(w[:T] * a[:T], axis=1)
    lb = int(np.clip(int(lookback), 2, max(2, T)))
    tgt = float(np.clip(float(target_annual_vol), 0.01, 2.0))
    lo = float(np.clip(float(min_scalar), 0.01, 5.0))
    hi = float(np.clip(float(max_scalar), lo, 5.0))
    alpha = float(np.clip(float(smooth_alpha), 0.0, 1.0))

    raw = np.ones(T, dtype=float)
    for t in range(T):
        i0 = max(0, t - lb + 1)
        seg = r[i0 : t + 1]
        vol_d = float(np.std(seg)) if seg.size else 0.0
        vol_a = vol_d * np.sqrt(252.0)
        if vol_a > 1e-9:
            raw[t] = tgt / vol_a
        else:
            raw[t] = 1.0
    raw = np.clip(raw, lo, hi)

    if alpha <= 0.0:
        sm = raw
    else:
        sm = np.zeros(T, dtype=float)
        sm[0] = raw[0]
        for t in range(1, T):
            sm[t] = alpha * raw[t] + (1.0 - alpha) * sm[t - 1]
    return np.clip(sm, lo, hi)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, float)
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def compute_hit_gate_scalars(
    hit_proxy: np.ndarray,
    *,
    threshold: float = 0.50,
    floor: float = 0.35,
    ceiling: float = 1.05,
    slope: float = 14.0,
) -> np.ndarray:
    hp = np.clip(np.asarray(hit_proxy, float).ravel(), 0.0, 1.0)
    thr = float(np.clip(float(threshold), 0.35, 0.75))
    lo = float(np.clip(float(floor), 0.0, 1.0))
    hi = float(np.clip(float(ceiling), lo, 1.5))
    k = float(np.clip(float(slope), 1.0, 40.0))
    raw = _sigmoid((hp - thr) * k)
    out = lo + (hi - lo) * raw
    return np.clip(out, lo, hi)


def apply_signal_deadzone(
    weights: np.ndarray,
    *,
    base_deadzone: float,
    hit_proxy: np.ndarray | None = None,
    hit_threshold: float = 0.50,
    hit_sensitivity: float = 1.0,
) -> tuple[np.ndarray, dict]:
    w = np.asarray(weights, float).copy()
    T = w.shape[0]
    dz0 = float(np.clip(float(base_deadzone), 0.0, 0.10))
    if T == 0 or dz0 <= 0.0:
        return w, {
            "enabled": False,
            "base_deadzone": dz0,
            "active_before": int(np.count_nonzero(np.abs(w) > 0.0)),
            "active_after": int(np.count_nonzero(np.abs(w) > 0.0)),
            "pruned_fraction": 0.0,
        }

    if hit_proxy is None:
        dz = np.full(T, dz0, dtype=float)
    else:
        hp = np.clip(np.asarray(hit_proxy, float).ravel(), 0.0, 1.0)
        if hp.size < T:
            fill = float(hp[-1]) if hp.size else 0.5
            pad = np.full(T - hp.size, fill, dtype=float)
            hp = np.concatenate([hp, pad], axis=0)
        hp = hp[:T]
        hs = float(np.clip(float(hit_sensitivity), 0.0, 3.0))
        weak = np.clip((float(hit_threshold) - hp) / 0.5, 0.0, 1.0)
        dz = dz0 * (1.0 + hs * weak)

    before = int(np.count_nonzero(np.abs(w) > 0.0))
    mask = np.abs(w) < dz.reshape(-1, 1)
    w[mask] = 0.0
    after = int(np.count_nonzero(np.abs(w) > 0.0))
    pruned = float((before - after) / max(1, before))
    info = {
        "enabled": True,
        "base_deadzone": dz0,
        "deadzone_mean": float(np.mean(dz)),
        "deadzone_max": float(np.max(dz)),
        "active_before": before,
        "active_after": after,
        "pruned_fraction": pruned,
    }
    return w, info


def append_card(title, html):
    if str(os.getenv("Q_DISABLE_REPORT_CARDS", "0")).strip().lower() in {"1", "true", "yes", "on"}:
        return
    for name in ["report_all.html","report_best_plus.html","report_plus.html","report.html"]:
        f = ROOT/name
        if not f.exists(): continue
        txt = f.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        f.write_text(txt.replace("</body>", card+"</body>") if "</body>" in txt else txt+card, encoding="utf-8")

if __name__ == "__main__":
    # 1) Base weights preference
    W, source = first_mat(_base_weight_candidates())
    if W is None:
        print("(!) No base weights found; run your pipeline first."); raise SystemExit(0)
    steps = [f"base={source}"]

    T, N = W.shape
    trace = {}

    def _trace_put(name, vec=None):
        x = np.ones(T, dtype=float)
        if vec is not None:
            v = np.asarray(vec, float).ravel()
            L = min(T, len(v))
            if L > 0:
                x[:L] = v[:L]
        trace[name] = x

    if str(source or "").endswith("weights_asset_class_diversified.csv"):
        steps.append("asset_class_diversification")
        _trace_put("asset_class_diversification", np.ones(T, dtype=float))

    council_gate_strength = _env_or_profile_float(
        "Q_COUNCIL_GATE_STRENGTH",
        "council_gate_strength",
        1.0,
        0.0,
        2.0,
    )
    rank_sleeve_blend = _env_or_profile_float(
        "Q_RANK_SLEEVE_BLEND",
        "rank_sleeve_blend",
        0.0,
        0.0,
        0.60,
    )
    low_vol_sleeve_blend = _env_or_profile_float(
        "Q_LOW_VOL_SLEEVE_BLEND",
        "low_vol_sleeve_blend",
        0.0,
        0.0,
        0.35,
    )
    meta_execution_gate_strength = _env_or_profile_float(
        "Q_META_EXECUTION_GATE_STRENGTH",
        "meta_execution_gate_strength",
        1.0,
        0.0,
        2.0,
    )
    hit_gate_strength = _env_or_profile_float(
        "Q_HIT_GATE_STRENGTH",
        "hit_gate_strength",
        1.0,
        0.0,
        2.0,
    )
    hit_gate_threshold = _env_or_profile_float(
        "Q_HIT_GATE_THRESHOLD",
        "hit_gate_threshold",
        0.50,
        0.35,
        0.75,
    )
    hit_gate_floor = _env_or_profile_float(
        "Q_HIT_GATE_FLOOR",
        "hit_gate_floor",
        0.35,
        0.0,
        1.0,
    )
    hit_gate_ceiling = _env_or_profile_float(
        "Q_HIT_GATE_CEILING",
        "hit_gate_ceiling",
        1.05,
        0.5,
        1.5,
    )
    hit_gate_slope = _env_or_profile_float(
        "Q_HIT_GATE_SLOPE",
        "hit_gate_slope",
        14.0,
        1.0,
        40.0,
    )
    signal_deadzone = _env_or_profile_float(
        "Q_SIGNAL_DEADZONE",
        "signal_deadzone",
        0.0,
        0.0,
        0.10,
    )
    signal_deadzone_hit_sens = _env_or_profile_float(
        "Q_SIGNAL_DEADZONE_HIT_SENS",
        "signal_deadzone_hit_sens",
        1.0,
        0.0,
        3.0,
    )
    meta_reliability_strength = _env_or_profile_float(
        "Q_META_RELIABILITY_STRENGTH",
        "meta_reliability_strength",
        1.0,
        0.0,
        2.0,
    )
    meta_mix_leverage_strength = _env_or_profile_float(
        "Q_META_MIX_LEVERAGE_STRENGTH",
        "meta_mix_leverage_strength",
        1.0,
        0.0,
        2.0,
    )
    global_governor_strength = _env_or_profile_float(
        "Q_GLOBAL_GOVERNOR_STRENGTH",
        "global_governor_strength",
        1.0,
        0.0,
        2.0,
    )
    heartbeat_scaler_strength = _env_or_profile_float(
        "Q_HEARTBEAT_SCALER_STRENGTH",
        "heartbeat_scaler_strength",
        1.0,
        0.0,
        2.0,
    )
    quality_governor_strength = _env_or_profile_float(
        "Q_QUALITY_GOVERNOR_STRENGTH",
        "quality_governor_strength",
        1.0,
        0.0,
        2.0,
    )
    regime_moe_strength = _env_or_profile_float(
        "Q_REGIME_MOE_STRENGTH",
        "regime_moe_strength",
        1.0,
        0.0,
        2.0,
    )
    uncertainty_sizing_strength = _env_or_profile_float(
        "Q_UNCERTAINTY_SIZING_STRENGTH",
        "uncertainty_sizing_strength",
        1.0,
        0.0,
        2.0,
    )
    macro_proxy_strength = _env_or_profile_float(
        "Q_MACRO_PROXY_STRENGTH",
        "macro_proxy_strength",
        0.0,
        0.0,
        2.0,
    )
    capacity_impact_strength = _env_or_profile_float(
        "Q_CAPACITY_IMPACT_STRENGTH",
        "capacity_impact_strength",
        0.0,
        0.0,
        2.0,
    )
    credit_leadlag_strength = _env_or_profile_float(
        "Q_CREDIT_LEADLAG_STRENGTH",
        "credit_leadlag_strength",
        0.35,
        0.0,
        2.0,
    )
    microstructure_strength = _env_or_profile_float(
        "Q_MICROSTRUCTURE_STRENGTH",
        "microstructure_strength",
        0.20,
        0.0,
        2.0,
    )
    calendar_event_strength = _env_or_profile_float(
        "Q_CALENDAR_EVENT_STRENGTH",
        "calendar_event_strength",
        0.12,
        0.0,
        2.0,
    )
    vol_target_strength = _env_or_profile_float(
        "Q_VOL_TARGET_STRENGTH",
        "vol_target_strength",
        1.0,
        0.0,
        2.0,
    )
    vol_target_annual = _env_or_profile_float(
        "Q_VOL_TARGET_ANNUAL",
        "vol_target_annual",
        0.10,
        0.01,
        1.00,
    )
    vol_target_lookback = _env_or_profile_int(
        "Q_VOL_TARGET_LOOKBACK",
        "vol_target_lookback",
        63,
        5,
        756,
    )
    vol_target_min_scalar = _env_or_profile_float(
        "Q_VOL_TARGET_MIN_SCALAR",
        "vol_target_min_scalar",
        0.40,
        0.01,
        5.0,
    )
    vol_target_max_scalar = _env_or_profile_float(
        "Q_VOL_TARGET_MAX_SCALAR",
        "vol_target_max_scalar",
        1.80,
        0.01,
        5.0,
    )
    vol_target_smooth_alpha = _env_or_profile_float(
        "Q_VOL_TARGET_SMOOTH_ALPHA",
        "vol_target_smooth_alpha",
        0.20,
        0.0,
        1.0,
    )

    # 2) Cross-sectional rank sleeve blend.
    Wr = load_mat("runs_plus/weights_rank_sleeve.csv")
    if _gov_enabled("rank_sleeve_blend") and Wr is not None and Wr.shape[:2] == W.shape and rank_sleeve_blend > 0.0:
        b = float(np.clip(rank_sleeve_blend, 0.0, 0.60))
        W = (1.0 - b) * W + b * Wr
        steps.append("rank_sleeve_blend")
        _trace_put("rank_sleeve_blend", np.ones(T, float))

    # 2b) Low-vol anomaly sleeve blend.
    Wlv = load_mat("runs_plus/weights_low_vol_sleeve.csv")
    if _gov_enabled("low_vol_sleeve_blend") and Wlv is not None and Wlv.shape[:2] == W.shape and low_vol_sleeve_blend > 0.0:
        b = float(np.clip(low_vol_sleeve_blend, 0.0, 0.35))
        W = (1.0 - b) * W + b * Wlv
        steps.append("low_vol_sleeve_blend")
        _trace_put("low_vol_sleeve_blend", np.ones(T, float))

    # 3) Cluster caps
    Wc = load_mat("runs_plus/weights_cluster_capped.csv")
    if Wc is not None and Wc.shape[:2] == W.shape:
        W = Wc; steps.append("cluster_caps")
    # 4) Adaptive caps (per-time cap applied to weights)
    Wcap = load_mat("runs_plus/weights_capped.csv")
    if Wcap is not None and Wcap.shape[:2] == W.shape:
        W = Wcap; steps.append("adaptive_caps")

    # 5) Drawdown scaler (guardrails) -> weights_dd_scaled.csv is a full weight matrix
    Wdd = load_mat("runs_plus/weights_dd_scaled.csv")
    if Wdd is not None and Wdd.shape[:2] == W.shape:
        W = Wdd; steps.append("drawdown_floor")

    # 6) Turnover governor (guardrails)
    if _gov_enabled("turnover_governor"):
        Wtg, wtg_source = first_mat([
            "runs_plus/weights_turnover_budget_governed.csv",
            "runs_plus/weights_turnover_governed.csv",
        ])
        if Wtg is not None and Wtg.shape[:2] == W.shape:
            W = Wtg; steps.append("turnover_governor")
            steps.append(f"turnover_source={wtg_source}")
        else:
            Wauto, auto_tag, auto_scale = _auto_turnover_govern(W)
            if Wauto is not None and Wauto.shape[:2] == W.shape:
                W = Wauto
                steps.append("turnover_governor")
                steps.append(f"turnover_source={auto_tag}")
                _trace_put("turnover_governor", auto_scale)

    # 6) Meta execution gate (trade selectivity multiplier).
    meg = load_series("runs_plus/meta_execution_gate.csv")
    if _gov_enabled("meta_execution_gate") and meg is not None:
        L = min(len(meg), W.shape[0])
        mg_raw = np.clip(meg[:L], 0.0, 1.5)
        mg = _apply_governor_strength(mg_raw, meta_execution_gate_strength, lo=0.0, hi=2.0).reshape(-1, 1)
        W[:L] = W[:L] * mg
        steps.append("meta_execution_gate")
        _trace_put("meta_execution_gate", mg.ravel())

    # 6b) Hit-rate execution gate from calibrated confidence/probability.
    hit_proxy, hit_src = None, None
    for rel in [
        "runs_plus/meta_execution_prob.csv",
        "runs_plus/meta_mix_confidence_calibrated.csv",
        "runs_plus/meta_mix_confidence_raw.csv",
    ]:
        hp = load_series(rel)
        if hp is not None:
            hit_proxy, hit_src = hp, rel
            break
    if _gov_enabled("execution_hit_gate") and hit_proxy is not None and hit_gate_strength > 0.0:
        L = min(len(hit_proxy), W.shape[0])
        hg_raw = compute_hit_gate_scalars(
            hit_proxy[:L],
            threshold=hit_gate_threshold,
            floor=hit_gate_floor,
            ceiling=hit_gate_ceiling,
            slope=hit_gate_slope,
        )
        hg = _apply_governor_strength(hg_raw, hit_gate_strength, lo=0.0, hi=2.0).reshape(-1, 1)
        W[:L] = W[:L] * hg
        steps.append("execution_hit_gate")
        steps.append(f"execution_hit_source={hit_src}")
        _trace_put("execution_hit_gate", hg.ravel())

    # 7) Council disagreement gate (scalar per t) → scale exposure
    gate = load_series("runs_plus/disagreement_gate.csv")
    if _gov_enabled("council_gate") and gate is not None:
        L = min(len(gate), W.shape[0])
        g_raw = np.clip(gate[:L], 0.0, 1.0)
        g = _apply_governor_strength(g_raw, council_gate_strength, lo=0.0, hi=2.0).reshape(-1, 1)
        W[:L] = W[:L] * g
        steps.append("council_gate")
        _trace_put("council_gate", g.ravel())

    # 7) Council/meta leverage (scalar per t) from mix confidence
    lev = load_series("runs_plus/meta_mix_leverage.csv")
    if lev is None:
        mix = load_series("runs_plus/meta_mix.csv")
        if mix is not None:
            lev = np.clip(1.0 + 0.20 * np.abs(mix), 0.80, 1.30)
    if _gov_enabled("meta_mix_leverage") and lev is not None:
        L = min(len(lev), W.shape[0])
        lv_raw = np.clip(lev[:L], 0.70, 1.40)
        lv = _apply_governor_strength(lv_raw, meta_mix_leverage_strength, lo=0.0, hi=2.0).reshape(-1, 1)
        W[:L] = W[:L] * lv
        steps.append("meta_mix_leverage")
        _trace_put("meta_mix_leverage", lv.ravel())

    # 8) Meta/council reliability governor from confidence calibration.
    mrg = load_series("runs_plus/meta_mix_reliability_governor.csv")
    if _gov_enabled("meta_mix_reliability") and mrg is not None:
        L = min(len(mrg), W.shape[0])
        mr_raw = np.clip(mrg[:L], 0.70, 1.20)
        mr = _apply_governor_strength(mr_raw, meta_reliability_strength, lo=0.0, hi=2.0).reshape(-1, 1)
        W[:L] = W[:L] * mr
        steps.append("meta_mix_reliability")
        _trace_put("meta_mix_reliability", mr.ravel())

    # 9) Heartbeat exposure scaler (risk metabolism)
    hb = load_series("runs_plus/heartbeat_exposure_scaler.csv")
    if _gov_enabled("heartbeat_scaler") and hb is not None:
        L = min(len(hb), W.shape[0])
        hs_raw = np.clip(hb[:L], 0.40, 1.20)
        hs = _apply_governor_strength(hs_raw, heartbeat_scaler_strength, lo=0.0, hi=2.0).reshape(-1, 1)
        W[:L] = W[:L] * hs
        steps.append("heartbeat_scaler")
        _trace_put("heartbeat_scaler", hs.ravel())

    # 10) Legacy blended scaler from DNA/Heartbeat/Symbolic/Reflex tuner.
    lex = load_series("runs_plus/legacy_exposure.csv")
    if _gov_enabled("legacy_scaler") and lex is not None:
        L = min(len(lex), W.shape[0])
        ls = np.clip(lex[:L], 0.40, 1.30).reshape(-1, 1)
        W[:L] = W[:L] * ls
        steps.append("legacy_scaler")
        _trace_put("legacy_scaler", ls.ravel())

    # 11) DNA stress governor from drift/velocity regime diagnostics.
    dsg = load_series("runs_plus/dna_stress_governor.csv")
    if _gov_enabled("dna_stress_governor") and dsg is not None:
        L = min(len(dsg), W.shape[0])
        ds = np.clip(dsg[:L], 0.70, 1.15).reshape(-1, 1)
        W[:L] = W[:L] * ds
        steps.append("dna_stress_governor")
        _trace_put("dna_stress_governor", ds.ravel())

    # 12) Symbolic affective governor.
    sgg = load_series("runs_plus/symbolic_governor.csv")
    if _gov_enabled("symbolic_governor") and sgg is not None:
        L = min(len(sgg), W.shape[0])
        sg = np.clip(sgg[:L], 0.70, 1.15).reshape(-1, 1)
        W[:L] = W[:L] * sg
        steps.append("symbolic_governor")
        _trace_put("symbolic_governor", sg.ravel())

    # 13) Dream/reflex/symbolic coherence governor.
    dcg = load_series("runs_plus/dream_coherence_governor.csv")
    if _gov_enabled("dream_coherence") and dcg is not None:
        L = min(len(dcg), W.shape[0])
        ds = np.clip(dcg[:L], 0.70, 1.20).reshape(-1, 1)
        W[:L] = W[:L] * ds
        steps.append("dream_coherence")
        _trace_put("dream_coherence", ds.ravel())

    # 14) Reflex health governor from reflexive feedback diagnostics.
    rhg = load_series("runs_plus/reflex_health_governor.csv")
    if _gov_enabled("reflex_health_governor") and rhg is not None:
        L = min(len(rhg), W.shape[0])
        rs = np.clip(rhg[:L], 0.70, 1.15).reshape(-1, 1)
        W[:L] = W[:L] * rs
        steps.append("reflex_health_governor")
        _trace_put("reflex_health_governor", rs.ravel())

    # 15) Hive diversification governor from ecosystem layer.
    hg = load_series("runs_plus/hive_diversification_governor.csv")
    if _gov_enabled("hive_diversification") and hg is not None:
        L = min(len(hg), W.shape[0])
        hs = np.clip(hg[:L], 0.75, 1.08).reshape(-1, 1)
        W[:L] = W[:L] * hs
        steps.append("hive_diversification")
        _trace_put("hive_diversification", hs.ravel())

    # 16) Hive persistence governor from ecosystem action pressure.
    hpg = load_series("runs_plus/hive_persistence_governor.csv")
    if _gov_enabled("hive_persistence") and hpg is not None:
        L = min(len(hpg), W.shape[0])
        hp = np.clip(hpg[:L], 0.75, 1.06).reshape(-1, 1)
        W[:L] = W[:L] * hp
        steps.append("hive_persistence")
        _trace_put("hive_persistence", hp.ravel())

    # 17) Global governor (regime * stability) from guardrails.
    gg = load_series("runs_plus/global_governor.csv")
    if gg is None:
        rg = load_series("runs_plus/regime_governor.csv")
        sg = load_series("runs_plus/stability_governor.csv")
        if rg is not None and sg is not None:
            L = min(len(rg), len(sg))
            gg = np.clip(0.55 * rg[:L] + 0.45 * sg[:L], 0.45, 1.10)
        elif rg is not None:
            gg = np.clip(rg, 0.45, 1.10)
        elif sg is not None:
            gg = np.clip(sg, 0.45, 1.10)
    if _gov_enabled("global_governor") and gg is not None:
        L = min(len(gg), W.shape[0])
        g_raw = np.clip(gg[:L], 0.30, 1.15)
        g = _apply_governor_strength(g_raw, global_governor_strength, lo=0.0, hi=2.0).reshape(-1, 1)
        W[:L] = W[:L] * g
        steps.append("global_governor")
        _trace_put("global_governor", g.ravel())

    # 18) Reliability quality governor from nested/hive/council diagnostics.
    qg = load_series("runs_plus/quality_governor.csv")
    if _gov_enabled("quality_governor") and qg is not None:
        L = min(len(qg), W.shape[0])
        qs_raw = np.clip(qg[:L], 0.45, 1.20)
        qs = _apply_governor_strength(qs_raw, quality_governor_strength, lo=0.0, hi=2.0).reshape(-1, 1)
        W[:L] = W[:L] * qs
        steps.append("quality_governor")
        _trace_put("quality_governor", qs.ravel())

    # 19) NovaSpine recall-context boost (if available).
    rfg = load_series("runs_plus/regime_fracture_governor.csv")
    if _gov_enabled("regime_fracture_governor") and rfg is not None:
        L = min(len(rfg), W.shape[0])
        rf = np.clip(rfg[:L], 0.70, 1.06).reshape(-1, 1)
        W[:L] = W[:L] * rf
        steps.append("regime_fracture_governor")
        _trace_put("regime_fracture_governor", rf.ravel())

    # 20) Regime MoE governor (specialist blend scalar).
    rmg = load_series("runs_plus/regime_moe_governor.csv")
    if _gov_enabled("regime_moe_governor") and rmg is not None:
        L = min(len(rmg), W.shape[0])
        rm_raw = np.clip(rmg[:L], 0.50, 1.50)
        rm = _apply_governor_strength(rm_raw, regime_moe_strength, lo=0.0, hi=2.0).reshape(-1, 1)
        W[:L] = W[:L] * rm
        steps.append("regime_moe_governor")
        _trace_put("regime_moe_governor", rm.ravel())

    # 21) Uncertainty-aware sizing scalar.
    us = load_series("runs_plus/uncertainty_size_scalar.csv")
    if _gov_enabled("uncertainty_sizing") and us is not None:
        L = min(len(us), W.shape[0])
        ur_raw = np.clip(us[:L], 0.30, 1.30)
        ur = _apply_governor_strength(ur_raw, uncertainty_sizing_strength, lo=0.0, hi=2.0).reshape(-1, 1)
        W[:L] = W[:L] * ur
        steps.append("uncertainty_sizing")
        _trace_put("uncertainty_sizing", ur.ravel())

    # 22) Capacity/impact proxy guard from participation pressure.
    cis = load_series("runs_plus/capacity_impact_scalar.csv")
    if _gov_enabled("capacity_impact_guard") and cis is not None:
        L = min(len(cis), W.shape[0])
        ci_raw = np.clip(cis[:L], 0.60, 1.10)
        ci = _apply_governor_strength(ci_raw, capacity_impact_strength, lo=0.0, hi=2.0).reshape(-1, 1)
        W[:L] = W[:L] * ci
        steps.append("capacity_impact_guard")
        _trace_put("capacity_impact_guard", ci.ravel())

    # 23) Credit lead/lag overlay (credit divergence as predictive alpha).
    clo = load_series("runs_plus/credit_leadlag_overlay.csv")
    if _gov_enabled("credit_leadlag_overlay") and clo is not None:
        L = min(len(clo), W.shape[0])
        cl_raw = np.clip(clo[:L], 0.70, 1.30)
        cl = _apply_governor_strength(cl_raw, credit_leadlag_strength, lo=0.0, hi=2.0).reshape(-1, 1)
        W[:L] = W[:L] * cl
        steps.append("credit_leadlag_overlay")
        _trace_put("credit_leadlag_overlay", cl.ravel())

    # 24) Microstructure proxy overlay (liquidity + close-location pressure).
    mso = load_series("runs_plus/microstructure_overlay.csv")
    if _gov_enabled("microstructure_overlay") and mso is not None:
        L = min(len(mso), W.shape[0])
        ms_raw = np.clip(mso[:L], 0.70, 1.30)
        ms = _apply_governor_strength(ms_raw, microstructure_strength, lo=0.0, hi=2.0).reshape(-1, 1)
        W[:L] = W[:L] * ms
        steps.append("microstructure_overlay")
        _trace_put("microstructure_overlay", ms.ravel())

    # 25) Calendar/event overlay.
    ceo = load_series("runs_plus/calendar_event_overlay.csv")
    if _gov_enabled("calendar_event_overlay") and ceo is not None:
        L = min(len(ceo), W.shape[0])
        ce_raw = np.clip(ceo[:L], 0.70, 1.30)
        ce = _apply_governor_strength(ce_raw, calendar_event_strength, lo=0.0, hi=2.0).reshape(-1, 1)
        W[:L] = W[:L] * ce
        steps.append("calendar_event_overlay")
        _trace_put("calendar_event_overlay", ce.ravel())

    # 26) Macro proxy guard (forward-looking stress proxies).
    mps = load_series("runs_plus/macro_risk_scalar.csv")
    if _gov_enabled("macro_proxy_guard") and mps is not None:
        L = min(len(mps), W.shape[0])
        mp_raw = np.clip(mps[:L], 0.70, 1.10)
        mp = _apply_governor_strength(mp_raw, macro_proxy_strength, lo=0.0, hi=2.0).reshape(-1, 1)
        W[:L] = W[:L] * mp
        steps.append("macro_proxy_guard")
        _trace_put("macro_proxy_guard", mp.ravel())

    # 27) NovaSpine recall-context boost (if available).
    ncb = load_series("runs_plus/novaspine_context_boost.csv")
    if _gov_enabled("novaspine_context_boost") and ncb is not None:
        L = min(len(ncb), W.shape[0])
        nb = np.clip(ncb[:L], 0.85, 1.15).reshape(-1, 1)
        W[:L] = W[:L] * nb
        steps.append("novaspine_context_boost")
        _trace_put("novaspine_context_boost", nb.ravel())

    # 28) NovaSpine per-hive alignment boost (global projection).
    nhb = load_series("runs_plus/novaspine_hive_boost.csv")
    if _gov_enabled("novaspine_hive_boost") and nhb is not None:
        L = min(len(nhb), W.shape[0])
        hb = np.clip(nhb[:L], 0.85, 1.15).reshape(-1, 1)
        W[:L] = W[:L] * hb
        steps.append("novaspine_hive_boost")
        _trace_put("novaspine_hive_boost", hb.ravel())

    # 29) Shock/news mask exposure cut.
    shock_alpha = _env_or_profile_float("Q_SHOCK_ALPHA", "shock_alpha", 0.35, 0.0, 1.0)
    sm = load_series("runs_plus/shock_mask.csv")
    if _gov_enabled("shock_mask_guard") and sm is not None:
        L = min(len(sm), W.shape[0])
        sc = (1.0 - shock_alpha * np.clip(sm[:L], 0.0, 1.0)).reshape(-1, 1)
        W[:L] = W[:L] * sc
        steps.append("shock_mask_guard")
        _trace_put("shock_mask_guard", sc.ravel())

    # 30) Explicit portfolio volatility targeting (global scalar).
    if _gov_enabled("vol_target") and vol_target_strength > 0.0:
        Aret = load_mat("runs_plus/asset_returns.csv")
        if Aret is not None and Aret.shape[1] == W.shape[1]:
            scalars_raw = compute_vol_target_scalars(
                W,
                Aret,
                target_annual_vol=vol_target_annual,
                lookback=vol_target_lookback,
                min_scalar=vol_target_min_scalar,
                max_scalar=vol_target_max_scalar,
                smooth_alpha=vol_target_smooth_alpha,
            )
            L = min(len(scalars_raw), W.shape[0])
            sv = _apply_governor_strength(scalars_raw[:L], vol_target_strength, lo=0.0, hi=3.0).reshape(-1, 1)
            W[:L] = W[:L] * sv
            steps.append("vol_target")
            _trace_put("vol_target", sv.ravel())
            (RUNS / "vol_target_info.json").write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "target_annual_vol": float(vol_target_annual),
                        "lookback": int(vol_target_lookback),
                        "min_scalar": float(vol_target_min_scalar),
                        "max_scalar": float(vol_target_max_scalar),
                        "smooth_alpha": float(vol_target_smooth_alpha),
                        "strength": float(vol_target_strength),
                        "scalar_mean": float(np.mean(sv)),
                        "scalar_min": float(np.min(sv)),
                        "scalar_max": float(np.max(sv)),
                    },
                    indent=2,
                )
            )

    # 31) Runtime floor guard: avoid excessive exposure collapse from stacked governors.
    runtime_floor = _runtime_total_floor_default()
    if _gov_enabled("runtime_floor") and runtime_floor > 0.0:
        active_keys = [k for k in TRACE_STEPS if (k in trace and k != "runtime_floor")]
        if active_keys:
            trace_mat_pre = np.column_stack([trace[k] for k in active_keys])
            trace_total_pre = np.prod(trace_mat_pre, axis=1)
            adj = np.ones(T, dtype=float)
            mask = trace_total_pre < runtime_floor
            if np.any(mask):
                adj[mask] = runtime_floor / np.maximum(trace_total_pre[mask], 1e-9)
                W = W * adj.reshape(-1, 1)
                steps.append("runtime_floor")
            _trace_put("runtime_floor", adj)

    # 32) Concentration governor (top1/top3 + HHI caps).
    use_conc = _env_or_profile_bool("Q_USE_CONCENTRATION_GOV", "use_concentration_governor", True)
    conc_top1 = _env_or_profile_float("Q_CONCENTRATION_TOP1_CAP", "concentration_top1_cap", 0.18, 0.01, 1.0)
    conc_top3 = _env_or_profile_float("Q_CONCENTRATION_TOP3_CAP", "concentration_top3_cap", 0.42, 0.01, 1.0)
    conc_hhi = _env_or_profile_float("Q_CONCENTRATION_MAX_HHI", "concentration_max_hhi", 0.14, 0.01, 1.0)
    if use_conc:
        top1 = conc_top1
        top3 = conc_top3
        hhi = conc_hhi
        W, cstats = govern_matrix(W, top1_cap=top1, top3_cap=top3, max_hhi=hhi)
        steps.append("concentration_governor")
        (RUNS / "concentration_governor_info.json").write_text(
            json.dumps(
                {
                    "enabled": True,
                    "top1_cap": top1,
                    "top3_cap": top3,
                    "max_hhi": hhi,
                    "stats": cstats,
                },
                indent=2,
            )
        )

    # 33) Signal deadzone filter to drop micro-positions/noise.
    deadzone_info = {
        "enabled": False,
        "base_deadzone": float(signal_deadzone),
        "active_before": int(np.count_nonzero(np.abs(W) > 0.0)),
        "active_after": int(np.count_nonzero(np.abs(W) > 0.0)),
        "pruned_fraction": 0.0,
    }
    if _gov_enabled("signal_deadzone") and signal_deadzone > 0.0:
        W, deadzone_info = apply_signal_deadzone(
            W,
            base_deadzone=signal_deadzone,
            hit_proxy=hit_proxy,
            hit_threshold=hit_gate_threshold,
            hit_sensitivity=signal_deadzone_hit_sens,
        )
        steps.append("signal_deadzone")
    (RUNS / "signal_deadzone_info.json").write_text(json.dumps(deadzone_info, indent=2))

    # Build governor trace artifact for auditability.
    for name in TRACE_STEPS:
        if name not in trace:
            _trace_put(name, None)
    trace_mat = np.column_stack([trace[name] for name in TRACE_STEPS])
    trace_total = np.prod(trace_mat, axis=1)
    trace_out = np.column_stack([trace_mat, trace_total])
    trace_cols = TRACE_STEPS + ["runtime_total_scalar"]
    np.savetxt(
        RUNS / "final_governor_trace.csv",
        trace_out,
        delimiter=",",
        header=",".join(trace_cols),
        comments="",
    )

    # 34) Save final
    outp = RUNS/"portfolio_weights_final.csv"
    np.savetxt(outp, W, delimiter=",")

    # 35) Small JSON breadcrumb
    (RUNS/"final_portfolio_info.json").write_text(
        json.dumps(
            {
                "steps": steps,
                "T": int(T),
                "N": int(N),
                "disabled_governors": sorted(list(_DISABLED_GOVS)),
                "runtime_total_floor_target": _runtime_total_floor_default(),
                "governor_params_profile_file": str(RUNS / "governor_params_profile.json"),
                "governor_params_applied": {
                    "runtime_total_floor": _runtime_total_floor_default(),
                    "shock_alpha": shock_alpha,
                    "rank_sleeve_blend": rank_sleeve_blend,
                    "low_vol_sleeve_blend": low_vol_sleeve_blend,
                    "meta_execution_gate_strength": meta_execution_gate_strength,
                    "hit_gate_strength": hit_gate_strength,
                    "hit_gate_threshold": hit_gate_threshold,
                    "hit_gate_floor": hit_gate_floor,
                    "hit_gate_ceiling": hit_gate_ceiling,
                    "hit_gate_slope": hit_gate_slope,
                    "signal_deadzone": signal_deadzone,
                    "signal_deadzone_hit_sens": signal_deadzone_hit_sens,
                    "council_gate_strength": council_gate_strength,
                    "meta_mix_leverage_strength": meta_mix_leverage_strength,
                    "meta_reliability_strength": meta_reliability_strength,
                    "global_governor_strength": global_governor_strength,
                    "heartbeat_scaler_strength": heartbeat_scaler_strength,
                    "quality_governor_strength": quality_governor_strength,
                    "regime_moe_strength": regime_moe_strength,
                    "uncertainty_sizing_strength": uncertainty_sizing_strength,
                    "capacity_impact_strength": capacity_impact_strength,
                    "credit_leadlag_strength": credit_leadlag_strength,
                    "microstructure_strength": microstructure_strength,
                    "calendar_event_strength": calendar_event_strength,
                    "macro_proxy_strength": macro_proxy_strength,
                    "vol_target_strength": vol_target_strength,
                    "vol_target_annual": vol_target_annual,
                    "vol_target_lookback": vol_target_lookback,
                    "vol_target_min_scalar": vol_target_min_scalar,
                    "vol_target_max_scalar": vol_target_max_scalar,
                    "vol_target_smooth_alpha": vol_target_smooth_alpha,
                    "use_concentration_governor": bool(use_conc),
                    "concentration_top1_cap": conc_top1,
                    "concentration_top3_cap": conc_top3,
                    "concentration_max_hhi": conc_hhi,
                    "auto_turnover_max_step": _env_or_profile_float(
                        "Q_AUTO_TURNOVER_MAX_STEP",
                        "auto_turnover_max_step",
                        0.30,
                        0.01,
                        5.0,
                    ),
                    "auto_turnover_budget_window": _env_or_profile_int(
                        "Q_AUTO_TURNOVER_BUDGET_WINDOW",
                        "auto_turnover_budget_window",
                        5,
                        1,
                        120,
                    ),
                    "auto_turnover_budget_limit": _env_or_profile_float(
                        "Q_AUTO_TURNOVER_BUDGET_LIMIT",
                        "auto_turnover_budget_limit",
                        1.00,
                        0.01,
                        20.0,
                    ),
                },
                "governor_trace_file": str(RUNS / "final_governor_trace.csv"),
                "runtime_total_scalar_mean": float(np.mean(trace_total)),
                "runtime_total_scalar_min": float(np.min(trace_total)),
                "runtime_total_scalar_max": float(np.max(trace_total)),
                "signal_deadzone_info_file": str(RUNS / "signal_deadzone_info.json"),
            },
            indent=2,
        )
    )

    # 36) Report card
    html = f"<p>Built <b>portfolio_weights_final.csv</b> (T={T}, N={N}). Steps: {', '.join(steps)}.</p>"
    append_card("Final Portfolio ✔", html)

    print(f"✅ Wrote {outp}  | Steps: {', '.join(steps)}")
