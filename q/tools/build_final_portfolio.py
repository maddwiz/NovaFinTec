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
    "rank_sleeve_blend",
    "turnover_governor",
    "meta_execution_gate",
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
    "novaspine_context_boost",
    "novaspine_hive_boost",
    "shock_mask_guard",
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
    raw = str(os.getenv("Q_DISABLE_GOVERNORS", "")).strip()
    out = set()
    if raw:
        for token in raw.split(","):
            t = str(token).strip().lower()
            if t:
                out.add(t)
        return out
    vals = _GOV_PROFILE.get("disable_governors", [])
    if isinstance(vals, list):
        for token in vals:
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

def append_card(title, html):
    for name in ["report_all.html","report_best_plus.html","report_plus.html","report.html"]:
        f = ROOT/name
        if not f.exists(): continue
        txt = f.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        f.write_text(txt.replace("</body>", card+"</body>") if "</body>" in txt else txt+card, encoding="utf-8")

if __name__ == "__main__":
    # 1) Base weights preference
    W, source = first_mat([
        "runs_plus/weights_regime.csv",
        "runs_plus/weights_tail_blend.csv",
        "runs_plus/portfolio_weights.csv",
        "portfolio_weights.csv",
    ])
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
    meta_execution_gate_strength = _env_or_profile_float(
        "Q_META_EXECUTION_GATE_STRENGTH",
        "meta_execution_gate_strength",
        1.0,
        0.0,
        2.0,
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

    # 2) Cross-sectional rank sleeve blend.
    Wr = load_mat("runs_plus/weights_rank_sleeve.csv")
    if _gov_enabled("rank_sleeve_blend") and Wr is not None and Wr.shape[:2] == W.shape and rank_sleeve_blend > 0.0:
        b = float(np.clip(rank_sleeve_blend, 0.0, 0.60))
        W = (1.0 - b) * W + b * Wr
        steps.append("rank_sleeve_blend")
        _trace_put("rank_sleeve_blend", np.ones(T, float))

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

    # 22) NovaSpine recall-context boost (if available).
    ncb = load_series("runs_plus/novaspine_context_boost.csv")
    if _gov_enabled("novaspine_context_boost") and ncb is not None:
        L = min(len(ncb), W.shape[0])
        nb = np.clip(ncb[:L], 0.85, 1.15).reshape(-1, 1)
        W[:L] = W[:L] * nb
        steps.append("novaspine_context_boost")
        _trace_put("novaspine_context_boost", nb.ravel())

    # 23) NovaSpine per-hive alignment boost (global projection).
    nhb = load_series("runs_plus/novaspine_hive_boost.csv")
    if _gov_enabled("novaspine_hive_boost") and nhb is not None:
        L = min(len(nhb), W.shape[0])
        hb = np.clip(nhb[:L], 0.85, 1.15).reshape(-1, 1)
        W[:L] = W[:L] * hb
        steps.append("novaspine_hive_boost")
        _trace_put("novaspine_hive_boost", hb.ravel())

    # 24) Shock/news mask exposure cut.
    shock_alpha = _env_or_profile_float("Q_SHOCK_ALPHA", "shock_alpha", 0.35, 0.0, 1.0)
    sm = load_series("runs_plus/shock_mask.csv")
    if _gov_enabled("shock_mask_guard") and sm is not None:
        L = min(len(sm), W.shape[0])
        sc = (1.0 - shock_alpha * np.clip(sm[:L], 0.0, 1.0)).reshape(-1, 1)
        W[:L] = W[:L] * sc
        steps.append("shock_mask_guard")
        _trace_put("shock_mask_guard", sc.ravel())

    # 25) Runtime floor guard: avoid excessive exposure collapse from stacked governors.
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

    # 26) Concentration governor (top1/top3 + HHI caps).
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

    # 27) Save final
    outp = RUNS/"portfolio_weights_final.csv"
    np.savetxt(outp, W, delimiter=",")

    # 28) Small JSON breadcrumb
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
                    "meta_execution_gate_strength": meta_execution_gate_strength,
                    "council_gate_strength": council_gate_strength,
                    "meta_mix_leverage_strength": meta_mix_leverage_strength,
                    "meta_reliability_strength": meta_reliability_strength,
                    "global_governor_strength": global_governor_strength,
                    "heartbeat_scaler_strength": heartbeat_scaler_strength,
                    "quality_governor_strength": quality_governor_strength,
                    "regime_moe_strength": regime_moe_strength,
                    "uncertainty_sizing_strength": uncertainty_sizing_strength,
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
            },
            indent=2,
        )
    )

    # 29) Report card
    html = f"<p>Built <b>portfolio_weights_final.csv</b> (T={T}, N={N}). Steps: {', '.join(steps)}.</p>"
    append_card("Final Portfolio ✔", html)

    print(f"✅ Wrote {outp}  | Steps: {', '.join(steps)}")
