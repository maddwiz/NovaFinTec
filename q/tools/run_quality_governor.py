#!/usr/bin/env python3
# Build reliability-aware exposure governor from nested WF / hive WF / council diagnostics.
#
# Writes:
#   runs_plus/quality_governor.csv
#   runs_plus/quality_snapshot.json

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmods.quality_governor import (  # noqa: E402
    blend_quality,
    build_governor_series,
    drawdown_quality,
    hit_quality,
    sharpe_quality,
)

RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _load_series(path: Path):
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
    if a.ndim == 2 and a.shape[1] >= 1:
        a = a[:, -1]
    return a.ravel()


def _load_row_mean_series(path: Path):
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    cols = [c for c in df.columns if str(c).lower() not in {"date", "timestamp", "time"}]
    if not cols:
        return None
    vals = []
    for c in cols:
        vals.append(pd.to_numeric(df[c], errors="coerce").fillna(0.0).values.astype(float))
    if not vals:
        return None
    mat = np.column_stack(vals)
    return np.nan_to_num(np.mean(mat, axis=1), nan=0.0, posinf=0.0, neginf=0.0)


def _infer_length():
    # Keep priority aligned with final portfolio flow.
    for rel in [
        "portfolio_weights_final.csv",
        "weights_regime.csv",
        "weights_tail_blend.csv",
        "portfolio_weights.csv",
    ]:
        p = RUNS / rel
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
        if a.ndim == 1:
            return int(len(a))
        if a.ndim == 2:
            return int(a.shape[0])
    return 0


def _append_card(title, html):
    for name in ["report_all.html", "report_best_plus.html", "report_plus.html", "report.html"]:
        p = ROOT / name
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        txt = txt.replace("</body>", card + "</body>") if "</body>" in txt else txt + card
        p.write_text(txt, encoding="utf-8")


def _execution_constraint_quality(exec_info: dict | None, health_issues: list[str] | None = None):
    """
    Convert execution-throttle telemetry into a [0,1] quality score.
    Lower score means constraints are likely either over-throttling (no tradability)
    or increasing turnover unexpectedly (instability).
    """
    if not isinstance(exec_info, dict):
        return None, {}

    def _f(k):
        try:
            v = float(exec_info.get(k, np.nan))
            return v if np.isfinite(v) else None
        except Exception:
            return None

    g0 = _f("gross_before_mean")
    g1 = _f("gross_after_mean")
    t0 = _f("turnover_before_mean")
    t1 = _f("turnover_after_mean")
    tmax = _f("turnover_after_max")
    step_cap = _f("max_step_turnover")

    gross_ret = None
    if g0 is not None and g0 > 1e-9 and g1 is not None:
        gross_ret = float(np.clip(g1 / g0, 0.0, 3.0))
    turn_ret = None
    if t0 is not None and t0 > 1e-9 and t1 is not None:
        turn_ret = float(np.clip(t1 / t0, 0.0, 3.0))

    q_gross = None
    if gross_ret is not None:
        q_gross = float(np.clip((gross_ret - 0.05) / (0.85 - 0.05 + 1e-9), 0.0, 1.0))
    q_turn = None
    if turn_ret is not None:
        q_turn = float(np.clip((turn_ret - 0.03) / (0.70 - 0.03 + 1e-9), 0.0, 1.0))
        if turn_ret > 1.10:
            q_turn *= 0.60

    parts = [x for x in [q_turn, q_gross] if x is not None]
    if not parts:
        return None, {
            "gross_retention": gross_ret,
            "turnover_retention": turn_ret,
            "step_cap": step_cap,
            "turnover_after_max": tmax,
        }

    if q_turn is not None and q_gross is not None:
        q = float(np.clip(0.55 * q_turn + 0.45 * q_gross, 0.0, 1.0))
    else:
        q = float(parts[0])

    if step_cap is not None and tmax is not None and tmax > step_cap + 1e-6:
        q *= 0.80
    if turn_ret is not None and turn_ret > 1.10:
        overload = float(np.clip((turn_ret - 1.10) / (1.80 - 1.10 + 1e-9), 0.0, 1.0))
        q *= float(np.clip(1.0 - 0.35 * overload, 0.55, 1.0))

    h_issues = [str(x).lower() for x in (health_issues or [])]
    if any("over-throttling turnover" in x for x in h_issues):
        q *= 0.75
    if any("collapsed gross exposure" in x for x in h_issues):
        q *= 0.75

    q = float(np.clip(q, 0.0, 1.0))
    detail = {
        "gross_retention": gross_ret,
        "turnover_retention": turn_ret,
        "step_cap": step_cap,
        "turnover_after_max": tmax,
        "q_gross": q_gross,
        "q_turnover": q_turn,
    }
    return q, detail


def _cap_step_change(series: np.ndarray, max_step: float | None, lo: float, hi: float):
    s = np.asarray(series, float).copy()
    if s.size <= 1:
        return np.clip(s, lo, hi)
    m = None if max_step is None else float(max_step)
    if m is None or (not np.isfinite(m)) or m <= 0.0:
        return np.clip(s, lo, hi)
    m = float(np.clip(m, 1e-6, 1.0))
    for t in range(1, len(s)):
        d = float(s[t] - s[t - 1])
        if d > m:
            s[t] = s[t - 1] + m
        elif d < -m:
            s[t] = s[t - 1] - m
    return np.clip(s, lo, hi)


if __name__ == "__main__":
    nested = _load_json(RUNS / "nested_wf_summary.json") or {}
    health = _load_json(RUNS / "system_health.json") or {}
    meta = _load_json(RUNS / "meta_stack_summary.json") or {}
    syn = _load_json(RUNS / "synapses_summary.json") or {}
    mix = _load_json(RUNS / "meta_mix_info.json") or {}
    nctx = _load_json(RUNS / "novaspine_context.json") or {}
    eco = _load_json(RUNS / "hive_evolution.json") or {}
    shock_info = _load_json(RUNS / "shock_mask_info.json") or {}
    fracture_info = _load_json(RUNS / "regime_fracture_info.json") or {}
    dream_info = _load_json(RUNS / "dream_coherence_info.json") or {}
    dna_info = _load_json(RUNS / "dna_stress_info.json") or {}
    reflex_info = _load_json(RUNS / "reflex_health_info.json") or {}
    sym_info = _load_json(RUNS / "symbolic_governor_info.json") or {}
    cross_info = _load_json(RUNS / "cross_hive_summary.json") or {}
    exec_info = _load_json(RUNS / "execution_constraints_info.json") or {}

    hive_sh = None
    hive_hit = None
    hive_dd = None
    hm = RUNS / "hive_wf_metrics.csv"
    if hm.exists():
        try:
            m = pd.read_csv(hm)
            if len(m):
                if "sharpe_oos" in m.columns:
                    hive_sh = float(pd.to_numeric(m["sharpe_oos"], errors="coerce").mean())
                if "hit_rate" in m.columns:
                    hive_hit = float(pd.to_numeric(m["hit_rate"], errors="coerce").mean())
                if "max_dd" in m.columns:
                    hive_dd = float(pd.to_numeric(m["max_dd"], errors="coerce").mean())
        except Exception:
            pass

    nested_sh = nested.get("avg_oos_sharpe", None)
    nested_hit = nested.get("avg_hit", None)
    nested_dd = nested.get("avg_oos_maxDD", None)

    # Council quality from both meta and synapse summaries plus realized mix stats.
    council_sh = np.nanmean(
        [
            float(meta.get("oos_like_sharpe", np.nan)),
            float(mix.get("oos_like_sharpe", np.nan)),
        ]
    )
    council_hit = np.nanmean(
        [
            float(meta.get("oos_like_hit_rate", np.nan)),
            float(mix.get("hit_rate", np.nan)),
        ]
    )
    council_conf = np.nanmean(
        [
            float(meta.get("mean_confidence", np.nan)),
            float(syn.get("mean_confidence", np.nan)),
            float(mix.get("mean_confidence", np.nan)),
        ]
    )
    syn_disp_mean = float(syn.get("mean_dispersion", np.nan))
    if np.isfinite(syn_disp_mean):
        syn_disp_q = float(np.clip(1.0 - syn_disp_mean / 0.35, 0.0, 1.0))
    else:
        syn_disp_q = None
    meta_cal_q = None
    try:
        mcal = float(mix.get("mean_confidence_calibrated", np.nan))
        if np.isfinite(mcal):
            meta_cal_q = float(np.clip(mcal, 0.0, 1.0))
    except Exception:
        meta_cal_q = None

    eco_q = None
    try:
        counts = eco.get("event_counts", {}) if isinstance(eco, dict) else {}
        total_actions = float(
            float(counts.get("atrophy_applied", 0))
            + float(counts.get("split_applied", 0))
            + float(counts.get("fusion_applied", 0))
        )
        Teco = max(1.0, float(_infer_length()))
        # Higher action density means more ecosystem instability.
        action_density = total_actions / (3.0 * Teco)
        eco_q = float(np.clip(1.0 - action_density / 0.50, 0.0, 1.0))
    except Exception:
        eco_q = None
    persistence_q = None
    try:
        pg = _load_series(RUNS / "hive_persistence_governor.csv")
        if pg is not None and len(pg):
            persistence_q = float(np.clip((float(np.mean(pg)) - 0.78) / (1.04 - 0.78 + 1e-9), 0.0, 1.0))
    except Exception:
        persistence_q = None

    nested_q, nested_detail = blend_quality(
        {
            "nested_sharpe": (sharpe_quality(nested_sh), 0.55),
            "nested_hit": (hit_quality(nested_hit), 0.20),
            "nested_dd": (drawdown_quality(nested_dd), 0.25),
        }
    )
    hive_q, hive_detail = blend_quality(
        {
            "hive_sharpe": (sharpe_quality(hive_sh), 0.55),
            "hive_hit": (hit_quality(hive_hit), 0.20),
            "hive_dd": (drawdown_quality(hive_dd), 0.25),
        }
    )
    council_q, council_detail = blend_quality(
        {
            "council_sharpe": (sharpe_quality(council_sh), 0.45),
            "council_hit": (hit_quality(council_hit), 0.25),
            "council_conf": (float(np.clip(council_conf, 0.0, 1.0)) if np.isfinite(council_conf) else None, 0.30),
            "synapses_agreement": (syn_disp_q, 0.10),
            "meta_calibration": (meta_cal_q, 0.15),
        }
    )
    health_q = float(np.clip(float(health.get("health_score", 50.0)) / 100.0, 0.0, 1.0))
    exec_q, exec_detail = _execution_constraint_quality(exec_info, health.get("issues", []) if isinstance(health, dict) else [])
    nctx_q = None
    if isinstance(nctx, dict):
        try:
            if str(nctx.get("status", "")).lower() == "ok":
                nctx_q = float(np.clip(float(nctx.get("context_resonance", 0.5)), 0.0, 1.0))
        except Exception:
            nctx_q = None

    shock_q = None
    try:
        sr = float(shock_info.get("shock_rate", np.nan))
        if np.isfinite(sr):
            shock_q = float(np.clip(1.0 - sr / 0.30, 0.0, 1.0))
    except Exception:
        shock_q = None

    dream_q = None
    try:
        dream_q = float(dream_info.get("mean_coherence", np.nan))
        if np.isfinite(dream_q):
            dream_q = float(np.clip(dream_q, 0.0, 1.0))
        else:
            dream_q = None
    except Exception:
        dream_q = None
    dna_q = None
    try:
        ms = float(dna_info.get("mean_stress", np.nan))
        if np.isfinite(ms):
            dna_q = float(np.clip(1.0 - ms, 0.0, 1.0))
    except Exception:
        dna_q = None
    reflex_q = None
    try:
        rm = float(reflex_info.get("governor_mean", np.nan))
        rh = float(reflex_info.get("health_mean", np.nan))
        q_rm = float(np.clip((rm - 0.72) / (1.10 - 0.72 + 1e-9), 0.0, 1.0)) if np.isfinite(rm) else np.nan
        q_rh = float(np.clip(rh / 1.50, 0.0, 1.0)) if np.isfinite(rh) else np.nan
        if np.isfinite(q_rm) and np.isfinite(q_rh):
            reflex_q = float(np.clip(0.55 * q_rm + 0.45 * q_rh, 0.0, 1.0))
        elif np.isfinite(q_rm):
            reflex_q = q_rm
        elif np.isfinite(q_rh):
            reflex_q = q_rh
    except Exception:
        reflex_q = None
    sym_q = None
    try:
        ms = float(sym_info.get("mean_stress", np.nan))
        if np.isfinite(ms):
            sym_q = float(np.clip(1.0 - ms, 0.0, 1.0))
    except Exception:
        sym_q = None
    hive_downside_q = None
    try:
        dn = cross_info.get("downside_penalty_mean", {})
        if isinstance(dn, dict) and dn:
            vals = [float(v) for v in dn.values() if np.isfinite(float(v))]
            if vals:
                hive_downside_q = float(np.clip(1.0 - float(np.mean(vals)), 0.0, 1.0))
    except Exception:
        hive_downside_q = None
    reflex_downside_q = None
    try:
        d = float(reflex_info.get("downside_penalty_mean", np.nan))
        if np.isfinite(d):
            reflex_downside_q = float(np.clip(1.0 - d, 0.0, 1.0))
    except Exception:
        reflex_downside_q = None
    dna_downside_q = None
    try:
        d = float(dna_info.get("mean_downside_stress", np.nan))
        if np.isfinite(d):
            dna_downside_q = float(np.clip(1.0 - d, 0.0, 1.0))
    except Exception:
        dna_downside_q = None
    fracture_q = None
    try:
        fs = float(fracture_info.get("latest_score", np.nan))
        if np.isfinite(fs):
            fracture_q = float(np.clip(1.0 - fs, 0.0, 1.0))
    except Exception:
        fracture_q = None

    # Blend across subsystems, using whatever is available.
    quality, quality_detail = blend_quality(
        {
            "nested_wf": (nested_q, 0.30),
            "hive_wf": (hive_q, 0.23),
            "council": (council_q, 0.20),
            "dream_coherence": (dream_q, 0.10),
            "dna_stress": (dna_q, 0.08),
            "reflex_health": (reflex_q, 0.06),
            "symbolic": (sym_q, 0.08),
            "hive_downside": (hive_downside_q, 0.10),
            "reflex_downside": (reflex_downside_q, 0.08),
            "dna_downside": (dna_downside_q, 0.06),
            "system_health": (health_q, 0.13),
            "execution_constraints": (exec_q, 0.12),
            "ecosystem": (eco_q, 0.07),
            "ecosystem_persistence": (persistence_q, 0.06),
            "shock_env": (shock_q, 0.05),
            "regime_fracture": (fracture_q, 0.09),
            "novaspine_context": (nctx_q, 0.12),
        }
    )

    T = _infer_length()
    if T <= 0:
        print("(!) No weight matrix found to infer quality governor length; skipping.")
        raise SystemExit(0)

    gate = _load_series(RUNS / "disagreement_gate.csv")
    global_gov = _load_series(RUNS / "global_governor.csv")
    qg = build_governor_series(
        length=T,
        base_quality=quality,
        disagreement_gate=gate,
        global_governor=global_gov,
        lo=float(np.clip(float(os.getenv("Q_QUALITY_GOV_LO", "0.58")), 0.30, 1.20)),
        hi=float(np.clip(float(os.getenv("Q_QUALITY_GOV_HI", "1.15")), 0.50, 1.30)),
        smooth=0.85,
    )

    runtime_mod = np.ones(T, dtype=float)
    # Synapses ensemble disagreement modifier (lower confidence when members diverge).
    syn_disp = _load_series(RUNS / "synapses_ensemble_dispersion.csv")
    if syn_disp is not None and len(syn_disp):
        L = min(T, len(syn_disp))
        d = np.abs(np.asarray(syn_disp[:L], float))
        p90 = float(np.percentile(d, 90)) if len(d) else 0.0
        dn = np.clip(d / (p90 + 1e-9), 0.0, 2.0)
        runtime_mod[:L] *= np.clip(1.06 - 0.18 * dn, 0.82, 1.06)

    # Cross-hive adaptive control modifier from alpha/inertia diagnostics, if available.
    cwh = RUNS / "cross_hive_weights.csv"
    if cwh.exists():
        try:
            cw = pd.read_csv(cwh)
            if {"arb_alpha", "arb_inertia"}.issubset(cw.columns):
                aa = pd.to_numeric(cw["arb_alpha"], errors="coerce").ffill().fillna(2.2).values
                ii = pd.to_numeric(cw["arb_inertia"], errors="coerce").ffill().fillna(0.80).values
                L = min(T, len(aa), len(ii))
                aa_n = np.clip((aa[:L] - 0.8) / (4.5 - 0.8 + 1e-9), 0.0, 1.0)
                ii_n = np.clip((ii[:L] - 0.40) / (0.97 - 0.40 + 1e-9), 0.0, 1.0)
                stress = 0.5 * (1.0 - aa_n) + 0.5 * ii_n
                runtime_mod[:L] *= np.clip(1.03 - 0.20 * stress, 0.84, 1.03)
        except Exception:
            pass

    # Shock mask modifier.
    sm = _load_series(RUNS / "shock_mask.csv")
    if sm is not None and len(sm):
        L = min(T, len(sm))
        alpha = float(np.clip(float(os.getenv("Q_SHOCK_ALPHA", "0.35")), 0.0, 1.0))
        runtime_mod[:L] *= np.clip(1.0 - 0.80 * alpha * np.clip(sm[:L], 0.0, 1.0), 0.70, 1.0)

    # Meta/council reliability governor from confidence calibration.
    mrg = _load_series(RUNS / "meta_mix_reliability_governor.csv")
    if mrg is not None and len(mrg):
        L = min(T, len(mrg))
        mm = np.clip(np.asarray(mrg[:L], float), 0.70, 1.20)
        runtime_mod[:L] *= np.clip(mm, 0.75, 1.15)

    # Dream coherence modifier from dream/reflex/symbolic consistency.
    dcg = _load_series(RUNS / "dream_coherence_governor.csv")
    if dcg is not None and len(dcg):
        L = min(T, len(dcg))
        dn = np.clip((np.asarray(dcg[:L], float) - 0.70) / (1.15 - 0.70 + 1e-9), 0.0, 1.0)
        runtime_mod[:L] *= np.clip(0.88 + 0.24 * dn, 0.78, 1.12)

    # DNA stress governor modifier.
    dsg = _load_series(RUNS / "dna_stress_governor.csv")
    if dsg is not None and len(dsg):
        L = min(T, len(dsg))
        ds = np.clip(np.asarray(dsg[:L], float), 0.70, 1.15)
        runtime_mod[:L] *= np.clip(ds, 0.75, 1.12)

    # Reflex health governor modifier.
    rhg = _load_series(RUNS / "reflex_health_governor.csv")
    if rhg is not None and len(rhg):
        L = min(T, len(rhg))
        rs = np.clip(np.asarray(rhg[:L], float), 0.70, 1.15)
        runtime_mod[:L] *= np.clip(rs, 0.75, 1.12)

    # Symbolic governor modifier.
    sgg = _load_series(RUNS / "symbolic_governor.csv")
    if sgg is not None and len(sgg):
        L = min(T, len(sgg))
        sg = np.clip(np.asarray(sgg[:L], float), 0.70, 1.15)
        runtime_mod[:L] *= np.clip(sg, 0.75, 1.12)

    # Ecosystem persistence governor modifier.
    hpg = _load_series(RUNS / "hive_persistence_governor.csv")
    if hpg is not None and len(hpg):
        L = min(T, len(hpg))
        hp = np.clip(np.asarray(hpg[:L], float), 0.75, 1.06)
        runtime_mod[:L] *= np.clip(hp, 0.80, 1.08)

    # Hive downside penalty modifier (mean across hives).
    hdp = _load_row_mean_series(RUNS / "hive_downside_penalty.csv")
    if hdp is not None and len(hdp):
        L = min(T, len(hdp))
        dn = np.clip(np.asarray(hdp[:L], float), 0.0, 1.0)
        runtime_mod[:L] *= np.clip(1.02 - 0.22 * dn, 0.78, 1.02)

    # Reflex downside penalty modifier.
    rcomp = RUNS / "reflex_health_components.csv"
    if rcomp.exists():
        try:
            rdf = pd.read_csv(rcomp)
            if "downside_penalty" in rdf.columns:
                rd = pd.to_numeric(rdf["downside_penalty"], errors="coerce").fillna(0.0).values.astype(float)
                L = min(T, len(rd))
                dn = np.clip(rd[:L], 0.0, 1.0)
                runtime_mod[:L] *= np.clip(1.02 - 0.16 * dn, 0.82, 1.02)
        except Exception:
            pass

    # Execution constraints quality modifier.
    if exec_q is not None:
        runtime_mod *= float(np.clip(0.82 + 0.35 * exec_q, 0.70, 1.10))

    q_lo = float(np.clip(float(os.getenv("Q_QUALITY_GOV_LO", "0.58")), 0.30, 1.20))
    q_hi = float(np.clip(float(os.getenv("Q_QUALITY_GOV_HI", "1.15")), 0.50, 1.30))

    qg = np.clip(qg * runtime_mod, q_lo, q_hi)
    if len(qg) > 1:
        for t in range(1, len(qg)):
            qg[t] = 0.86 * qg[t - 1] + 0.14 * qg[t]
        qg = np.clip(qg, q_lo, q_hi)
    max_step = float(np.clip(float(os.getenv("Q_QUALITY_GOV_MAX_STEP", "0.06")), 0.005, 0.50))
    qg = _cap_step_change(qg, max_step=max_step, lo=q_lo, hi=q_hi)
    step_abs_max = float(np.max(np.abs(np.diff(qg)))) if len(qg) > 1 else 0.0
    step_abs_mean = float(np.mean(np.abs(np.diff(qg)))) if len(qg) > 1 else 0.0

    np.savetxt(RUNS / "quality_governor.csv", qg, delimiter=",")
    np.savetxt(RUNS / "quality_runtime_modifier.csv", runtime_mod, delimiter=",")

    snapshot = {
        "quality_score": float(quality),
        "quality_governor_mean": float(np.mean(qg)) if len(qg) else None,
        "quality_governor_min": float(np.min(qg)) if len(qg) else None,
        "quality_governor_max": float(np.max(qg)) if len(qg) else None,
        "quality_governor_max_abs_step": step_abs_max,
        "quality_governor_mean_abs_step": step_abs_mean,
        "quality_governor_max_step_cfg": max_step,
        "length": int(T),
        "components": {
            "nested_wf": {"score": float(nested_q), "detail": nested_detail},
            "hive_wf": {"score": float(hive_q), "detail": hive_detail},
            "council": {"score": float(council_q), "detail": council_detail},
            "dream_coherence": {"score": float(dream_q) if dream_q is not None else None},
            "dna_stress": {"score": float(dna_q) if dna_q is not None else None},
            "reflex_health": {"score": float(reflex_q) if reflex_q is not None else None},
            "symbolic": {"score": float(sym_q) if sym_q is not None else None},
            "hive_downside": {"score": float(hive_downside_q) if hive_downside_q is not None else None},
            "reflex_downside": {"score": float(reflex_downside_q) if reflex_downside_q is not None else None},
            "dna_downside": {"score": float(dna_downside_q) if dna_downside_q is not None else None},
            "ecosystem": {"score": float(eco_q) if eco_q is not None else None},
            "ecosystem_persistence": {"score": float(persistence_q) if persistence_q is not None else None},
            "shock_env": {"score": float(shock_q) if shock_q is not None else None},
            "regime_fracture": {"score": float(fracture_q) if fracture_q is not None else None},
            "system_health": {"score": float(health_q)},
            "execution_constraints": {"score": float(exec_q) if exec_q is not None else None, "detail": exec_detail},
            "novaspine_context": {"score": float(nctx_q) if nctx_q is not None else None},
        },
        "blend_detail": quality_detail,
        "sources": {
            "nested_wf_summary": (RUNS / "nested_wf_summary.json").exists(),
            "hive_wf_metrics": hm.exists(),
            "meta_stack_summary": (RUNS / "meta_stack_summary.json").exists(),
            "synapses_summary": (RUNS / "synapses_summary.json").exists(),
            "synapses_ensemble_dispersion": (RUNS / "synapses_ensemble_dispersion.csv").exists(),
            "meta_mix_info": (RUNS / "meta_mix_info.json").exists(),
            "hive_evolution": (RUNS / "hive_evolution.json").exists(),
            "hive_downside_penalty": (RUNS / "hive_downside_penalty.csv").exists(),
            "meta_mix_reliability_governor": (RUNS / "meta_mix_reliability_governor.csv").exists(),
            "shock_mask_info": (RUNS / "shock_mask_info.json").exists(),
            "shock_mask": (RUNS / "shock_mask.csv").exists(),
            "regime_fracture_info": (RUNS / "regime_fracture_info.json").exists(),
            "dream_coherence_info": (RUNS / "dream_coherence_info.json").exists(),
            "dream_coherence_governor": (RUNS / "dream_coherence_governor.csv").exists(),
            "dna_stress_info": (RUNS / "dna_stress_info.json").exists(),
            "dna_stress_governor": (RUNS / "dna_stress_governor.csv").exists(),
            "reflex_health_info": (RUNS / "reflex_health_info.json").exists(),
            "reflex_health_components": (RUNS / "reflex_health_components.csv").exists(),
            "reflex_health_governor": (RUNS / "reflex_health_governor.csv").exists(),
            "symbolic_governor_info": (RUNS / "symbolic_governor_info.json").exists(),
            "symbolic_governor": (RUNS / "symbolic_governor.csv").exists(),
            "hive_persistence_governor": (RUNS / "hive_persistence_governor.csv").exists(),
            "execution_constraints_info": (RUNS / "execution_constraints_info.json").exists(),
            "system_health": (RUNS / "system_health.json").exists(),
            "novaspine_context": (RUNS / "novaspine_context.json").exists(),
        },
    }
    (RUNS / "quality_snapshot.json").write_text(json.dumps(snapshot, indent=2))

    _append_card(
        "Quality Governor ✔",
        (
            f"<p>quality={quality:.3f}, scaler mean={snapshot['quality_governor_mean']:.3f}, "
            f"min={snapshot['quality_governor_min']:.3f}, max={snapshot['quality_governor_max']:.3f}</p>"
        ),
    )
    print(f"✅ Wrote {RUNS/'quality_governor.csv'}")
    print(f"✅ Wrote {RUNS/'quality_snapshot.json'}")
