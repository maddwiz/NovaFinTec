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


if __name__ == "__main__":
    nested = _load_json(RUNS / "nested_wf_summary.json") or {}
    health = _load_json(RUNS / "system_health.json") or {}
    meta = _load_json(RUNS / "meta_stack_summary.json") or {}
    syn = _load_json(RUNS / "synapses_summary.json") or {}
    mix = _load_json(RUNS / "meta_mix_info.json") or {}
    nctx = _load_json(RUNS / "novaspine_context.json") or {}
    eco = _load_json(RUNS / "hive_evolution.json") or {}
    shock_info = _load_json(RUNS / "shock_mask_info.json") or {}
    dream_info = _load_json(RUNS / "dream_coherence_info.json") or {}

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

    # Blend across subsystems, using whatever is available.
    quality, quality_detail = blend_quality(
        {
            "nested_wf": (nested_q, 0.30),
            "hive_wf": (hive_q, 0.23),
            "council": (council_q, 0.20),
            "dream_coherence": (dream_q, 0.10),
            "system_health": (health_q, 0.13),
            "ecosystem": (eco_q, 0.07),
            "shock_env": (shock_q, 0.05),
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
        lo=0.55,
        hi=1.15,
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

    qg = np.clip(qg * runtime_mod, 0.55, 1.15)
    if len(qg) > 1:
        for t in range(1, len(qg)):
            qg[t] = 0.86 * qg[t - 1] + 0.14 * qg[t]
        qg = np.clip(qg, 0.55, 1.15)

    np.savetxt(RUNS / "quality_governor.csv", qg, delimiter=",")
    np.savetxt(RUNS / "quality_runtime_modifier.csv", runtime_mod, delimiter=",")

    snapshot = {
        "quality_score": float(quality),
        "quality_governor_mean": float(np.mean(qg)) if len(qg) else None,
        "quality_governor_min": float(np.min(qg)) if len(qg) else None,
        "quality_governor_max": float(np.max(qg)) if len(qg) else None,
        "length": int(T),
        "components": {
            "nested_wf": {"score": float(nested_q), "detail": nested_detail},
            "hive_wf": {"score": float(hive_q), "detail": hive_detail},
            "council": {"score": float(council_q), "detail": council_detail},
            "dream_coherence": {"score": float(dream_q) if dream_q is not None else None},
            "ecosystem": {"score": float(eco_q) if eco_q is not None else None},
            "shock_env": {"score": float(shock_q) if shock_q is not None else None},
            "system_health": {"score": float(health_q)},
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
            "meta_mix_reliability_governor": (RUNS / "meta_mix_reliability_governor.csv").exists(),
            "shock_mask_info": (RUNS / "shock_mask_info.json").exists(),
            "shock_mask": (RUNS / "shock_mask.csv").exists(),
            "dream_coherence_info": (RUNS / "dream_coherence_info.json").exists(),
            "dream_coherence_governor": (RUNS / "dream_coherence_governor.csv").exists(),
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
