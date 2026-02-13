#!/usr/bin/env python3
# Build reliability-aware exposure governor from nested WF / hive WF / council diagnostics.
#
# Writes:
#   runs_plus/quality_governor.csv
#   runs_plus/quality_snapshot.json

from __future__ import annotations

import json
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

    # Blend across subsystems, using whatever is available.
    quality, quality_detail = blend_quality(
        {
            "nested_wf": (nested_q, 0.30),
            "hive_wf": (hive_q, 0.23),
            "council": (council_q, 0.22),
            "system_health": (health_q, 0.13),
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
    np.savetxt(RUNS / "quality_governor.csv", qg, delimiter=",")

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
            "system_health": {"score": float(health_q)},
            "novaspine_context": {"score": float(nctx_q) if nctx_q is not None else None},
        },
        "blend_detail": quality_detail,
        "sources": {
            "nested_wf_summary": (RUNS / "nested_wf_summary.json").exists(),
            "hive_wf_metrics": hm.exists(),
            "meta_stack_summary": (RUNS / "meta_stack_summary.json").exists(),
            "synapses_summary": (RUNS / "synapses_summary.json").exists(),
            "meta_mix_info": (RUNS / "meta_mix_info.json").exists(),
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
