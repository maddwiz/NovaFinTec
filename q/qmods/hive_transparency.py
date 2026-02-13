from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def _to_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def build_hive_snapshot(
    hive_names: Iterable[str],
    metrics_by_hive: Dict[str, dict] | None = None,
    latest_weights: Dict[str, float] | None = None,
    feedback_by_hive: Dict[str, dict] | None = None,
) -> dict:
    metrics = metrics_by_hive or {}
    weights = latest_weights or {}
    feedback = feedback_by_hive or {}

    seen = []
    for h in hive_names:
        hs = str(h).strip().upper()
        if hs and hs not in seen:
            seen.append(hs)
    for src in (metrics.keys(), weights.keys(), feedback.keys()):
        for h in src:
            hs = str(h).strip().upper()
            if hs and hs not in seen:
                seen.append(hs)

    rows: List[dict] = []
    for h in seen:
        mm = metrics.get(h, {}) if isinstance(metrics.get(h, {}), dict) else {}
        ff = feedback.get(h, {}) if isinstance(feedback.get(h, {}), dict) else {}
        row = {
            "hive": h,
            "weight": _to_float(weights.get(h, 0.0), 0.0),
            "sharpe_oos": _to_float(mm.get("sharpe_oos", 0.0), 0.0),
            "hit_rate": _to_float(mm.get("hit_rate", 0.0), 0.0),
            "max_dd": _to_float(mm.get("max_dd", 0.0), 0.0),
            "novaspine_boost": _to_float(ff.get("boost", 1.0), 1.0),
            "novaspine_resonance": _to_float(ff.get("resonance", 0.0), 0.0),
            "novaspine_status": str(ff.get("status", "na")),
        }
        rows.append(row)

    rows.sort(key=lambda r: abs(float(r["weight"])), reverse=True)

    w = np.asarray([float(r["weight"]) for r in rows], float) if rows else np.asarray([0.0], float)
    sh = np.asarray([float(r["sharpe_oos"]) for r in rows], float) if rows else np.asarray([0.0], float)
    hb = np.asarray([float(r["novaspine_boost"]) for r in rows], float) if rows else np.asarray([1.0], float)

    summary = {
        "hive_count": int(len(rows)),
        "top_hive": str(rows[0]["hive"]) if rows else None,
        "top_weight": float(rows[0]["weight"]) if rows else 0.0,
        "weight_l1": float(np.sum(np.abs(w))),
        "weight_hhi": float(np.sum(np.square(np.abs(w)))) if rows else 0.0,
        "mean_sharpe_oos": float(np.mean(sh)) if rows else 0.0,
        "mean_novaspine_boost": float(np.mean(hb)) if rows else 1.0,
    }

    return {"rows": rows, "summary": summary}
