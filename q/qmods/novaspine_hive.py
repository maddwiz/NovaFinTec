from __future__ import annotations

import numpy as np


def clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def build_hive_query(hive: str, sharpe: float | None, hit: float | None, max_dd: float | None) -> str:
    hs = str(hive or "UNKNOWN").upper()
    sh = float(sharpe) if sharpe is not None else 0.0
    hr = float(hit) if hit is not None else 0.5
    dd = float(max_dd) if max_dd is not None else 0.0
    return (
        f"Recall high-confidence memories for hive {hs}. "
        f"Recent metrics: sharpe={sh:.3f}, hit={hr:.3f}, max_dd={dd:.3f}. "
        "Return the most relevant risk lessons and winning patterns for current regime alignment."
    )


def hive_resonance(retrieved_count: int | float, top_k: int | float, scores: list[float] | None) -> float:
    c = max(0.0, float(retrieved_count or 0.0))
    k = max(1.0, float(top_k or 1.0))
    coverage = clip01(c / k)
    if scores:
        s = [clip01(float(x)) for x in scores if np.isfinite(float(x))]
        score_mean = float(np.mean(s)) if s else 0.0
    else:
        score_mean = 0.0
    return clip01(0.60 * coverage + 0.40 * score_mean)


def hive_boost(resonance: float, status_ok: bool = True) -> float:
    if not bool(status_ok):
        return 1.0
    r = clip01(resonance)
    # Keep bounds tight; this is a prior, not an override.
    return float(np.clip(0.90 + 0.20 * r, 0.85, 1.10))
