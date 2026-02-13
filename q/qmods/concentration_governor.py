from __future__ import annotations

import numpy as np


def _renorm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, None)
    s = float(np.sum(x))
    if s <= 1e-12:
        n = len(x)
        return np.full(n, 1.0 / max(1, n), dtype=float)
    return x / s


def _blend_to_hhi_target(e: np.ndarray, max_hhi: float) -> np.ndarray:
    e = _renorm(e)
    n = len(e)
    if n <= 1:
        return e
    hhi = float(np.sum(e * e))
    tgt = float(np.clip(max_hhi, 1.0 / n, 1.0))
    if hhi <= tgt + 1e-12:
        return e
    u = np.full(n, 1.0 / n, dtype=float)
    lo, hi = 0.0, 1.0
    best = e.copy()
    for _ in range(30):
        m = 0.5 * (lo + hi)
        z = _renorm((1.0 - m) * e + m * u)
        hz = float(np.sum(z * z))
        if hz <= tgt:
            best = z
            hi = m
        else:
            lo = m
    return best


def _cap_topk(e: np.ndarray, k: int, cap_sum: float) -> np.ndarray:
    e = _renorm(e)
    n = len(e)
    k = int(max(1, min(k, n)))
    cap_sum = float(np.clip(cap_sum, 1e-6, 1.0))
    idx = np.argsort(-e)
    top = idx[:k]
    cur = float(np.sum(e[top]))
    if cur <= cap_sum + 1e-12:
        return e
    # Reduce top-k proportionally and redistribute to remainder.
    scale = cap_sum / (cur + 1e-12)
    out = e.copy()
    out[top] = out[top] * scale
    excess = float(np.sum(e[top]) - np.sum(out[top]))
    rest = idx[k:]
    if len(rest) == 0:
        return _renorm(out)
    rest_w = np.clip(out[rest], 0.0, None)
    sw = float(np.sum(rest_w))
    if sw <= 1e-12:
        out[rest] += excess / len(rest)
    else:
        out[rest] += excess * (rest_w / sw)
    return _renorm(out)


def govern_row(
    w: np.ndarray,
    top1_cap: float = 0.18,
    top3_cap: float = 0.42,
    max_hhi: float = 0.14,
) -> tuple[np.ndarray, dict]:
    v = np.asarray(w, float).ravel()
    n = len(v)
    if n == 0:
        return v, {"hhi_before": 0.0, "hhi_after": 0.0}
    gross = float(np.sum(np.abs(v)))
    if gross <= 1e-12:
        return v.copy(), {"hhi_before": 0.0, "hhi_after": 0.0}

    sgn = np.sign(v)
    e = np.abs(v) / gross
    hhi0 = float(np.sum(e * e))

    e = _cap_topk(e, k=1, cap_sum=float(np.clip(top1_cap, 0.01, 1.0)))
    e = _cap_topk(e, k=min(3, n), cap_sum=float(np.clip(top3_cap, 0.01, 1.0)))
    e = _blend_to_hhi_target(e, max_hhi=max_hhi)
    # Re-apply hard top caps after HHI blending.
    e = _cap_topk(e, k=1, cap_sum=float(np.clip(top1_cap, 0.01, 1.0)))
    e = _cap_topk(e, k=min(3, n), cap_sum=float(np.clip(top3_cap, 0.01, 1.0)))

    out = sgn * e * gross
    hhi1 = float(np.sum(e * e))
    info = {
        "hhi_before": hhi0,
        "hhi_after": hhi1,
        "top1_before": float(np.max(np.abs(v)) / (gross + 1e-12)),
        "top1_after": float(np.max(np.abs(out)) / (gross + 1e-12)),
        "top3_before": float(np.sum(np.sort(np.abs(v))[-min(3, n) :]) / (gross + 1e-12)),
        "top3_after": float(np.sum(np.sort(np.abs(out))[-min(3, n) :]) / (gross + 1e-12)),
    }
    return out, info


def govern_matrix(
    W: np.ndarray,
    top1_cap: float = 0.18,
    top3_cap: float = 0.42,
    max_hhi: float = 0.14,
) -> tuple[np.ndarray, dict]:
    A = np.asarray(W, float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    out = np.zeros_like(A)
    infos = []
    for t in range(A.shape[0]):
        rr, inf = govern_row(A[t], top1_cap=top1_cap, top3_cap=top3_cap, max_hhi=max_hhi)
        out[t] = rr
        infos.append(inf)
    if infos:
        keys = infos[0].keys()
        stats = {k: float(np.mean([d[k] for d in infos])) for k in keys}
        stats.update({f"{k}_max": float(np.max([d[k] for d in infos])) for k in keys if k.endswith("after")})
    else:
        stats = {}
    return out, stats
