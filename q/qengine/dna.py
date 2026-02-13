import numpy as np
from numpy.fft import rfft


def zscore(x, eps=1e-8):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x) if x.size else 0.0
    sd = np.nanstd(x) if x.size else 0.0
    return (x - mu) / (sd + eps)


def dna_from_window(x, topk=32):
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if len(x) < topk * 2:
        x = np.pad(x, (0, topk * 2 - len(x)), mode="edge")
    x = zscore(x)
    spec = np.abs(rfft(x))
    if spec.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    k = int(max(1, min(topk, spec.size)))
    idx = np.argpartition(spec, -k)[-k:]
    vals = spec[idx] / (spec.max() + 1e-8)
    return idx.astype(int), vals.astype(float)


def _to_dense(idx, vals):
    idx = np.asarray(idx, dtype=int).ravel()
    vals = np.asarray(vals, dtype=float).ravel()
    if idx.size == 0:
        return {}
    return {int(i): float(v) for i, v in zip(idx, vals)}


def cosine_1_minus(a_idx, a_vals, b_idx, b_vals, eps=1e-12):
    a = _to_dense(a_idx, a_vals)
    b = _to_dense(b_idx, b_vals)
    keys = sorted(set(a.keys()) | set(b.keys()))
    if not keys:
        return 0.0
    va = np.array([a.get(k, 0.0) for k in keys], dtype=float)
    vb = np.array([b.get(k, 0.0) for k in keys], dtype=float)
    num = float(np.dot(va, vb))
    den = float(np.linalg.norm(va) * np.linalg.norm(vb) + eps)
    cos = np.clip(num / den, -1.0, 1.0)
    return float(1.0 - cos)


def rolling_drift(close, window=64, topk=32, smooth_span=5):
    x = np.asarray(close, dtype=float)
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    prev = None
    w = int(max(8, window))
    for i in range(w, n):
        idx, vals = dna_from_window(x[i - w : i], topk=topk)
        if prev is not None:
            out[i] = cosine_1_minus(prev[0], prev[1], idx, vals)
        prev = (idx, vals)
    if smooth_span and smooth_span > 1 and np.isfinite(out).any():
        alpha = 2.0 / (float(smooth_span) + 1.0)
        sm = out.copy()
        first = int(np.argmax(np.isfinite(sm)))
        for i in range(first + 1, len(sm)):
            if np.isfinite(sm[i - 1]) and np.isfinite(sm[i]):
                sm[i] = alpha * sm[i] + (1.0 - alpha) * sm[i - 1]
        out = sm
    return out


def drift_velocity(drift):
    d = np.asarray(drift, dtype=float)
    return np.gradient(np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0))


def drift_regime_flags(drift, z_win=63, hi=1.25, lo=-1.25):
    d = np.asarray(drift, dtype=float)
    z = np.full_like(d, np.nan, dtype=float)
    w = int(max(10, z_win))
    for i in range(w, len(d)):
        s = d[i - w : i]
        mu = np.nanmean(s)
        sd = np.nanstd(s) + 1e-12
        z[i] = (d[i] - mu) / sd
    state = np.where(z >= hi, 1, np.where(z <= lo, -1, 0))
    return z, state
