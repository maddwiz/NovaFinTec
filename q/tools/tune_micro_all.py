#!/usr/bin/env python3
# Micro tuner for Phases 1–3 (guardrails + hive brain + refinements).
# Safe & fast: ~176 combos by default. Handles empty/short inputs gracefully.
#
# Reads (uses what's available, skips safely):
# - runs_plus/meta_stack_pred.csv, runs_plus/synapses_pred.csv
# - runs_plus/daily_returns.csv   (portfolio pnl proxy)
# - runs_plus/weights_tail_blend.csv  (Risk-On base)  or portfolio_weights*.csv
# - runs_plus/risk_parity_weights.csv (Defensive base) or weights_capped.csv
# - runs_plus/cluster_map.csv (optional)
# - runs_plus/shock_mask.csv (optional) or builds from returns
# - runs_plus/asset_returns.csv (optional) or builds from data/*.csv (Adj Close/Close)
#
# Writes:
# - runs_plus/tune_best_weights.csv
# - runs_plus/tune_best_config.json
# - Appends "Micro Tuning (P1–P3) ✔" card to report
# - Prints terminal summary

import json, csv, math
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"; RUNS.mkdir(exist_ok=True)
DATA = ROOT / "data"

np.set_printoptions(suppress=True, linewidth=140)

# ---------- IO helpers ----------
def load_series(rel):
    p = ROOT/rel
    if not p.exists(): return None
    try:    return np.loadtxt(p, delimiter=",").ravel()
    except: return np.loadtxt(p, delimiter=",", skiprows=1).ravel()

def load_matrix(rel):
    p = ROOT/rel
    if not p.exists(): return None
    try:
        a = np.loadtxt(p, delimiter=",")
    except:
        a = np.loadtxt(p, delimiter=",", skiprows=1)
    if a.ndim == 1: a = a.reshape(-1,1)
    return a

def first_series(paths):
    for rel in paths:
        a = load_series(rel)
        if a is not None: return a
    return None

def first_matrix(paths):
    for rel in paths:
        a = load_matrix(rel)
        if a is not None: return a
    return None

def append_card(title, html):
    for name in ["report_all.html","report_best_plus.html","report_plus.html","report.html"]:
        f = ROOT/name
        if not f.exists(): continue
        txt = f.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        f.write_text(txt.replace("</body>", card+"</body>") if "</body>" in txt else txt+card, encoding="utf-8")

# ---------- Stats ----------
def sharpe(r):
    r = np.asarray(r, float)
    if r.size == 0 or np.all(~np.isfinite(r)): return -1e9
    r = r[np.isfinite(r)]
    if r.size == 0: return -1e9
    mu = np.nanmean(r); sd = np.nanstd(r) + 1e-12
    return float((mu/sd)*np.sqrt(252.0))

def zscore(x):
    x = np.asarray(x, float)
    mu = np.nanmean(x); sd = np.nanstd(x) + 1e-12
    return (x - mu) / sd

# ---------- Build asset return matrix (fallback) ----------
def build_asset_returns():
    A = first_matrix(["runs_plus/asset_returns.csv"])
    if A is not None: return A
    series = []
    for fp in sorted(DATA.glob("*.csv")):
        try:
            with fp.open() as f:
                rdr = csv.DictReader(f)
                closes = [row.get("Adj Close") or row.get("Close") or row.get("close") for row in rdr]
                closes = [float(x) for x in closes if x not in (None, "", "NA")]
            c = np.asarray(closes, float)
            if len(c) > 5:
                r = np.diff(c) / (c[:-1] + 1e-12)
                series.append(r)
        except:
            pass
    if not series: 
        return None
    T = min(len(s) for s in series)
    if T < 10: 
        return None
    series = [s[-T:] for s in series]
    return np.stack(series, axis=1)

# ---------- Helper transforms ----------
def adaptive_cap_from_vol(vol, cap_min, cap_max):
    v = np.asarray(vol, float)
    if v.size == 0: return v
    z = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
    u = 1.0 / (1.0 + np.exp(-z))
    return cap_min + (cap_max - cap_min) * (1.0 - u)  # higher vol -> smaller cap

def apply_caps(W, caps):
    W = np.asarray(W, float).copy()
    C = np.asarray(caps, float).ravel()
    T = min(W.shape[0], len(C))
    if T == 0: return W
    for t in range(T):
        cap = float(C[t])
        W[t] = np.clip(W[t], -cap, cap)
    return W

def cap_by_cluster(W, clusters, cap=0.20):
    Wc = W.copy()
    labels = np.array(clusters, dtype=str)
    uniq = sorted(set(labels.tolist()))
    for t in range(Wc.shape[0]):
        for g in uniq:
            idx = np.where(labels == g)[0]
            if idx.size == 0: continue
            tot = float(np.sum(np.abs(Wc[t, idx])))
            if tot > cap + 1e-12:
                scale = cap / (tot + 1e-12)
                Wc[t, idx] *= scale
    return Wc

def regime_from_vol(r, lb=63, q=0.5):
    r = np.asarray(r, float)
    T = len(r)
    lb = int(max(2, lb))
    vol = np.full(T, np.nan)
    # rolling std (simple loop, robust)
    for i in range(T):
        j = max(0, i - lb + 1)
        w = r[j:i+1]
        if w.size >= 2:
            vol[i] = np.nanstd(w)
        else:
            vol[i] = np.nan
    # fill initial NaNs with first valid
    first_valid = np.nanmin(np.where(np.isfinite(vol), np.arange(T), np.inf))
    if math.isfinite(first_valid):
        vol[:int(first_valid)] = vol[int(first_valid)]
    v_ok = vol[np.isfinite(vol)]
    thr = np.nanquantile(v_ok, q) if v_ok.size else np.nan
    reg = np.ones(T, float)
    if math.isfinite(thr):
        reg = (vol <= thr).astype(float)
    # smooth
    lbeta = 0.2
    sm = np.zeros_like(reg)
    for i, x in enumerate(reg):
        sm[i] = (1-lbeta)*(sm[i-1] if i>0 else x) + lbeta*x
    return np.clip(sm, 0, 1), (thr if math.isfinite(thr) else 0.0)

# ---------- Main ----------
if __name__ == "__main__":
    # 0) Inputs
    y = first_series(["runs_plus/daily_returns.csv","daily_returns.csv"])
    A = build_asset_returns()

    W_on = first_matrix([
        "runs_plus/weights_tail_blend.csv",
        "runs_plus/portfolio_weights.csv",
        "portfolio_weights.csv",
    ])
    W_def = first_matrix([
        "runs_plus/risk_parity_weights.csv",
        "runs_plus/weights_capped.csv"
    ])

    if y is None and (A is None or W_on is None or W_def is None):
        print("(!) Need returns or asset/weight matrices; run your pipeline first.")
        raise SystemExit(0)

    # Align lengths
    lengths = []
    for arr in [y, A, W_on, W_def]:
        if arr is None: continue
        lengths.append(arr.shape[0] if hasattr(arr, "shape") else len(arr))
    T = min(lengths) if lengths else None

    def trim(o):
        if o is None: return None
        if hasattr(o, "shape"):
            return o[:T] if T is not None and o.shape[0] >= T else o
        return o[:T] if T is not None and len(o) >= T else o

    y = trim(y); A = trim(A); W_on = trim(W_on); W_def = trim(W_def)

    # Council predictions (optional)
    m = load_series("runs_plus/meta_stack_pred.csv"); m = trim(m)
    s = load_series("runs_plus/synapses_pred.csv");  s = trim(s)

    # Shock mask (or build from returns proxy)
    mask = load_series("runs_plus/shock_mask.csv"); mask = trim(mask)
    if mask is None:
        base_r = y
        if base_r is None and (A is not None and W_on is not None and A.shape[1] == W_on.shape[1]):
            base_r = np.sum(A[:W_on.shape[0]] * W_on[:A.shape[0]], axis=1)
        if base_r is not None:
            v = np.abs(base_r)
            z = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
            raw = (np.abs(z) > 2.5).astype(int)
            m2 = np.zeros_like(raw)
            streak = 0
            for i, r2 in enumerate(raw):
                streak = streak + 1 if r2 else 0
                m2[i] = 1 if streak >= 2 else 0
            mask = trim(m2)

    # Cluster labels
    clusters = None
    p = RUNS/"cluster_map.csv"
    if p.exists():
        try:
            rows = [ln.strip().split(",") for ln in p.read_text().strip().splitlines()]
            col = [r[1] for r in rows[1:]] if len(rows) > 1 else [r[0] for r in rows]
            clusters = np.array(col, dtype=str)
        except:
            clusters = None

    # --------- Compact grids (fast & meaningful) ----------
    grid = {
        "alpha_meta": np.linspace(0.0, 1.0, 11),        # council mix
        "reg_lb":     np.array([42, 63]),               # regime lookback
        "reg_q":      np.array([0.45, 0.55]),           # vol threshold quantile
        "cap_cluster":np.array([0.15, 0.20]),           # per-cluster cap
        "cap_minmax": [(0.05,0.15)],                    # adaptive cap band
        "shock_alpha":np.array([0.5]),                  # mask strength
        "bags":       np.array([10, 20]),               # time bags
        "bag_size":   np.array([0.7]),                  # bag window size
    }

    # --------- Scoring primitive ----------
    def score_combo(params):
        # 1) Council mix
        mix_sig = None
        if (m is not None) and (s is not None) and (len(m) == len(s)):
            Zm = zscore(m); Zs = zscore(s)
            a = params["alpha_meta"]
            base_mix = a*Zm + (1-a)*Zs

            # Bagging by time
            Tloc = len(base_mix)
            if params["bags"] > 1:
                L = max(1, int(Tloc * params["bag_size"]))
                if L >= Tloc: L = max(1, Tloc - 1)
                rng = np.random.default_rng(42)
                mats = []
                for _ in range(int(params["bags"])):
                    if Tloc - L <= 0:
                        start = 0
                    else:
                        start = int(rng.integers(0, Tloc - L + 1))
                    end = start + L
                    pvec = np.full(Tloc, np.nan)
                    pvec[start:end] = base_mix[start:end]
                    mats.append(pvec)
                M = np.vstack(mats)
                with np.errstate(all='ignore'):
                    bagged = np.nanmean(M, axis=0)
                # if everything NaN (shouldn't happen), fall back
                if np.all(~np.isfinite(bagged)):
                    bagged = base_mix
                mix_sig = bagged
            else:
                mix_sig = base_mix

            # Shock/news gating
            if mask is not None and len(mask) >= len(mix_sig):
                Lg = len(mix_sig)
                mix_sig = mix_sig[:Lg] * (1.0 - params["shock_alpha"] * mask[:Lg])

        # 2) Regime weights (needs both bases)
        if W_on is None or W_def is None: 
            return -1e9, None
        # regime signal from returns proxy
        rproxy = y
        if rproxy is None and (A is not None and W_on is not None and A.shape[1] == W_on.shape[1]):
            Tloc = min(A.shape[0], W_on.shape[0])
            rproxy = np.sum(A[:Tloc] * W_on[:Tloc], axis=1)
        if rproxy is None:
            return -1e9, None

        reg, _thr = regime_from_vol(rproxy, lb=int(params["reg_lb"]), q=float(params["reg_q"]))
        Tloc = min(W_on.shape[0], W_def.shape[0], len(reg))
        if Tloc <= 5: 
            return -1e9, None

        W = reg[:Tloc].reshape(-1,1)*W_on[:Tloc] + (1-reg[:Tloc]).reshape(-1,1)*W_def[:Tloc]

        # 3) Cluster caps
        if clusters is not None and len(clusters) == W.shape[1]:
            W = cap_by_cluster(W, clusters, cap=float(params["cap_cluster"]))

        # 4) Adaptive caps from vol
        vol = np.abs(rproxy[:Tloc])
        caps = adaptive_cap_from_vol(vol, *params["cap_minmax"])[:Tloc]
        W = apply_caps(W, caps)

        # 5) Optional gentle leverage using council mix
        if mix_sig is not None:
            L = min(len(mix_sig), W.shape[0])
            z = zscore(mix_sig[:L])
            lev = 1.0 + 0.2 * np.tanh(z)
            W[:L] = W[:L] * lev.reshape(-1,1)

        # 6) Portfolio returns
        if A is not None and A.shape[1] == W.shape[1] and A.shape[0] >= W.shape[0]:
            pnl = np.sum(W * A[:W.shape[0]], axis=1)
        elif y is not None and len(y) >= W.shape[0]:
            pnl = y[:W.shape[0]]
        else:
            return -1e9, None

        return sharpe(pnl), W

    # --------- Grid search ----------
    best = {"score": -1e9, "params": None, "W": None}
    for a in grid["alpha_meta"]:
        for lb in grid["reg_lb"]:
            for q in grid["reg_q"]:
                for capc in grid["cap_cluster"]:
                    for capmn in grid["cap_minmax"]:
                        for sa in grid["shock_alpha"]:
                            for bags in grid["bags"]:
                                for bsize in grid["bag_size"]:
                                    params = {
                                        "alpha_meta": float(a),
                                        "reg_lb": int(lb),
                                        "reg_q": float(q),
                                        "cap_cluster": float(capc),
                                        "cap_minmax": (float(capmn[0]), float(capmn[1])),
                                        "shock_alpha": float(sa),
                                        "bags": int(bags),
                                        "bag_size": float(bsize),
                                    }
                                    score, W = score_combo(params)
                                    if score > best["score"]:
                                        best = {"score": score, "params": params, "W": W}

    if best["W"] is None:
        print("(!) Tuning failed to build any weights. Ensure inputs exist (tail_blend, risk_parity, returns).")
        raise SystemExit(0)

    # Save results
    Wb = best["W"]
    np.savetxt(RUNS/"tune_best_weights.csv", Wb, delimiter=",")
    (RUNS/"tune_best_config.json").write_text(json.dumps({
        "best_sharpe": best["score"],
        **best["params"]
    }, indent=2))

    # Report card
    html = (f"<p>Sharpe {best['score']:.3f} | α(meta)={best['params']['alpha_meta']:.2f} | "
            f"Regime(lb={best['params']['reg_lb']}, q={best['params']['reg_q']:.2f}) | "
            f"ClusterCap={best['params']['cap_cluster']:.2f} | "
            f"Caps({best['params']['cap_minmax'][0]:.2f}–{best['params']['cap_minmax'][1]:.2f}) | "
            f"Shockα={best['params']['shock_alpha']:.2f} | "
            f"Bags={best['params']['bags']}, size={best['params']['bag_size']:.2f}.</p>")
    append_card("Micro Tuning (P1–P3) ✔", html)

    # Terminal summary
    print("✅ Tuning complete")
    print("Best Sharpe:", f"{best['score']:.3f}")
    print("Best params:", json.dumps(best["params"], indent=2))
    print("Saved:", RUNS/"tune_best_weights.csv", "and", RUNS/"tune_best_config.json")
