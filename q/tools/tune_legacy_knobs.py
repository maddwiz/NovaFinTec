#!/usr/bin/env python3
# Legacy knobs micro-tuner: builds ONE smooth exposure scaler from old add-ons.
# robust CSV loader: can read single-column or DATE,value with header.

import json, csv
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"; RUNS.mkdir(exist_ok=True)
DATA = ROOT/"data"

# ---------- IO helpers (robust) ----------
def load_series(rel):
    p = ROOT/rel
    if not p.exists(): return None
    # try plain numeric CSV first
    try:
        a = np.loadtxt(p, delimiter=",").ravel()
        if np.isfinite(a).any():
            return a
    except Exception:
        pass
    # robust: read last column as float, skip header/DATE
    try:
        vals = []
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            first = True
            for line in f:
                line=line.strip()
                if not line: continue
                parts = [s.strip() for s in line.split(",")]
                # skip header-like first line
                if first and any(tok.lower() in ("date","time","timestamp") for tok in parts):
                    first=False
                    continue
                first=False
                # take LAST column as value
                try:
                    vals.append(float(parts[-1]))
                except Exception:
                    # skip rows that aren't numeric
                    continue
        return np.array(vals, float).ravel() if vals else None
    except Exception:
        return None

def load_matrix(rel):
    p = ROOT/rel
    if not p.exists(): return None
    # plain numeric
    try:
        a = np.loadtxt(p, delimiter=",")
    except Exception:
        # robust: drop header row; keep numeric tail
        try:
            a = np.loadtxt(p, delimiter=",", skiprows=1)
        except Exception:
            # ultimate fallback: parse row by row, keep numeric
            rows=[]
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts=[t.strip() for t in line.strip().split(",")]
                    nums=[]
                    for t in parts:
                        try: nums.append(float(t))
                        except: pass
                    if nums:
                        rows.append(nums)
            if not rows: return None
            maxc=max(len(r) for r in rows)
            m=np.full((len(rows), maxc), np.nan, float)
            for i,r in enumerate(rows):
                m[i,:len(r)]=r
            a=m
    if a.ndim==1: a=a.reshape(-1,1)
    return a

def first_series(paths):
    for rel in paths:
        a = load_series(rel)
        if a is not None and len(a)>5: return a
    return None

def first_matrix(paths):
    for rel in paths:
        a = load_matrix(rel)
        if a is not None and a.size>0: return a
    return None

def append_card(title, html):
    for name in ["report_all.html","report_best_plus.html","report_plus.html","report.html"]:
        f = ROOT/name
        if not f.exists(): continue
        txt = f.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        f.write_text(txt.replace("</body>", card+"</body>") if "</body>" in txt else txt+card, encoding="utf-8")

# ---------- stats & helpers ----------
def sharpe(r):
    r = np.asarray(r, float)
    if r.size == 0: return -1e9
    mu = np.nanmean(r); sd = np.nanstd(r) + 1e-12
    return float((mu/sd)*np.sqrt(252.0))

def z(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / (np.nanstd(x)+1e-12)

def build_asset_returns():
    A = first_matrix(["runs_plus/asset_returns.csv"])
    if A is not None: return A
    series = []
    for fp in sorted((ROOT/"data").glob("*.csv")):
        try:
            with fp.open() as f:
                rdr = csv.DictReader(f)
                closes = [float(row.get("Adj Close") or row.get("Close") or row.get("close")) for row in rdr]
            c = np.array(closes, float)
            if len(c) > 5:
                r = np.diff(c) / (c[:-1] + 1e-12)
                series.append(r)
        except: pass
    if not series: return None
    T = min(len(s) for s in series)
    series = [s[-T:] for s in series]
    return np.stack(series, axis=1)

def smooth_ema(x, beta=0.2):
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    for i, v in enumerate(x):
        out[i] = (1-beta)*(out[i-1] if i>0 else v) + beta*v
    return out

def align_tail_series(a, T, fill=0.0):
    if a is None:
        return None
    x = np.asarray(a, float).ravel()
    x = np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill)
    if len(x) >= T:
        return x[-T:]
    if len(x) == 0:
        return np.full(T, fill, float)
    out = np.full(T, x[0], float)
    out[-len(x):] = x
    return out

def align_tail_matrix(a, T):
    if a is None:
        return None
    x = np.asarray(a, float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.shape[0] >= T:
        return x[-T:]
    out = np.zeros((T, x.shape[1]), float)
    out[-x.shape[0]:] = x
    return out

# ---------- main ----------
if __name__ == "__main__":
    drift = load_series("runs_plus/dna_drift.csv")
    bpm   = load_series("runs_plus/heartbeat_bpm.csv")
    sym   = load_series("runs_plus/symbolic_latent.csv")
    refx  = load_series("runs_plus/reflex_latent.csv")

    y     = first_series(["runs_plus/daily_returns.csv","daily_returns.csv"])
    Wbase = first_matrix([
        "runs_plus/portfolio_weights_final.csv",
        "runs_plus/weights_regime.csv",
        "runs_plus/weights_tail_blend.csv",
        "runs_plus/portfolio_weights.csv",
        "portfolio_weights.csv",
    ])
    A     = build_asset_returns()

    if Wbase is None:
        print("(!) Need base weights/returns first."); raise SystemExit(0)

    # Align to base horizon; short legacy signals are tail-aligned and padded.
    lens = [Wbase.shape[0]]
    if y is not None: lens.append(len(y))
    if A is not None: lens.append(A.shape[0])
    T = int(min(lens))
    Wbase = align_tail_matrix(Wbase, T)
    y = align_tail_series(y, T) if y is not None else None
    A = align_tail_matrix(A, T) if A is not None else None
    drift = align_tail_series(drift, T) if drift is not None else None
    bpm = align_tail_series(bpm, T) if bpm is not None else None
    sym = align_tail_series(sym, T) if sym is not None else None
    refx = align_tail_series(refx, T) if refx is not None else None

    # components
    comps = []; names=[]
    if drift is not None and np.isfinite(drift).any() and np.nanstd(drift) > 1e-9:
        comps.append(-z(drift)); names.append("dna")
    if bpm is not None and np.isfinite(bpm).any() and np.nanstd(bpm) > 1e-9:
        comps.append(-z(bpm));   names.append("bpm")
    if sym is not None and np.isfinite(sym).any() and np.nanstd(sym) > 1e-9:
        comps.append(z(sym));    names.append("sym")
    if refx is not None and np.isfinite(refx).any() and np.nanstd(refx) > 1e-9:
        comps.append(z(refx));   names.append("reflex")

    if not comps:
        print("(!) No legacy signals found (dna/heartbeat/symbolic/reflex). Nothing to tune.")
        raise SystemExit(0)

    X = np.vstack(comps).T
    betas  = [0.1, 0.2, 0.3]
    scales = [0.1, 0.2, 0.3]
    mixes  = [np.eye(X.shape[1])[i] for i in range(X.shape[1])] + [np.ones(X.shape[1])/X.shape[1]]

    if A is not None and A.shape[1] == Wbase.shape[1]:
        base_pnl = np.sum(Wbase[:T] * A[:T], axis=1)
    elif y is not None:
        base_pnl = y[:T]
    else:
        base_pnl = np.zeros(T)

    best = {"score": -1e9}
    for w in mixes:
        raw = X.dot(w)
        for b in betas:
            s  = smooth_ema(raw, beta=b)
            for sc in scales:
                lev = 1.0 + sc * np.tanh(z(s))
                W   = Wbase.copy()
                L   = min(len(lev), W.shape[0])
                W[:L] = W[:L] * lev[:L,None]
                pnl = (np.sum(W[:L] * A[:L], axis=1) if (A is not None and A.shape[1]==W.shape[1]) else base_pnl[:L]*lev[:L])
                score = sharpe(pnl)
                if score > best["score"]:
                    best = {"score": score, "w": w.tolist(), "beta": b, "scale": sc, "lev": lev, "W": W, "names": names}

    np.savetxt(RUNS/"legacy_exposure.csv", best["lev"][:T], delimiter=",")
    np.savetxt(RUNS/"legacy_best_weights.csv", best["W"][:T], delimiter=",")
    (RUNS/"legacy_tune_config.json").write_text(json.dumps({
        "components": best["names"],
        "weights": best["w"],
        "beta": best["beta"],
        "scale": best["scale"],
        "best_sharpe": best["score"],
        "length": int(T)
    }, indent=2))

    html = (f"<p>Sharpe {best['score']:.3f} | mix={best['w']} over {best['names']} | "
            f"EMA β={best['beta']:.2f} | lev±={best['scale']:.2f}. "
            f"Saved legacy_exposure.csv + legacy_best_weights.csv.</p>")
    append_card("Legacy Tuning ✔", html)
    print(f"✅ Legacy tuning complete. Sharpe={best['score']:.3f}")
