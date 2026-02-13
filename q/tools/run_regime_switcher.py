#!/usr/bin/env python3
# Regime Switcher (side-by-side):
# - Builds a simple regime signal from rolling volatility of daily returns
# - Blends Risk-On weights vs Defensive weights
# Reads:
#   runs_plus/daily_returns.csv or daily_returns.csv   [T]
#   Risk-On:  runs_plus/weights_tail_blend.csv OR portfolio_weights.csv OR runs_plus/portfolio_weights.csv
#   Defensive: runs_plus/risk_parity_weights.csv OR runs_plus/weights_capped.csv
# Writes:
#   runs_plus/regime_signal.csv           [T] (0=defensive..1=on)
#   runs_plus/weights_regime.csv          [T x N]
# Appends a card.

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"; RUNS.mkdir(exist_ok=True)

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

def regime_from_vol(r, lb=63, q=0.5):
    r = np.asarray(r, float)
    vol = np.zeros_like(r)
    for i in range(len(r)):
        j = max(0, i-lb+1)
        w = r[j:i+1]
        vol[i] = np.nanstd(w)
    # Low vol → regime=1 (risk-on), High vol → 0
    thr = np.nanquantile(vol, q)
    reg = (vol <= thr).astype(float)
    return reg, vol, thr

if __name__ == "__main__":
    r = first_series(["runs_plus/daily_returns.csv","daily_returns.csv"])
    if r is None:
        print("(!) No returns; skipping."); raise SystemExit(0)

    # Load weight paths
    W_on  = first_matrix(["runs_plus/weights_tail_blend.csv","portfolio_weights.csv","runs_plus/portfolio_weights.csv"])
    W_def = first_matrix(["runs_plus/risk_parity_weights.csv","runs_plus/weights_capped.csv"])
    if W_on is None or W_def is None:
        print("(!) Missing weight paths; run tail_blender and risk_parity first. Skipping.")
        raise SystemExit(0)

    T = min(len(r), W_on.shape[0], W_def.shape[0])
    r = r[:T]; W_on = W_on[:T]; W_def = W_def[:T]
    N = W_on.shape[1]

    reg, vol, thr = regime_from_vol(r, lb=63, q=0.5)
    np.savetxt(RUNS/"regime_signal.csv", reg[:T], delimiter=",")

    # Smooth a bit to reduce flip-flop
    lbeta = 0.2
    sm = np.zeros_like(reg)
    for i, x in enumerate(reg):
        sm[i] = (1-lbeta)* (sm[i-1] if i>0 else x) + lbeta*x
    sm = np.clip(sm, 0, 1)

    W_reg = sm.reshape(-1,1) * W_on + (1-sm).reshape(-1,1) * W_def
    np.savetxt(RUNS/"weights_regime.csv", W_reg, delimiter=",")

    html = f"<p>Vol-threshold={thr:.4f} (lb=63). Blended Risk-On vs Defensive → weights_regime.csv (T={T}, N={N}).</p>"
    append_card("Regime Switcher ✔", html)
    print(f"✅ Wrote runs_plus/weights_regime.csv and regime_signal.csv")
