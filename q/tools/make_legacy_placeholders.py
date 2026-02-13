#!/usr/bin/env python3
# Create safe placeholders for legacy signals if missing:
#   runs_plus/heartbeat_bpm.csv      (volatility→"bpm")
#   runs_plus/symbolic_latent.csv    (from shock_mask if present, else 0)
#   runs_plus/reflex_latent.csv      (from meta_mix or synapses)
# Appends a report card. Does nothing if files already exist.

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"; RUNS.mkdir(exist_ok=True)

def load_series(rel):
    p = ROOT/rel
    if not p.exists(): return None
    # try raw; fallback skip 1 line
    try:    return np.loadtxt(p, delimiter=",").ravel()
    except: return np.loadtxt(p, delimiter=",", skiprows=1).ravel()

def first_series(paths):
    for rel in paths:
        a = load_series(rel)
        if a is not None: return a
    return None

def z(x):
    x = np.asarray(x, float)
    mu = np.nanmean(x); sd = np.nanstd(x)+1e-12
    return (x-mu)/sd

def ema(x, beta=0.2):
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    for i,v in enumerate(x):
        out[i] = (1-beta)*(out[i-1] if i>0 else v) + beta*v
    return out

def append_card(title, html):
    for name in ["report_all.html","report_best_plus.html","report_plus.html","report.html"]:
        f = ROOT/name
        if not f.exists(): continue
        txt = f.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        f.write_text(txt.replace("</body>", card+"</body>") if "</body>" in txt else txt+card, encoding="utf-8")

if __name__ == "__main__":
    made = []

    # 1) HEARTBEAT (bpm) from returns vol (scaled to ~60..140)
    hb = load_series("runs_plus/heartbeat_bpm.csv")
    if hb is None:
        r = first_series(["runs_plus/daily_returns.csv","daily_returns.csv"])
        if r is not None and r.size>50:
            vol = ema(np.abs(r), beta=0.2)
            v = (z(vol) + 2.0) / 4.0       # map roughly 0..1
            bpm = 60 + 80*np.clip(v, 0, 1) # 60..140
            np.savetxt(RUNS/"heartbeat_bpm.csv", bpm, delimiter=",")
            made.append("heartbeat_bpm.csv")

    # 2) SYMBOLIC latent from shock_mask (if exists), else zeros
    sym = load_series("runs_plus/symbolic_latent.csv")
    if sym is None:
        mask = load_series("runs_plus/shock_mask.csv")
        Tref = None
        for rel in ["runs_plus/daily_returns.csv","daily_returns.csv","runs_plus/heartbeat_bpm.csv"]:
            a = load_series(rel)
            if a is not None:
                Tref = len(a); break
        if Tref is None and mask is not None:
            Tref = len(mask)
        if Tref is not None:
            if mask is None:
                symv = np.zeros(Tref)
            else:
                symv = ema(mask[:Tref].astype(float), beta=0.2)
            np.savetxt(RUNS/"symbolic_latent.csv", symv, delimiter=",")
            made.append("symbolic_latent.csv")

    # 3) REFLEX latent from meta_mix (preferred) or synapses/meta_stack
    refx = load_series("runs_plus/reflex_latent.csv")
    if refx is None:
        src = first_series(["runs_plus/meta_mix.csv",
                            "runs_plus/synapses_pred.csv",
                            "runs_plus/meta_stack_pred.csv"])
        if src is not None:
            ref = ema(z(src), beta=0.2)  # smoothed latent
            np.savetxt(RUNS/"reflex_latent.csv", ref, delimiter=",")
            made.append("reflex_latent.csv")

    if not made:
        print("Nothing to create; all legacy files already exist.")
    else:
        print("✅ Created:", ", ".join(made))
        append_card("Legacy Placeholders ✔", f"<p>Created: {', '.join(made)}</p>")
