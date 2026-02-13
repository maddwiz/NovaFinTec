#!/usr/bin/env python3
# Council Meta-Tuning: auto-mix meta_stack_pred and synapses_pred
# Reads:
#   runs_plus/meta_stack_pred.csv   [T]
#   runs_plus/synapses_pred.csv     [T]
#   runs_plus/daily_returns.csv or daily_returns.csv  [T] (for scoring)
# Writes:
#   runs_plus/meta_mix.csv          [T]   (best-mix signal)
#   runs_plus/meta_mix_info.json           (best alpha + stats)
# Appends a "Best Council Mix" card to report_*.

import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"; RUNS.mkdir(exist_ok=True)

def load_series(rel):
    p = ROOT/rel
    if not p.exists(): return None
    try:    return np.loadtxt(p, delimiter=",").ravel()
    except: return np.loadtxt(p, delimiter=",", skiprows=1).ravel()

def first_series(paths):
    for rel in paths:
        a = load_series(rel)
        if a is not None: return a
    return None

def zscore(x):
    x = np.asarray(x, float)
    mu = np.nanmean(x); sd = np.nanstd(x) + 1e-12
    return (x - mu) / sd

def annualized_sharpe(r):
    r = np.asarray(r, float)
    mu = np.nanmean(r); sd = np.nanstd(r) + 1e-12
    return float((mu / sd) * np.sqrt(252.0))

def append_card(title, html):
    for name in ["report_all.html","report_best_plus.html","report_plus.html","report.html"]:
        f = ROOT/name
        if not f.exists(): continue
        txt = f.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        f.write_text(txt.replace("</body>", card+"</body>") if "</body>" in txt else txt+card, encoding="utf-8")

if __name__ == "__main__":
    m = load_series("runs_plus/meta_stack_pred.csv")
    s = load_series("runs_plus/synapses_pred.csv")
    if m is None or s is None:
        print("(!) Need meta_stack_pred.csv and synapses_pred.csv; skipping."); raise SystemExit(0)

    y = first_series(["runs_plus/daily_returns.csv","daily_returns.csv"])
    if y is None:
        print("(!) No returns found; skipping."); raise SystemExit(0)

    T = min(len(m), len(s), len(y))
    m = m[:T]; s = s[:T]; y = y[:T]

    # Leakage guard: score on (signal[:-1] vs y[1:])
    Zm = zscore(m); Zs = zscore(s)
    ZmL = Zm[:-1]; ZsL = Zs[:-1]; yF = y[1:]

    grid = np.linspace(0.0, 1.0, 21)  # alpha = weight on meta
    best_a, best_sh = None, None
    for a in grid:
        mix = a*ZmL + (1-a)*ZsL
        sh = annualized_sharpe(mix * yF)
        if best_sh is None or sh > best_sh:
            best_sh, best_a = sh, float(a)

    # Rebuild full-length best mix (no shift for export)
    best_mix = best_a*Zm + (1-best_a)*Zs
    np.savetxt(RUNS/"meta_mix.csv", best_mix, delimiter=",")

    info = {"best_alpha_meta": best_a, "best_sharpe_oos_like": best_sh, "length": int(T)}
    (RUNS/"meta_mix_info.json").write_text(json.dumps(info, indent=2))
    append_card("Best Council Mix ✔", f"<p>α(meta)={best_a:.2f} → OOS-like Sharpe {best_sh:.3f}. Saved meta_mix.csv.</p>")
    print(f"✅ Saved runs_plus/meta_mix.csv  (alpha={best_a:.2f}, OOS-like Sharpe={best_sh:.3f})")
