#!/usr/bin/env python3
# Final portfolio assembler:
# Picks best base weights then applies (if available):
#   cluster caps → adaptive caps → drawdown scaler → council gate
# Outputs:
#   runs_plus/portfolio_weights_final.csv
# Appends a card to report_*.

import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"; RUNS.mkdir(exist_ok=True)

def load_mat(rel):
    p = ROOT/rel
    if not p.exists(): return None
    try:
        a = np.loadtxt(p, delimiter=",")
    except:
        a = np.loadtxt(p, delimiter=",", skiprows=1)
    if a.ndim == 1: a = a.reshape(-1,1)
    return a

def load_series(rel):
    p = ROOT/rel
    if not p.exists(): return None
    try:    return np.loadtxt(p, delimiter=",").ravel()
    except: return np.loadtxt(p, delimiter=",", skiprows=1).ravel()

def first_mat(paths):
    for rel in paths:
        a = load_mat(rel)
        if a is not None: return a, rel
    return None, None

def append_card(title, html):
    for name in ["report_all.html","report_best_plus.html","report_plus.html","report.html"]:
        f = ROOT/name
        if not f.exists(): continue
        txt = f.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        f.write_text(txt.replace("</body>", card+"</body>") if "</body>" in txt else txt+card, encoding="utf-8")

if __name__ == "__main__":
    # 1) Base weights preference
    W, source = first_mat([
        "runs_plus/weights_regime.csv",
        "runs_plus/weights_tail_blend.csv",
        "runs_plus/portfolio_weights.csv",
        "portfolio_weights.csv",
    ])
    if W is None:
        print("(!) No base weights found; run your pipeline first."); raise SystemExit(0)
    steps = [f"base={source}"]

    T, N = W.shape

    # 2) Cluster caps
    Wc = load_mat("runs_plus/weights_cluster_capped.csv")
    if Wc is not None and Wc.shape[:2] == W.shape:
        W = Wc; steps.append("cluster_caps")
    # 3) Adaptive caps (per-time cap applied to weights)
    Wcap = load_mat("runs_plus/weights_capped.csv")
    if Wcap is not None and Wcap.shape[:2] == W.shape:
        W = Wcap; steps.append("adaptive_caps")

    # 4) Drawdown scaler (guardrails) -> weights_dd_scaled.csv is a full weight matrix
    Wdd = load_mat("runs_plus/weights_dd_scaled.csv")
    if Wdd is not None and Wdd.shape[:2] == W.shape:
        W = Wdd; steps.append("drawdown_floor")

    # 5) Turnover governor (guardrails)
    Wtg = load_mat("runs_plus/weights_turnover_governed.csv")
    if Wtg is not None and Wtg.shape[:2] == W.shape:
        W = Wtg; steps.append("turnover_governor")

    # 6) Council disagreement gate (scalar per t) → scale exposure
    gate = load_series("runs_plus/disagreement_gate.csv")
    if gate is not None:
        L = min(len(gate), W.shape[0])
        g = gate[:L].reshape(-1,1)
        W[:L] = W[:L] * g
        steps.append("council_gate")

    # 7) Save final
    outp = RUNS/"portfolio_weights_final.csv"
    np.savetxt(outp, W, delimiter=",")

    # 8) Small JSON breadcrumb
    (RUNS/"final_portfolio_info.json").write_text(
        json.dumps({"steps": steps, "T": int(T), "N": int(N)}, indent=2)
    )

    # 9) Report card
    html = f"<p>Built <b>portfolio_weights_final.csv</b> (T={T}, N={N}). Steps: {', '.join(steps)}.</p>"
    append_card("Final Portfolio ✔", html)

    print(f"✅ Wrote {outp}  | Steps: {', '.join(steps)}")
