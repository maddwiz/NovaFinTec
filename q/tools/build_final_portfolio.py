#!/usr/bin/env python3
# Final portfolio assembler:
# Picks best base weights then applies (if available):
#   cluster caps → adaptive caps → drawdown scaler → turnover governor
#   → council gate → council/meta leverage → heartbeat/legacy/hive/global/quality/novaspine governors
# Outputs:
#   runs_plus/portfolio_weights_final.csv
# Appends a card to report_*.

import json
import os
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"; RUNS.mkdir(exist_ok=True)

import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from qmods.concentration_governor import govern_matrix

TRACE_STEPS = [
    "council_gate",
    "meta_mix_leverage",
    "meta_mix_reliability",
    "heartbeat_scaler",
    "legacy_scaler",
    "dna_stress_governor",
    "symbolic_governor",
    "dream_coherence",
    "reflex_health_governor",
    "hive_diversification",
    "global_governor",
    "quality_governor",
    "novaspine_context_boost",
    "novaspine_hive_boost",
    "shock_mask_guard",
]

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
    try:
        a = np.loadtxt(p, delimiter=",")
    except Exception:
        try:
            a = np.loadtxt(p, delimiter=",", skiprows=1)
        except Exception:
            vals = []
            try:
                with p.open("r", encoding="utf-8", errors="ignore") as f:
                    first = True
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = [s.strip() for s in line.split(",")]
                        if first and any(tok.lower() in ("date", "time", "timestamp") for tok in parts):
                            first = False
                            continue
                        first = False
                        try:
                            vals.append(float(parts[-1]))
                        except Exception:
                            continue
            except Exception:
                return None
            return np.asarray(vals, float).ravel() if vals else None
    a = np.asarray(a, float)
    if a.ndim == 2 and a.shape[1] >= 1:
        a = a[:, -1]
    return a.ravel()

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
    trace = {}

    def _trace_put(name, vec=None):
        x = np.ones(T, dtype=float)
        if vec is not None:
            v = np.asarray(vec, float).ravel()
            L = min(T, len(v))
            if L > 0:
                x[:L] = v[:L]
        trace[name] = x

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
        _trace_put("council_gate", np.clip(gate[:L], 0.0, 1.0))

    # 7) Council/meta leverage (scalar per t) from mix confidence
    lev = load_series("runs_plus/meta_mix_leverage.csv")
    if lev is None:
        mix = load_series("runs_plus/meta_mix.csv")
        if mix is not None:
            lev = np.clip(1.0 + 0.20 * np.abs(mix), 0.80, 1.30)
    if lev is not None:
        L = min(len(lev), W.shape[0])
        lv = np.clip(lev[:L], 0.70, 1.40).reshape(-1, 1)
        W[:L] = W[:L] * lv
        steps.append("meta_mix_leverage")
        _trace_put("meta_mix_leverage", lv.ravel())

    # 8) Meta/council reliability governor from confidence calibration.
    mrg = load_series("runs_plus/meta_mix_reliability_governor.csv")
    if mrg is not None:
        L = min(len(mrg), W.shape[0])
        mr = np.clip(mrg[:L], 0.70, 1.20).reshape(-1, 1)
        W[:L] = W[:L] * mr
        steps.append("meta_mix_reliability")
        _trace_put("meta_mix_reliability", mr.ravel())

    # 9) Heartbeat exposure scaler (risk metabolism)
    hb = load_series("runs_plus/heartbeat_exposure_scaler.csv")
    if hb is not None:
        L = min(len(hb), W.shape[0])
        hs = np.clip(hb[:L], 0.40, 1.20).reshape(-1, 1)
        W[:L] = W[:L] * hs
        steps.append("heartbeat_scaler")
        _trace_put("heartbeat_scaler", hs.ravel())

    # 10) Legacy blended scaler from DNA/Heartbeat/Symbolic/Reflex tuner.
    lex = load_series("runs_plus/legacy_exposure.csv")
    if lex is not None:
        L = min(len(lex), W.shape[0])
        ls = np.clip(lex[:L], 0.40, 1.30).reshape(-1, 1)
        W[:L] = W[:L] * ls
        steps.append("legacy_scaler")
        _trace_put("legacy_scaler", ls.ravel())

    # 11) DNA stress governor from drift/velocity regime diagnostics.
    dsg = load_series("runs_plus/dna_stress_governor.csv")
    if dsg is not None:
        L = min(len(dsg), W.shape[0])
        ds = np.clip(dsg[:L], 0.70, 1.15).reshape(-1, 1)
        W[:L] = W[:L] * ds
        steps.append("dna_stress_governor")
        _trace_put("dna_stress_governor", ds.ravel())

    # 12) Symbolic affective governor.
    sgg = load_series("runs_plus/symbolic_governor.csv")
    if sgg is not None:
        L = min(len(sgg), W.shape[0])
        sg = np.clip(sgg[:L], 0.70, 1.15).reshape(-1, 1)
        W[:L] = W[:L] * sg
        steps.append("symbolic_governor")
        _trace_put("symbolic_governor", sg.ravel())

    # 13) Dream/reflex/symbolic coherence governor.
    dcg = load_series("runs_plus/dream_coherence_governor.csv")
    if dcg is not None:
        L = min(len(dcg), W.shape[0])
        ds = np.clip(dcg[:L], 0.70, 1.20).reshape(-1, 1)
        W[:L] = W[:L] * ds
        steps.append("dream_coherence")
        _trace_put("dream_coherence", ds.ravel())

    # 14) Reflex health governor from reflexive feedback diagnostics.
    rhg = load_series("runs_plus/reflex_health_governor.csv")
    if rhg is not None:
        L = min(len(rhg), W.shape[0])
        rs = np.clip(rhg[:L], 0.70, 1.15).reshape(-1, 1)
        W[:L] = W[:L] * rs
        steps.append("reflex_health_governor")
        _trace_put("reflex_health_governor", rs.ravel())

    # 15) Hive diversification governor from ecosystem layer.
    hg = load_series("runs_plus/hive_diversification_governor.csv")
    if hg is not None:
        L = min(len(hg), W.shape[0])
        hs = np.clip(hg[:L], 0.75, 1.08).reshape(-1, 1)
        W[:L] = W[:L] * hs
        steps.append("hive_diversification")
        _trace_put("hive_diversification", hs.ravel())

    # 16) Global governor (regime * stability) from guardrails.
    gg = load_series("runs_plus/global_governor.csv")
    if gg is None:
        rg = load_series("runs_plus/regime_governor.csv")
        sg = load_series("runs_plus/stability_governor.csv")
        if rg is not None and sg is not None:
            L = min(len(rg), len(sg))
            gg = np.clip(0.55 * rg[:L] + 0.45 * sg[:L], 0.45, 1.10)
        elif rg is not None:
            gg = np.clip(rg, 0.45, 1.10)
        elif sg is not None:
            gg = np.clip(sg, 0.45, 1.10)
    if gg is not None:
        L = min(len(gg), W.shape[0])
        g = np.clip(gg[:L], 0.30, 1.15).reshape(-1, 1)
        W[:L] = W[:L] * g
        steps.append("global_governor")
        _trace_put("global_governor", g.ravel())

    # 17) Reliability quality governor from nested/hive/council diagnostics.
    qg = load_series("runs_plus/quality_governor.csv")
    if qg is not None:
        L = min(len(qg), W.shape[0])
        qs = np.clip(qg[:L], 0.45, 1.20).reshape(-1, 1)
        W[:L] = W[:L] * qs
        steps.append("quality_governor")
        _trace_put("quality_governor", qs.ravel())

    # 18) NovaSpine recall-context boost (if available).
    ncb = load_series("runs_plus/novaspine_context_boost.csv")
    if ncb is not None:
        L = min(len(ncb), W.shape[0])
        nb = np.clip(ncb[:L], 0.85, 1.15).reshape(-1, 1)
        W[:L] = W[:L] * nb
        steps.append("novaspine_context_boost")
        _trace_put("novaspine_context_boost", nb.ravel())

    # 19) NovaSpine per-hive alignment boost (global projection).
    nhb = load_series("runs_plus/novaspine_hive_boost.csv")
    if nhb is not None:
        L = min(len(nhb), W.shape[0])
        hb = np.clip(nhb[:L], 0.85, 1.15).reshape(-1, 1)
        W[:L] = W[:L] * hb
        steps.append("novaspine_hive_boost")
        _trace_put("novaspine_hive_boost", hb.ravel())

    # 20) Shock/news mask exposure cut.
    sm = load_series("runs_plus/shock_mask.csv")
    if sm is not None:
        L = min(len(sm), W.shape[0])
        alpha = float(np.clip(float(os.getenv("Q_SHOCK_ALPHA", "0.35")), 0.0, 1.0))
        sc = (1.0 - alpha * np.clip(sm[:L], 0.0, 1.0)).reshape(-1, 1)
        W[:L] = W[:L] * sc
        steps.append("shock_mask_guard")
        _trace_put("shock_mask_guard", sc.ravel())

    # 21) Concentration governor (top1/top3 + HHI caps).
    use_conc = str(os.getenv("Q_USE_CONCENTRATION_GOV", "1")).strip().lower() in {"1", "true", "yes", "on"}
    if use_conc:
        top1 = float(np.clip(float(os.getenv("Q_CONCENTRATION_TOP1_CAP", "0.18")), 0.01, 1.0))
        top3 = float(np.clip(float(os.getenv("Q_CONCENTRATION_TOP3_CAP", "0.42")), 0.01, 1.0))
        hhi = float(np.clip(float(os.getenv("Q_CONCENTRATION_MAX_HHI", "0.14")), 0.01, 1.0))
        W, cstats = govern_matrix(W, top1_cap=top1, top3_cap=top3, max_hhi=hhi)
        steps.append("concentration_governor")
        (RUNS / "concentration_governor_info.json").write_text(
            json.dumps(
                {
                    "enabled": True,
                    "top1_cap": top1,
                    "top3_cap": top3,
                    "max_hhi": hhi,
                    "stats": cstats,
                },
                indent=2,
            )
        )

    # Build governor trace artifact for auditability.
    for name in TRACE_STEPS:
        if name not in trace:
            _trace_put(name, None)
    trace_mat = np.column_stack([trace[name] for name in TRACE_STEPS])
    trace_total = np.prod(trace_mat, axis=1)
    trace_out = np.column_stack([trace_mat, trace_total])
    trace_cols = TRACE_STEPS + ["runtime_total_scalar"]
    np.savetxt(
        RUNS / "final_governor_trace.csv",
        trace_out,
        delimiter=",",
        header=",".join(trace_cols),
        comments="",
    )

    # 22) Save final
    outp = RUNS/"portfolio_weights_final.csv"
    np.savetxt(outp, W, delimiter=",")

    # 23) Small JSON breadcrumb
    (RUNS/"final_portfolio_info.json").write_text(
        json.dumps(
            {
                "steps": steps,
                "T": int(T),
                "N": int(N),
                "governor_trace_file": str(RUNS / "final_governor_trace.csv"),
                "runtime_total_scalar_mean": float(np.mean(trace_total)),
                "runtime_total_scalar_min": float(np.min(trace_total)),
                "runtime_total_scalar_max": float(np.max(trace_total)),
            },
            indent=2,
        )
    )

    # 24) Report card
    html = f"<p>Built <b>portfolio_weights_final.csv</b> (T={T}, N={N}). Steps: {', '.join(steps)}.</p>"
    append_card("Final Portfolio ✔", html)

    print(f"✅ Wrote {outp}  | Steps: {', '.join(steps)}")
