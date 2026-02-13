#!/usr/bin/env python3
# Computes guardrail summaries + optional DD scaling output.
# Writes:
#  - runs_plus/guardrails_summary.json
#  - runs_plus/weights_dd_scaled.csv (if inputs found)
#  - runs_plus/disagreement_gate.csv (if council_votes.csv found)
# Also appends a small card to report_*.

import json, csv, os
from pathlib import Path
import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmods.guardrails_bundle import (
    apply_turnover_governor,
    disagreement_gate,
    parameter_stability_filter,
    turnover_cost_penalty,
)
from qmods.drawdown_floor import drawdown_floor_series

RUNS = ROOT / "runs_plus"; RUNS.mkdir(exist_ok=True)

def _maybe_load_csv(p: Path):
    if p.exists():
        try:
            a = np.loadtxt(p, delimiter=",")
            if a.ndim == 1: a = a.reshape(-1,1)
            return a
        except Exception:
            try:
                a = np.loadtxt(p, delimiter=",", skiprows=1)
                if a.ndim == 1: a = a.reshape(-1,1)
                return a
            except Exception:
                return None
    return None

def _load_first_non_none(paths):
    for p in paths:
        a = _maybe_load_csv(p)
        if a is not None:
            return a
    return None

def _append_report_card(title, html):
    for name in ["report_all.html", "report_best_plus.html", "report_plus.html", "report.html"]:
        p = ROOT / name
        if not p.exists(): 
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        txt = txt.replace("</body>", card + "</body>") if "</body>" in txt else txt + card
        p.write_text(txt, encoding="utf-8")
        print(f"✅ Appended guardrails card to {name}")

def main():
    # 1) Parameter stability — runs_plus/params_history.csv (n_windows x n_params)
    params_hist = _maybe_load_csv(RUNS/"params_history.csv")
    if params_hist is not None and params_hist.ndim == 2 and params_hist.shape[0] >= 2:
        stab_res = parameter_stability_filter(params_hist, thresh=0.6)
        stab = {"stability_score": float(stab_res.stability_score),
                "kept_params": int(stab_res.keep_mask.sum()),
                "total_params": int(stab_res.keep_mask.size)}
    else:
        stab = {"note": "params_history.csv not found; skipped"}

    # 2) Turnover cost + turnover governor — portfolio_weights.csv (T x N)
    wts = _load_first_non_none([ROOT/"portfolio_weights.csv", RUNS/"portfolio_weights.csv"])
    if wts is not None and wts.shape[0] > 2:
        fee_bps = 5.0
        max_step = float(np.clip(float(os.getenv("TURNOVER_MAX_STEP", "0.35")), 0.0, 10.0))
        gov = apply_turnover_governor(wts, max_step_turnover=max_step)
        np.savetxt(RUNS/"weights_turnover_governed.csv", gov.weights, delimiter=",")
        np.savetxt(RUNS/"turnover_before.csv", gov.turnover_before, delimiter=",")
        np.savetxt(RUNS/"turnover_after.csv", gov.turnover_after, delimiter=",")

        cost_raw = turnover_cost_penalty(wts, fee_bps=fee_bps)
        cost_gov = turnover_cost_penalty(gov.weights, fee_bps=fee_bps)
        mean_scale = float(np.mean(gov.scale_applied)) if gov.scale_applied.size else 1.0
        cost = {
            "turnover_cost_sharpe_adj": float(cost_raw),
            "turnover_cost_sharpe_adj_governed": float(cost_gov),
            "turnover_before_mean": float(np.mean(gov.turnover_before)) if gov.turnover_before.size else 0.0,
            "turnover_after_mean": float(np.mean(gov.turnover_after)) if gov.turnover_after.size else 0.0,
            "turnover_scale_mean": mean_scale,
            "turnover_max_step": max_step,
        }
    else:
        cost = {"note": "portfolio_weights.csv not found; skipped"}

    # 3) Council disagreement gate — runs_plus/council_votes.csv (T x K)
    votes = _maybe_load_csv(RUNS/"council_votes.csv")
    if votes is not None and votes.shape[0] >= 1:
        gates = np.array([disagreement_gate(v) for v in votes])
        np.savetxt(RUNS/"disagreement_gate.csv", gates, delimiter=",")
        gate_stats = {"gate_mean": float(np.mean(gates)),
                      "gate_min": float(np.min(gates)),
                      "gate_max": float(np.max(gates))}
    else:
        gate_stats = {"note": "council_votes.csv not found; skipped"}

    # 4) Optional drawdown scaling — runs_plus/cum_pnl.csv + weights
    cum = _maybe_load_csv(RUNS/"cum_pnl.csv")
    if (cum is not None) and (wts is not None) and cum.shape[0] >= wts.shape[0]:
        scale = drawdown_floor_series(cum[:wts.shape[0], 0] if cum.ndim==2 and cum.shape[1]==1 else cum[:wts.shape[0]].ravel(),
                                      floor=-0.12, cut=0.6)
        w_scaled = wts.copy()
        T = min(len(scale), w_scaled.shape[0])
        for t in range(T):
            w_scaled[t] *= scale[t]
        np.savetxt(RUNS/"weights_dd_scaled.csv", w_scaled, delimiter=",")
        dd_out = {"dd_floor": -0.12, "cut": 0.6}
    else:
        dd_out = {"note": "cum_pnl.csv or weights missing; skipped"}

    out = {"stability": stab, "turnover_cost": cost, "disagreement_gate": gate_stats, "drawdown_scaler": dd_out}
    (RUNS/"guardrails_summary.json").write_text(json.dumps(out, indent=2))
    print(f"✅ Wrote {RUNS/'guardrails_summary.json'}")

    # 5) Report card
    html_bits = []
    if "stability_score" in stab:
        html_bits.append(f"<p><b>Stability:</b> score {stab['stability_score']:.2f} (kept {stab['kept_params']}/{stab['total_params']})</p>")
    if "turnover_cost_sharpe_adj" in cost:
        html_bits.append(
            f"<p><b>Turnover cost adj:</b> raw {cost['turnover_cost_sharpe_adj']:.4f}, "
            f"governed {cost.get('turnover_cost_sharpe_adj_governed', cost['turnover_cost_sharpe_adj']):.4f} "
            f"(max step {cost.get('turnover_max_step', 0.0):.3f})</p>"
        )
        html_bits.append(
            f"<p><b>Turnover mean:</b> before {cost.get('turnover_before_mean', 0.0):.4f}, "
            f"after {cost.get('turnover_after_mean', 0.0):.4f}, "
            f"scale mean {cost.get('turnover_scale_mean', 1.0):.3f}</p>"
        )
    if "gate_mean" in gate_stats:
        html_bits.append(f"<p><b>Council gate:</b> mean {gate_stats['gate_mean']:.3f} (min {gate_stats['gate_min']:.3f}, max {gate_stats['gate_max']:.3f})</p>")
    if "dd_floor" in dd_out:
        html_bits.append(f"<p><b>DD Reallocator:</b> floor {dd_out['dd_floor']}, cut {dd_out['cut']} (weights_dd_scaled.csv)</p>")
    if html_bits:
        _append_report_card("GUARDRAILS SUMMARY", "\n".join(html_bits))

if __name__ == "__main__":
    main()
