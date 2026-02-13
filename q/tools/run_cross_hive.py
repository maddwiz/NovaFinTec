#!/usr/bin/env python3
# Cross-Hive Arbitration (adaptive, from hive_signals)
# Reads: runs_plus/hive_signals.csv
# Writes:
#   runs_plus/cross_hive_weights.csv
#   runs_plus/hive_score_<hive>.csv
#   runs_plus/cross_hive_summary.json
# Appends a report card.

import json
import numpy as np
import pandas as pd
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from qmods.cross_hive_arb_v1 import arb_weights

RUNS = ROOT/"runs_plus"; RUNS.mkdir(exist_ok=True)

def append_card(title, html):
    for name in ["report_all.html", "report_best_plus.html", "report_plus.html"]:
        p = ROOT/name
        if not p.exists(): continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        txt = txt.replace("</body>", card + "</body>") if "</body>" in txt else txt+card
        p.write_text(txt, encoding="utf-8")
        print(f"✅ Appended card to {name}")

if __name__ == "__main__":
    p = RUNS / "hive_signals.csv"
    if not p.exists():
        raise SystemExit("Missing runs_plus/hive_signals.csv (run tools/make_hive.py first)")

    h = pd.read_csv(p)
    need = {"DATE", "HIVE", "hive_signal"}
    if not need.issubset(h.columns):
        raise SystemExit("hive_signals.csv missing required columns: DATE,HIVE,hive_signal")
    h["DATE"] = pd.to_datetime(h["DATE"], errors="coerce")
    h = h.dropna(subset=["DATE"]).sort_values(["DATE", "HIVE"])

    if "hive_health" not in h.columns:
        # fallback to rolling Sharpe proxy
        out = []
        for hive, g in h.groupby("HIVE"):
            gg = g.sort_values("DATE").copy()
            mu = gg["hive_signal"].rolling(63, min_periods=20).mean()
            sd = gg["hive_signal"].rolling(63, min_periods=20).std(ddof=1).replace(0, np.nan)
            gg["hive_health"] = np.tanh((mu / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0) / 2.0)
            out.append(gg)
        h = pd.concat(out, ignore_index=True)

    if "hive_stability" not in h.columns:
        out = []
        for hive, g in h.groupby("HIVE"):
            gg = g.sort_values("DATE").copy()
            gg["hive_stability"] = (1.0 - gg["hive_signal"].rolling(21, min_periods=7).std(ddof=1).fillna(0.0)).clip(0.0, 1.0)
            out.append(gg)
        h = pd.concat(out, ignore_index=True)

    # Build score + penalties per hive on aligned dates
    pivot_sig = h.pivot(index="DATE", columns="HIVE", values="hive_signal").fillna(0.0)
    pivot_health = h.pivot(index="DATE", columns="HIVE", values="hive_health").reindex(pivot_sig.index).fillna(0.0)
    pivot_stab = h.pivot(index="DATE", columns="HIVE", values="hive_stability").reindex(pivot_sig.index).fillna(0.0)
    # pseudo drawdown on cumulative hive signal
    eq = (1.0 + pivot_sig.clip(-0.95, 0.95)).cumprod()
    dd = (eq / np.maximum(eq.cummax(), 1e-12) - 1.0).clip(-1.0, 0.0).abs()
    disagree = (1.0 - pivot_stab).clip(0.0, 1.0)

    scores = {}
    dd_pen = {}
    dg_pen = {}
    for hive in pivot_sig.columns:
        score = 0.55 * pivot_health[hive].values + 0.35 * pivot_sig[hive].rolling(5, min_periods=2).mean().values + 0.10 * pivot_stab[hive].values
        scores[str(hive)] = np.nan_to_num(score, nan=0.0)
        dd_pen[str(hive)] = np.nan_to_num(dd[hive].values, nan=0.0)
        dg_pen[str(hive)] = np.nan_to_num(disagree[hive].values, nan=0.0)
        np.savetxt(RUNS / f"hive_score_{hive}.csv", scores[str(hive)], delimiter=",")

    names, W = arb_weights(scores, alpha=2.2, drawdown_penalty=dd_pen, disagreement_penalty=dg_pen)
    out = pd.DataFrame(W, index=pivot_sig.index, columns=names).reset_index().rename(columns={"index": "DATE"})
    out.to_csv(RUNS / "cross_hive_weights.csv", index=False)

    summary = {
        "hives": names,
        "rows": int(len(out)),
        "date_min": str(out["DATE"].min().date()) if len(out) else None,
        "date_max": str(out["DATE"].max().date()) if len(out) else None,
        "latest_weights": {k: float(out.iloc[-1][k]) for k in names} if len(out) else {},
    }
    (RUNS / "cross_hive_summary.json").write_text(json.dumps(summary, indent=2))

    html = f"<p>Cross-hive weights over {len(names)} hives saved to cross_hive_weights.csv</p><p>Latest: {summary['latest_weights']}</p>"
    append_card("Cross-Hive Arbitration ✔", html)
    print(f"✅ Wrote {RUNS/'cross_hive_weights.csv'}")
    print(f"✅ Wrote {RUNS/'cross_hive_summary.json'}")
