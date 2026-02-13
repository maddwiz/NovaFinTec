#!/usr/bin/env python3
# Ecosystem Age Governor
# Reads:
#   runs_plus/cross_hive_weights.csv
#   runs_plus/hive_signals.csv
# Writes:
#   runs_plus/weights_cross_hive_governed.csv
#   runs_plus/hive_evolution.json

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)

from qmods.ecosystem_age import govern_hive_weights


def append_card(title: str, html: str):
    for name in ["report_all.html", "report_best_plus.html", "report_plus.html", "report.html"]:
        p = ROOT / name
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        txt = txt.replace("</body>", card + "</body>") if "</body>" in txt else txt + card
        p.write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    p_w = RUNS / "cross_hive_weights.csv"
    p_h = RUNS / "hive_signals.csv"
    if not p_w.exists() or not p_h.exists():
        raise SystemExit("Need runs_plus/cross_hive_weights.csv and runs_plus/hive_signals.csv")

    w = pd.read_csv(p_w)
    h = pd.read_csv(p_h)

    governed, summary = govern_hive_weights(w, h)
    governed.to_csv(RUNS / "weights_cross_hive_governed.csv", index=False)
    (RUNS / "hive_evolution.json").write_text(json.dumps(summary, indent=2))

    html = (
        f"<p>Governed hive weights saved to <b>weights_cross_hive_governed.csv</b>.</p>"
        f"<p>Latest governed: {summary.get('latest_governed_weights', {})}</p>"
        f"<p>Events: {len(summary.get('events', []))}</p>"
    )
    append_card("Ecosystem Age Governor ✔", html)

    print(f"✅ Wrote {RUNS/'weights_cross_hive_governed.csv'}")
    print(f"✅ Wrote {RUNS/'hive_evolution.json'}")
