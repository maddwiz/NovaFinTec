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
import os
import sys
from pathlib import Path

import numpy as np
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

    governed, summary = govern_hive_weights(
        w,
        h,
        half_life_days=int(max(10, int(float(os.getenv("ECO_HALF_LIFE_DAYS", "63"))))),
        atrophy_floor=float(np.clip(float(os.getenv("ECO_ATROPHY_FLOOR", "0.10")), 0.01, 0.60)),
        inertia=float(np.clip(float(os.getenv("ECO_INERTIA", "0.85")), 0.0, 0.98)),
        atrophy_trigger=float(np.clip(float(os.getenv("ECO_ATROPHY_TRIGGER", "0.32")), 0.05, 0.95)),
        atrophy_cap=float(np.clip(float(os.getenv("ECO_ATROPHY_CAP", "0.06")), 0.01, 0.50)),
        split_trigger=float(np.clip(float(os.getenv("ECO_SPLIT_TRIGGER", "0.55")), 0.10, 0.95)),
        split_vol_trigger=float(np.clip(float(os.getenv("ECO_SPLIT_VOL_TRIGGER", "0.22")), 0.01, 2.0)),
        split_intensity=float(np.clip(float(os.getenv("ECO_SPLIT_INTENSITY", "0.25")), 0.01, 1.0)),
        fusion_corr=float(np.clip(float(os.getenv("ECO_FUSION_CORR", "0.92")), 0.50, 0.999)),
        fusion_intensity=float(np.clip(float(os.getenv("ECO_FUSION_INTENSITY", "0.12")), 0.0, 1.0)),
        recovery_slope_trigger=float(np.clip(float(os.getenv("ECO_RECOVERY_SLOPE_TRIGGER", "0.015")), 0.0, 0.50)),
        split_cooloff_strength=float(np.clip(float(os.getenv("ECO_SPLIT_COOLOFF_STRENGTH", "0.35")), 0.0, 1.0)),
    )
    governed.to_csv(RUNS / "weights_cross_hive_governed.csv", index=False)
    (RUNS / "hive_evolution.json").write_text(json.dumps(summary, indent=2))

    # Diversification governor from hive concentration.
    hive_cols = [c for c in governed.columns if c != "DATE"]
    div_stats = {}
    if hive_cols:
        mat = governed[hive_cols].astype(float).values
        hhi = (mat * mat).sum(axis=1)
        n = max(1, len(hive_cols))
        base = 1.0 / n
        # normalize concentration to [0,1]
        conc = np.clip((hhi - base) / (1.0 - base + 1e-12), 0.0, 1.0)
        gov = np.clip(1.05 - 0.25 * conc, 0.80, 1.05)
        pd.DataFrame({"DATE": governed["DATE"], "hive_diversification_governor": gov}).to_csv(
            RUNS / "hive_diversification_governor.csv", index=False
        )
        div_stats = {"mean": float(np.mean(gov)), "min": float(np.min(gov)), "max": float(np.max(gov))}
    summary["diversification_governor"] = div_stats
    (RUNS / "hive_evolution.json").write_text(json.dumps(summary, indent=2))

    # Persistence governor from ecosystem action pressure.
    pres = np.asarray(summary.get("action_pressure_series", []), float)
    if len(pres):
        # More ecosystem actions => reduce risk slightly; smooth for runtime stability.
        pg = np.clip(1.04 - 0.30 * np.clip(pres, 0.0, 1.0), 0.78, 1.04)
        if len(pg) > 1:
            out_pg = pg.copy()
            a = 0.88
            for i in range(1, len(out_pg)):
                out_pg[i] = a * out_pg[i - 1] + (1.0 - a) * out_pg[i]
            pg = np.clip(out_pg, 0.78, 1.04)
        pd.DataFrame({"DATE": governed["DATE"], "hive_persistence_governor": pg}).to_csv(
            RUNS / "hive_persistence_governor.csv", index=False
        )
        summary["persistence_governor"] = {
            "mean": float(np.mean(pg)),
            "min": float(np.min(pg)),
            "max": float(np.max(pg)),
        }
        (RUNS / "hive_evolution.json").write_text(json.dumps(summary, indent=2))

    html = (
        f"<p>Governed hive weights saved to <b>weights_cross_hive_governed.csv</b>.</p>"
        f"<p>Latest governed: {summary.get('latest_governed_weights', {})}</p>"
        f"<p>Events: {len(summary.get('events', []))}; counts={summary.get('event_counts', {})}</p>"
    )
    append_card("Ecosystem Age Governor ✔", html)

    print(f"✅ Wrote {RUNS/'weights_cross_hive_governed.csv'}")
    print(f"✅ Wrote {RUNS/'hive_evolution.json'}")
    if div_stats:
        print(f"✅ Wrote {RUNS/'hive_diversification_governor.csv'}")
    if len(np.asarray(summary.get("action_pressure_series", []), float)):
        print(f"✅ Wrote {RUNS/'hive_persistence_governor.csv'}")
