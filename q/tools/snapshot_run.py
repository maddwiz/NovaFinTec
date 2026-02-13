#!/usr/bin/env python3
# tools/snapshot_run.py
# Saves key outputs into snapshots/YYYYMMDD_HHMMSS/

from pathlib import Path
import datetime, json, shutil

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"
STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = ROOT/f"snapshots/{STAMP}"
OUT.mkdir(parents=True, exist_ok=True)

KEEP = [
    "walk_forward_table.csv",
    "portfolio_plus.csv",
    "portfolio_weights.csv",
    "final_portfolio_summary.json",
    "final_portfolio_regime_summary.json",
    "final_portfolio_regime_dna_summary.json",
    "regime_weights.csv",
    "regime_weights_dna.csv",
    "sweep_portfolio_knobs.csv",
    "sweep_regime_small.csv"
]

for k in KEEP:
    p = RUNS/k
    if p.exists():
        (OUT/k).parent.mkdir(parents=True, exist_ok=True)
        (OUT/k).write_bytes(p.read_bytes())

print("✅ Snapshot saved to", OUT)
