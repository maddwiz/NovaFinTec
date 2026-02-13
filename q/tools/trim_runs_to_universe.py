#!/usr/bin/env python3
"""
trim_runs_to_universe.py

Keeps only runs_plus/<symbol>/ that are in runs_plus/universe_freeze.json.
Moves non-universe folders into runs_plus_park/ (not deleted).
Run this AFTER auto_prune_assets.py and BEFORE walk_forward_plus.py.
"""

from pathlib import Path
import json, shutil

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
PARK = ROOT / "runs_plus_park"

def main():
    ufile = RUNS / "universe_freeze.json"
    if not ufile.exists():
        raise SystemExit("universe_freeze.json not found. Run auto_prune_assets.py first.")
    uni = json.loads(ufile.read_text()).get("universe", [])
    active = set(uni)

    PARK.mkdir(exist_ok=True)
    kept, moved = 0, 0
    for d in RUNS.iterdir():
        if not d.is_dir(): 
            continue
        name = d.name
        # skip non-asset folders
        if name in {"hive.json","council.json","dna_drift.json","heartbeat.json"}:
            continue
        if name in {"portfolio_plus.csv","portfolio_summary.json","walk_forward_table_plus.csv"}:
            continue
        # keep active symbols, move the rest
        if name in active:
            kept += 1
            continue
        # move if it looks like an asset folder (has oos/oos_plus or files)
        try:
            shutil.move(str(d), str(PARK / name))
            moved += 1
        except Exception as e:
            print(f"skip {name}: {e}")
    print(f"✅ trimmed runs_plus/: kept={kept}, parked={moved}")

if __name__ == "__main__":
    main()
