#!/usr/bin/env python3
"""
run_prune_retune.py  (v3 strict order)

One button:
  0) Fresh per-asset metrics on FULL set (for unbiased pruning)
  1) Prune (rules-based) -> runs_plus/universe_freeze.json
  2) Park CSVs NOT in the universe (data_park/)
  3) Trim runs_plus/ to universe (runs_plus_park/)
  4) Rebuild WF+ **on pruned set only**
  5) Build portfolio + inject cards
  6) Restore parked CSVs

Nothing is deleted.
"""

import sys, json, shutil, subprocess
from pathlib import Path

ROOT  = Path(__file__).resolve().parents[1]
DATA  = ROOT / "data"
PARK  = ROOT / "data_park"
RUNS  = ROOT / "runs_plus"
TOOLS = ROOT / "tools"

def run(cmd):
    print("▶", " ".join(map(str, cmd)))
    subprocess.check_call(cmd)

def load_universe():
    j = json.loads((RUNS/"universe_freeze.json").read_text())
    return set(j.get("universe", []))

def park_excluded(universe):
    DATA.mkdir(exist_ok=True)
    PARK.mkdir(exist_ok=True)
    moved=[]
    for csv in sorted(DATA.glob("*.csv")):
        sym = csv.stem
        if sym not in universe:
            shutil.move(csv, PARK/csv.name); moved.append(csv.name)
    return moved

def restore_all():
    restored=[]
    if PARK.exists():
        for csv in sorted(PARK.glob("*.csv")):
            shutil.move(csv, DATA/csv.name); restored.append(csv.name)
    return restored

def trim_runs_to_universe(universe):
    park = ROOT/"runs_plus_park"
    park.mkdir(exist_ok=True)
    kept=moved=0
    for d in RUNS.iterdir():
        if not d.is_dir(): 
            continue
        name=d.name
        # skip non-asset/summary files
        if name in {"hive.json","council.json","dna_drift.json","heartbeat.json"}: 
            continue
        if name in {"portfolio_plus.csv","portfolio_summary.json","walk_forward_table_plus.csv","symbolic.json","reflexive.json","hive_council.json","meta_council.json"}:
            continue
        if name in universe:
            kept+=1
        else:
            try:
                shutil.move(str(d), str(park/name)); moved+=1
            except Exception as e:
                print(f"skip {name}: {e}")
    print(f"✅ trimmed runs_plus/: kept={kept}, parked={moved}")

if __name__ == "__main__":
    # 0) Fresh per-asset OOS + WF+ on full set (for pruning metrics)
    run([sys.executable, str(TOOLS/"make_oos_all.py")])
    run([sys.executable, str(TOOLS/"walk_forward_plus.py")])

    # 1) Prune (rules-based)
    run([sys.executable, str(TOOLS/"auto_prune_assets.py")])
    universe = load_universe()
    print(f"Active universe ({len(universe)}):", ", ".join(sorted(universe)))

    # 2) Park CSVs not in universe
    parked = park_excluded(universe)
    print(f"🧳 Parked {len(parked)} CSVs from data/")

    try:
        # 3) Trim runs_plus/ to only active symbols
        trim_runs_to_universe(universe)

        # 4) WF+ on pruned set ONLY
        #    Remove any stale table so we don't read old counts
        (RUNS/"walk_forward_table_plus.csv").unlink(missing_ok=True)
        run([sys.executable, str(TOOLS/"walk_forward_plus.py")])

        # 5) Build portfolio + inject cards
        run([sys.executable, str(TOOLS/"build_portfolio_plus.py")])
        run([sys.executable, str(TOOLS/"add_wfplus_card_any.py")])
        run([sys.executable, str(TOOLS/"add_hive_meta_card.py")])
        run([sys.executable, str(TOOLS/"add_portfolio_card.py")])
        run([sys.executable, str(TOOLS/"add_portfolio_equity_chart.py")])
        run([sys.executable, str(TOOLS/"show_results_cli.py")])
    finally:
        # 6) Restore CSVs
        restored = restore_all()
        print(f"🔁 Restored {len(restored)} CSVs to data/")
        print("✅ Done. Open: report_best_plus.html")
