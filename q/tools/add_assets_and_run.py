#!/usr/bin/env python3
"""
add_assets_and_run.py  (auto-normalize + clean)

1) Copies any CSVs from ./data_new/ into ./data/
2) Normalizes all CSVs in ./data/ to two columns: DATE,VALUE
3) Quarantines any bad CSVs to ./data_bad/
4) Runs the full pipeline and refreshes the HTML

Usage:
  - Drop new CSVs into data_new/
  - python tools/add_assets_and_run.py
"""

import shutil, sys, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_NEW = ROOT / "data_new"
DATA     = ROOT / "data"
TOOLS    = ROOT / "tools"

def run(cmd):
    print("▶", " ".join(map(str, cmd)))
    subprocess.check_call(cmd)

if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    DATA_NEW.mkdir(exist_ok=True)

    copied = 0
    for p in sorted(DATA_NEW.glob("*.csv")):
        dst = DATA / p.name
        shutil.copy2(p, dst)
        copied += 1
        print(f"📦 copied {p.name} -> data/")
    if copied == 0:
        print("No CSVs found in data_new/. Put new assets there first.")

    # Always normalize & clean before running the pipeline
    run([sys.executable, str(TOOLS / "normalize_new_csvs.py")])
    run([sys.executable, str(TOOLS / "clean_data_folder.py")])

    # Full pipeline
    run([sys.executable, str(TOOLS / "run_final_best.py")])

    print("\n✅ Done. Open: report_best_plus.html (and report_all.html)")
