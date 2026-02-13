#!/usr/bin/env python3
# tools/revert_to_main.py
# No changes to your main; just re-add the standard cards so the report looks normal.

from pathlib import Path
import subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
def run(cmd):
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(r.returncode)

if __name__ == "__main__":
    # rebuild standard cards to ensure report focuses on Main/Regime/DNA
    run(["python","tools/add_regime_final_card_triple.py"])
    run(["python","tools/add_robust_card.py"])
    try:
        run(["python","tools/add_weights_card.py"])
    except Exception:
        pass
    print("✅ Report set back to Main + Regime + DNA cards.")
