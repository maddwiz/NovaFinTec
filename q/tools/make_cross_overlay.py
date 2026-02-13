#!/usr/bin/env python3
# tools/make_cross_overlay.py
# Runs Cross-Domain Dream Overlays and writes outputs under runs_plus/.

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmods.cross_overlay import run_cross_overlay

if __name__ == "__main__":
    fin, ext, res, meta = run_cross_overlay()
    print("✅ Wrote runs_plus/cross_finance_idx.csv")
    print("✅ Wrote runs_plus/cross_external_idx.csv")
    print("✅ Wrote runs_plus/cross_overlay.csv")
    print("✅ Wrote runs_plus/cross_overlay_summary.json")
    print("Domains:", ", ".join(meta.get("domains", [])) or "(none)")
