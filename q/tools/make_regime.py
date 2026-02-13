#!/usr/bin/env python3
# tools/make_regime.py
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from qmods.regime import run_regime

if __name__ == "__main__":
    feat, wdf, summary = run_regime()
    print("✅ Wrote runs_plus/regime_series.csv")
    print("✅ Wrote runs_plus/regime_weights.csv")
    print("✅ Wrote runs_plus/regime_summary.json")
