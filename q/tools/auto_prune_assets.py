#!/usr/bin/env python3
"""
auto_prune_assets.py

Rules-based universe reconstitution from WF+ results.

Rule (pre-declared):
- Drop if WF+ per-asset Sharpe < DROP_TH (default 0.20)
- Re-add only if Sharpe >= ADD_TH (default 0.30)
- Limit total changes to MAX_CHG_PCT of universe (default 0.10)

Inputs:
  runs_plus/*/oos_plus.csv     (per-asset WF+)
Optional:
  runs_plus/universe_freeze.json  (previous universe to apply hysteresis)

Outputs:
  runs_plus/universe_freeze.json  (new universe after reconstitution)
  runs_plus/prune_report.json     (what changed and why)
"""

from pathlib import Path
import json, math
import pandas as pd
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
FREEZE = RUNS / "universe_freeze.json"
PRUNE_REPORT = RUNS / "prune_report.json"

def read_threshold_file(name, default):
    p = Path(name)
    try:
        if p.exists():
            return float(p.read_text().strip())
    except Exception:
        pass
    return float(default)

DROP_TH = read_threshold_file("DROP_TH", 0.20)
ADD_TH  = read_threshold_file("ADD_TH", 0.30)
MAX_CHG_PCT = 0.10  # max fraction of universe changed per cycle

def per_asset_sharpes():
    sharpes = {}
    for d in RUNS.iterdir():
        if not d.is_dir(): continue
        p = d / "oos_plus.csv"
        if not p.exists(): continue
        try:
            df = pd.read_csv(p, parse_dates=["date"])
        except Exception:
            continue
        if "pnl_plus" not in df.columns: 
            continue
        s = df["pnl_plus"].dropna()
        if len(s) < 10: 
            continue
        sh = (s.mean() / (s.std() + 1e-9)) * (252 ** 0.5)
        sharpes[d.name] = float(sh)
    return sharpes

def main():
    sharpes = per_asset_sharpes()
    if not sharpes:
        raise SystemExit("No per-asset WF+ found. Run walk_forward_plus.py first.")

    prev = set()
    if FREEZE.exists():
        try:
            prev = set(json.loads(FREEZE.read_text()).get("universe", []))
        except Exception:
            prev = set()

    all_syms = set(sharpes.keys()) | prev
    # Start from previous universe if exists; else start from symbols above ADD_TH
    current = set(prev) if prev else set([s for s,v in sharpes.items() if v >= ADD_TH])

    # Candidates
    to_drop = sorted([s for s in current if sharpes.get(s, -9) < DROP_TH])
    to_add  = sorted([s for s in all_syms if sharpes.get(s, -9) >= ADD_TH and s not in current])

    # Enforce change cap
    max_changes = max(1, math.floor(MAX_CHG_PCT * max(1, len(all_syms))))
    proposed = to_drop + to_add
    if len(proposed) > max_changes:
        drops_ranked = sorted(to_drop, key=lambda s: sharpes.get(s, -9))
        adds_ranked  = sorted(to_add, key=lambda s: -sharpes.get(s, -9))
        trimmed = (drops_ranked + adds_ranked)[:max_changes]
        to_drop = [s for s in trimmed if s in to_drop]
        to_add  = [s for s in trimmed if s in to_add]

    new_universe = (current - set(to_drop)) | set(to_add)

    FREEZE.write_text(json.dumps({
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "rule": {"DROP_TH": DROP_TH, "ADD_TH": ADD_TH, "MAX_CHG_PCT": MAX_CHG_PCT},
        "universe": sorted(new_universe)
    }, indent=2))

    PRUNE_REPORT.write_text(json.dumps({
        "counts": {
            "prev": len(prev),
            "now": len(new_universe),
            "dropped": len(to_drop),
            "added": len(to_add)
        },
        "dropped": [{ "symbol": s, "sharpe": sharpes.get(s, None)} for s in to_drop],
        "added":   [{ "symbol": s, "sharpe": sharpes.get(s, None)} for s in to_add]
    }, indent=2))

    print("✅ Universe reconstituted.")
    print(f"Prev={len(prev)} → Now={len(new_universe)} | Dropped={len(to_drop)} Added={len(to_add)}")
    print(f"Rules: DROP_TH={DROP_TH}  ADD_TH={ADD_TH}  MAX_CHG_PCT={MAX_CHG_PCT}")

if __name__ == "__main__":
    main()
