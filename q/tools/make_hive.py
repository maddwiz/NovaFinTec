#!/usr/bin/env python3
# tools/make_hive.py
# Runs the Hive / Ecosystem layer and writes outputs to runs_plus/.

from pathlib import Path
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmods.hive import run_hive

if __name__ == "__main__":
    mapping, aset, grp, summary = run_hive()
    runs = ROOT / "runs_plus"

    hive_map = {}
    if not aset.empty and {"ASSET", "HIVE"}.issubset(aset.columns):
        for hive, g in aset.groupby("HIVE"):
            hive_map[str(hive).upper()] = sorted(set(str(x).upper() for x in g["ASSET"].tolist()))
    elif not mapping.empty and {"ASSET", "HIVE"}.issubset(mapping.columns):
        for hive, g in mapping.groupby("HIVE"):
            hive_map[str(hive).upper()] = sorted(set(str(x).upper() for x in g["ASSET"].tolist()))

    council_path = runs / "council.json"
    council = {}
    if council_path.exists():
        try:
            council = json.loads(council_path.read_text()).get("final_weights", {}) or {}
        except Exception:
            council = {}

    top_by_hive = {}
    for hive, members in hive_map.items():
        ranked = sorted(
            [(m, float(council.get(m, 0.0))) for m in members],
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:5]
        top_by_hive[hive] = {sym: w for sym, w in ranked}

    suggested = {}
    w_candidates = [runs / "weights_cross_hive_governed.csv", runs / "weights_cross_hive.csv"]
    w_cross = next((p for p in w_candidates if p.exists()), None)
    if w_cross is not None:
        try:
            wdf = pd.read_csv(w_cross)
            if not wdf.empty:
                last = wdf.iloc[-1].to_dict()
                for k, v in last.items():
                    if str(k).upper() == "DATE":
                        continue
                    try:
                        suggested[str(k).upper()] = float(v)
                    except Exception:
                        continue
        except Exception:
            pass

    hive_payload = {
        "hives": hive_map,
        "top_by_hive": top_by_hive,
        "suggested_weights": suggested,
        "suggested_source": str(w_cross.name) if w_cross is not None else None,
    }
    (runs / "hive.json").write_text(json.dumps(hive_payload, indent=2))

    print("✅ Wrote runs_plus/hive_signals.csv")
    print("✅ Wrote runs_plus/hive_summary.json")
    print("✅ Wrote runs_plus/hive_assets.csv")
    print("✅ Wrote runs_plus/hive.json")
    print("Hives:", ", ".join(summary.get("hives", [])) or "(none)")
