#!/usr/bin/env python3
# tools/make_reflexive.py
# Runs Reflexive Feedback and writes outputs under runs_plus/.

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmods.reflexive import run_reflexive

if __name__ == "__main__":
    ev, sig, info = run_reflexive()
    weights = {}
    conf = {}
    if not sig.empty and {"ASSET", "DATE", "reflex_signal"}.issubset(sig.columns):
        tmp = sig.sort_values(["ASSET", "DATE"]).groupby("ASSET", as_index=False).tail(1)
        for _, row in tmp.iterrows():
            try:
                weights[str(row["ASSET"]).upper()] = float(row["reflex_signal"])
                conf[str(row["ASSET"]).upper()] = float(row.get("reflex_confidence", abs(row["reflex_signal"])))
            except Exception:
                continue

    if weights:
        mean_abs = sum(abs(v) for v in weights.values()) / max(len(weights), 1)
        mean_conf = sum(conf.values()) / max(len(conf), 1) if conf else mean_abs
        # stronger reflex + high confidence => lower gross risk
        exposure_scaler = max(0.50, min(1.10, float(1.05 - 0.45 * mean_abs * mean_conf)))
    else:
        exposure_scaler = 1.0

    payload = {"weights": weights, "confidence": conf, "exposure_scaler": exposure_scaler}
    (ROOT / "runs_plus" / "reflexive.json").write_text(json.dumps(payload, indent=2))

    print("✅ Wrote runs_plus/reflexive_events.csv")
    print("✅ Wrote runs_plus/reflexive_signal.csv")
    print("✅ Wrote runs_plus/reflexive_summary.json")
    print("✅ Wrote runs_plus/reflexive.json")
