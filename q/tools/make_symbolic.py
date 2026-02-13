#!/usr/bin/env python3
# tools/make_symbolic.py
# Runs the Symbolic / Affective ingestion and writes outputs under runs_plus/.

from pathlib import Path
import json
import sys
import pandas as pd

# --- make sure project root is on sys.path so 'qmods' imports work ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmods.symbolic import run_symbolic

if __name__ == "__main__":
    ev, sig, info = run_symbolic()
    latest = {}
    latest_conf = {}
    if not sig.empty and {"ASSET", "DATE", "sym_signal"}.issubset(sig.columns):
        tmp = sig.sort_values(["ASSET", "DATE"]).groupby("ASSET", as_index=False).tail(1)
        for _, row in tmp.iterrows():
            try:
                latest[str(row["ASSET"]).upper()] = float(row["sym_signal"])
                latest_conf[str(row["ASSET"]).upper()] = float(row.get("confidence", abs(row["sym_signal"])))
            except Exception:
                continue
    heads = []
    if not ev.empty and {"DATE", "ASSET", "text", "sent"}.issubset(ev.columns):
        top = ev.sort_values("DATE").tail(200).copy()
        top["abs_sent"] = top["sent"].abs()
        top = top.sort_values("abs_sent", ascending=False).head(20)
        for _, row in top.iterrows():
            heads.append(
                {
                    "date": str(pd.to_datetime(row["DATE"]).date()),
                    "source": "symbolic_ingest",
                    "title": str(row.get("text", ""))[:160],
                    "score": float(row.get("sent", 0.0)),
                    "asset": str(row.get("ASSET", "ALL")).upper(),
                    "url": "",
                }
            )
    payload = {"symbolic": latest, "confidence": latest_conf, "headlines": heads}
    (ROOT / "runs_plus" / "symbolic.json").write_text(json.dumps(payload, indent=2))

    print("✅ Wrote runs_plus/symbolic_events.csv")
    print("✅ Wrote runs_plus/symbolic_signal.csv")
    print("✅ Wrote runs_plus/symbolic_summary.json")
    print("✅ Wrote runs_plus/symbolic.json")
    print("Top words:", list((info.get("top_words") or {}).keys())[:10])
