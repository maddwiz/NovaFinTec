#!/usr/bin/env python3
"""
make_hive_council.py

Reads:
  runs_plus/hive.json         -> {"hives": {"H1":[...], "H2":[...], ...}}
  runs_plus/council.json      -> {"final_weights": {"SPY": w, ...}}

Computes:
  - Per-hive council weights (hive_weights)
    * hive raw = mean(member council weights)
    * normalize L1 to 1.0 across hives
    * cap each hive |weight| to HIVE_CAP (default 0.35), renormalize L1<=1
  - Meta-council asset weights (asset_weights)
    * distribute each hive's weight to members proportional to |member council weight|
    * if all zero in a hive, split equally

Writes:
  runs_plus/hive_council.json  -> {"hive_weights": {...}, "meta": {"asset_weights": {...}}}
  runs_plus/meta_council.json   -> {"asset_weights": {...}}
"""
import json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
HIVE_CAP = 0.35  # abs cap per hive

def read_json(p, default):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default

def l1_norm(d):
    return sum(abs(v) for v in d.values()) or 0.0

def clip_and_renorm(hive_w, cap=HIVE_CAP):
    # clip abs to cap
    hw = {k: max(-cap, min(cap, v)) for k,v in hive_w.items()}
    s = l1_norm(hw)
    if s > 1.0 and s > 0:
        hw = {k: v / s for k,v in hw.items()}
    return hw

def main():
    hive = read_json(RUNS / "hive.json", {"hives": {}}).get("hives", {})
    council = read_json(RUNS / "council.json", {"final_weights": {}}).get("final_weights", {})

    if not hive:
        raise SystemExit("No runs_plus/hive.json found or empty.")
    if not council:
        print("(!) council.json had no weights; proceeding with equal splits.")

    # 1) per-hive raw = mean of member council weights
    hive_raw = {}
    for h, members in hive.items():
        vals = [council.get(sym, 0.0) for sym in members]
        hive_raw[h] = (sum(vals) / len(vals)) if members else 0.0

    # 2) normalize L1 to 1 across hives (if non-zero), then cap and renorm
    s = l1_norm(hive_raw)
    hive_norm = {k: (v / s if s > 0 else 0.0) for k,v in hive_raw.items()}
    hive_weights = clip_and_renorm(hive_norm, cap=HIVE_CAP)

    # 3) distribute hive weights to assets
    asset_weights = {}
    for h, members in hive.items():
        if not members:
            continue
        abs_sum = sum(abs(council.get(sym, 0.0)) for sym in members)
        if abs_sum == 0:
            # equal split if no council signal
            share = hive_weights[h] / len(members)
            for sym in members:
                asset_weights[sym] = share
        else:
            for sym in members:
                frac = abs(council.get(sym, 0.0)) / abs_sum
                # keep member's original sign to preserve direction
                sign = 1.0 if council.get(sym, 0.0) >= 0 else -1.0
                asset_weights[sym] = hive_weights[h] * frac * sign

    # Write outputs
    out_a = RUNS / "hive_council.json"
    out_b = RUNS / "meta_council.json"
    out_a.write_text(json.dumps({"hive_weights": hive_weights, "meta": {"asset_weights": asset_weights}}, indent=2))
    out_b.write_text(json.dumps({"asset_weights": asset_weights}, indent=2))

    print(f"✅ Wrote {out_a.as_posix()}")
    print(f"✅ Wrote {out_b.as_posix()}")

if __name__ == "__main__":
    main()
