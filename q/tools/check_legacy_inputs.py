#!/usr/bin/env python3
"""
Legacy Input Check — robust CSV loader that ignores DATE columns and headers.
It prints which legacy files exist and the first/last few numeric values so you
know they parsed correctly.
"""

from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"

def load_series_robust(rel):
    """
    Load a 1-D numeric series from CSV.
    - Skips header if present.
    - If there's a DATE column (or any non-numeric columns), it takes the LAST
      numeric column on each row.
    - Returns np.ndarray or None.
    """
    p = ROOT / rel
    if not p.exists():
        return None

    # Read all lines safely
    raw = p.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    if not raw:
        return None

    # If first line contains letters, treat as header and skip it
    start = 1 if any(c.isalpha() for c in raw[0]) else 0

    vals = []
    for ln in raw[start:]:
        if not ln.strip():
            continue
        cols = [c.strip() for c in ln.split(",") if c.strip() != ""]
        if not cols:
            continue
        # Try numeric from the RIGHT (last column most likely numeric)
        v = None
        for tok in reversed(cols):
            try:
                v = float(tok)
                break
            except ValueError:
                continue
        if v is not None:
            vals.append(v)
        # If no numeric token on the row, skip it silently

    if not vals:
        return None
    return np.asarray(vals, dtype=float)

def brief(name, arr):
    if arr is None:
        return f"{name:<24} : MISSING"
    n = len(arr)
    tail = ", ".join(f"{x:.6f}" for x in arr[-3:]) if n >= 3 else ", ".join(f"{x:.6f}" for x in arr)
    head = ", ".join(f"{x:.6f}" for x in arr[:3]) if n >= 3 else ", ".join(f"{x:.6f}" for x in arr)
    return f"{name:<24} : OK  (len={n})\n  head: {head}\n  tail: {tail}"

if __name__ == "__main__":
    print("=== Legacy Input Check (robust) ===")

    drift = load_series_robust("runs_plus/dna_drift.csv")
    bpm   = load_series_robust("runs_plus/heartbeat_bpm.csv")
    sym   = load_series_robust("runs_plus/symbolic_latent.csv")
    refx  = load_series_robust("runs_plus/reflex_latent.csv")

    print(brief("dna_drift.csv", drift))
    print(brief("heartbeat_bpm.csv", bpm))
    print(brief("symbolic_latent.csv", sym))
    print(brief("reflex_latent.csv", refx))

    if all(x is None for x in [drift, bpm, sym, refx]):
        print("\n(!) No legacy inputs found or they could not be parsed. "
              "Make sure those CSVs exist in runs_plus/ and that the numeric "
              "signal is in the LAST column (date can be first).")
    else:
        print("\n✅ Parsing looks good. You can run:  python tools/tune_legacy_knobs.py")
