#!/usr/bin/env python3
# tools/patch_regime_weights_with_dna.py
# Reads regime_weights.csv + dna_drift.csv and writes regime_weights_dna.csv
# Idea: as DNA drift rises, attenuate non-main sleeves by factor alpha in [0.6, 1.0].

from pathlib import Path
import pandas as pd
import numpy as np
import json

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"

DRIFT_COL_CANDIDATES = ["drift","dna_drift","drift_pct","drift_z"]

# gentle defaults
ALPHA_MIN = 0.6   # floor for add-on weights
DRIFT_P95 = 0.95   # normalize drift by its 95th percentile
STRENGTH  = 0.6   # attenuation strength (0 = off, 1 = max)

def _find_drift(df):
    for c in df.columns:
        if c == "DATE": continue
        if c.lower() in DRIFT_COL_CANDIDATES:
            return c
    for c in df.columns:
        if c != "DATE" and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

if __name__ == "__main__":
    rw = pd.read_csv(RUNS/"regime_weights.csv", parse_dates=["DATE"])
    # find dna file
    drift_path = None
    for name in ["dna_drift.csv","drift_series.csv","dna_series.csv"]:
        p = RUNS / name
        if p.exists():
            drift_path = p; break
    if drift_path is None:
        raise SystemExit("No dna drift csv found in runs_plus/. Run tools/make_dna_drift.py first.")

    dn = pd.read_csv(drift_path, parse_dates=["DATE"])
    col = _find_drift(dn)
    if col is None:
        raise SystemExit("Could not find a drift column in the DNA csv.")

    # normalize drift to 0..1 via P95
    x = dn[col].astype(float)
    p95 = float(np.nanpercentile(x.dropna(), 95)) if x.notna().sum() > 10 else (x.max() if x.notna().any() else 1.0)
    if not np.isfinite(p95) or p95 <= 0: p95 = 1.0
    dn["drift_norm"] = (x / p95).clip(0.0, 1.0)

    # merge on DATE
    m = rw.merge(dn[["DATE","drift_norm"]], on="DATE", how="left")
    m["drift_norm"] = m["drift_norm"].fillna(0.0)

    # attenuation factor: alpha = 1 - STRENGTH * drift_norm, floored at ALPHA_MIN
    m["alpha"] = (1.0 - STRENGTH * m["drift_norm"]).clip(ALPHA_MIN, 1.0)

    # attenuate non-main sleeves, reallocate to main
    for k in ["w_vol","w_osc","w_sym","w_reflex"]:
        m[f"{k}_dna"] = m[k] * m["alpha"]
    m["w_main_dna"] = 1.0 - (m["w_vol_dna"] + m["w_osc_dna"] + m["w_sym_dna"] + m["w_reflex_dna"])

    out = m[["DATE","regime","w_main_dna","w_vol_dna","w_osc_dna","w_sym_dna","w_reflex_dna","drift_norm","alpha"]].copy()
    out = out.sort_values("DATE").reset_index(drop=True)
    out.to_csv(RUNS/"regime_weights_dna.csv", index=False)

    # summary
    summary = {
        "alpha_min": ALPHA_MIN,
        "strength": STRENGTH,
        "drift_p95_used": p95,
        "note": "Higher DNA drift reduces add-on sleeves; weight re-added to Main."
    }
    (RUNS/"regime_weights_dna_summary.json").write_text(json.dumps(summary, indent=2))

    print("✅ Wrote runs_plus/regime_weights_dna.csv")
    print("Alpha in [%.2f, 1.00]; strength=%.2f; p95=%.4f" % (ALPHA_MIN, STRENGTH, p95))
