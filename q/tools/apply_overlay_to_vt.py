#!/usr/bin/env python3
# tools/apply_overlay_to_vt.py
# Multiplies Vol-Target daily returns by overlay_alpha (DATE-joined, forward-filled),
# bounds again to be safe. Produces VT+Overlay track and summary.
#
# Writes:
#   runs_plus/portfolio_plus_vt_overlay.csv
#   runs_plus/final_portfolio_vt_overlay_summary.json

from pathlib import Path
import pandas as pd, numpy as np, json

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"

# safety bounds in case overlay_alpha is missing early on
ALPHA_MIN = 0.70
ALPHA_MAX = 1.30
IS_FRAC   = 0.75
DAYS      = 252.0

def _ann_sharpe(r):
    s = pd.Series(r).replace([np.inf,-np.inf], np.nan).dropna()
    if s.empty: return 0.0
    sd = s.std()
    if not np.isfinite(sd) or sd == 0: return 0.0
    return float((s.mean()/sd)*np.sqrt(DAYS))
def _maxdd(r):
    s = pd.Series(r).fillna(0.0)
    eq = (1.0 + s).cumprod()
    dd = eq/eq.cummax() - 1.0
    return float(dd.min())
def _hit(r):
    s = pd.Series(r).dropna()
    return float((s>0).mean()) if not s.empty else 0.0
def _split(vals, frac=IS_FRAC):
    n=len(vals); k=int(n*frac); return vals[:k], vals[k:]

if __name__ == "__main__":
    p_vt = RUNS/"portfolio_plus_vt.csv"
    p_ov = RUNS/"overlay_alpha.csv"
    if not p_vt.exists():
        raise SystemExit("Missing runs_plus/portfolio_plus_vt.csv (build vol-target first).")
    if not p_ov.exists():
        raise SystemExit("Missing runs_plus/overlay_alpha.csv (run tools/make_overlay_alpha.py first).")

    vt = pd.read_csv(p_vt)
    dcol = None
    for c in vt.columns:
        if str(c).lower() in ("date","timestamp"):
            dcol = c; break
    if dcol is None and "DATE" in vt.columns: dcol = "DATE"
    if dcol is None:
        raise SystemExit("portfolio_plus_vt.csv needs a DATE column.")

    vt[dcol] = pd.to_datetime(vt[dcol], errors="coerce")
    vt = vt.dropna(subset=[dcol]).sort_values(dcol)

    alpha = pd.read_csv(p_ov)
    acol = "DATE" if "DATE" in alpha.columns else next((c for c in alpha.columns if str(c).lower() in ("date","timestamp")), None)
    if acol is None:
        raise SystemExit("overlay_alpha.csv needs DATE column.")
    alpha[acol] = pd.to_datetime(alpha[acol], errors="coerce")
    alpha = alpha.dropna(subset=[acol]).sort_values(acol).rename(columns={acol:"DATE"})

    # join & ffill back 10 days to cover alignment gaps
    merged = vt.rename(columns={dcol:"DATE"}).merge(alpha, on="DATE", how="left")
    merged["overlay_alpha"] = merged["overlay_alpha"].ffill(limit=10).fillna(1.0).clip(ALPHA_MIN, ALPHA_MAX)

    r = pd.to_numeric(merged.get("ret_vt", merged.get("ret", pd.Series(0.0))), errors="coerce").fillna(0.0).clip(-0.5,0.5)
    a = pd.to_numeric(merged["overlay_alpha"], errors="coerce").fillna(1.0).clip(ALPHA_MIN, ALPHA_MAX)
    r_plus = (r * a).values

    out = pd.DataFrame({
        "DATE": merged["DATE"],
        "ret_vt_overlay": r_plus,
        "eq_vt_overlay": (1.0 + pd.Series(r_plus)).cumprod()
    })
    out.to_csv(RUNS/"portfolio_plus_vt_overlay.csv", index=False)

    r_is, r_oos = _split(r_plus)
    summary = {
        "in_sample":  {"sharpe": _ann_sharpe(r_is),  "hit": _hit(r_is),  "maxdd": _maxdd(r_is)},
        "out_sample": {"sharpe": _ann_sharpe(r_oos), "hit": _hit(r_oos), "maxdd": _maxdd(r_oos)},
        "note": "Vol-Target scaled by breadth+calmness overlay."
    }
    (RUNS/"final_portfolio_vt_overlay_summary.json").write_text(json.dumps(summary, indent=2))
    print("✅ Wrote portfolio_plus_vt_overlay.csv and final_portfolio_vt_overlay_summary.json")
    print("OOS Sharpe=%.3f  OOS MaxDD=%.3f" % (summary["out_sample"]["sharpe"], summary["out_sample"]["maxdd"]))
