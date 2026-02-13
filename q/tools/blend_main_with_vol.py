#!/usr/bin/env python3
# tools/blend_main_with_vol.py
from pathlib import Path
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"

def smart_load_returns(path: Path, prefer="auto"):
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit(f"{path.name} is empty")

    lowers = {c.lower(): c for c in df.columns}
    dcol = lowers.get("date") or lowers.get("timestamp") or lowers.get("time") or df.columns[0]
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).sort_values(dcol)

    # MANY possible return column names (added port_ret)
    cands = []
    if prefer == "net":
        cands += ["ret_net"]
    cands += ["ret", "ret_gross", "return", "pnl", "pnl_plus", "daily_ret",
              "portfolio_ret", "port_ret"]

    r = None
    for c in cands:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 5:
                r = s.fillna(0.0); break

    if r is None:
        # derive from equity columns (added port_equity)
        for c in ["eq", "equity", "equity_curve", "portfolio_eq", "equity_index", "port_equity"]:
            if c in df.columns:
                eq = pd.to_numeric(df[c], errors="coerce")
                if eq.notna().sum() > 5:
                    r = eq.pct_change().fillna(0.0); break

    if r is None:
        raise SystemExit(f"Could not find returns in {path.name}")

    return pd.DataFrame({"DATE": df[dcol], "ret": r})

def ann_sharpe(r):
    r = pd.Series(r).dropna()
    s = r.std()
    if s == 0 or pd.isna(s): return 0.0
    return float((r.mean() / s) * (252.0 ** 0.5))

if __name__ == "__main__":
    main_p = RUNS / "portfolio_plus.csv"
    vol_p  = RUNS / "vol_overlay_costed.csv"
    if not main_p.exists(): raise SystemExit("Missing runs_plus/portfolio_plus.csv")
    if not vol_p.exists():  raise SystemExit("Missing runs_plus/vol_overlay_costed.csv")

    a = smart_load_returns(main_p, prefer="auto")  # main portfolio (your core)
    b = smart_load_returns(vol_p,  prefer="net")   # vol overlay (use net)

    df = pd.merge(a, b, on="DATE", how="inner", suffixes=("_main", "_vol")).sort_values("DATE")
    if df.empty:
        raise SystemExit("No overlapping dates between main and vol.")

    best = {"alpha": None, "sharpe": -9, "hit": 0, "maxdd": 0}
    rows = []
    for i in range(21):  # alpha 0.00..1.00
        alpha = i / 20.0
        r = alpha * df["ret_main"] + (1 - alpha) * df["ret_vol"]
        eq = (1 + r).cumprod()
        sh = ann_sharpe(r)
        hit = float((r > 0).mean())
        dd = float((eq / eq.cummax() - 1).min())
        rows.append((alpha, sh, hit, dd))
        if sh > best["sharpe"]:
            best = {"alpha": float(alpha), "sharpe": float(sh), "hit": hit, "maxdd": dd}

    out = pd.DataFrame({
        "DATE": df["DATE"],
        "ret": best["alpha"] * df["ret_main"] + (1 - best["alpha"]) * df["ret_vol"],
        "alpha": best["alpha"]
    })
    out["eq"] = (1 + out["ret"]).cumprod()
    out.to_csv(RUNS / "blend_main_vol.csv", index=False)
    (RUNS / "blend_main_vol_summary.json").write_text(json.dumps(best, indent=2))

    print(" alpha   Sharpe    Hit   MaxDD")
    for a_, sh, h, dd in rows:
        print(f" {a_:>4.2f}   {sh:>6.3f}  {h:>5.3f}  {dd:>6.3f}")
    print("\nBEST:", best)
    print("Saved: runs_plus/blend_main_vol.csv and blend_main_vol_summary.json")
