#!/usr/bin/env python3
from pathlib import Path
import json, sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"

def exists(p: Path) -> bool:
    try: return p.exists()
    except: return False

def read_json(p: Path, default):
    try:
        if exists(p):
            return json.loads(p.read_text())
    except Exception:
        pass
    return default

def table_print(title, rows, header=None, width=100):
    print("\n" + title)
    print("=" * len(title))
    if not rows:
        print("(no data)"); return
    cols = len(rows[0])
    colw = [0]*cols
    if header:
        for i,h in enumerate(header):
            colw[i] = max(colw[i], len(str(h)))
    for r in rows:
        for i,v in enumerate(r):
            colw[i] = max(colw[i], len(str(v)))
    total = sum(colw) + 3*(cols-1)
    if total > width:
        scale = (width - 3*(cols-1)) / sum(colw)
        colw = [max(6, int(w*scale)) for w in colw]
    if header:
        line = " | ".join(str(h).ljust(colw[i]) for i,h in enumerate(header))
        print(line); print("-"*len(line))
    for r in rows:
        print(" | ".join(str(r[i])[:colw[i]].ljust(colw[i]) for i in range(cols)))

def load_wf_table(name: str):
    p = RUNS / name
    if not exists(p): return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None

def summarize(df: pd.DataFrame):
    out = {"assets": len(df)}
    for key in ["hit","sharpe","maxDD","maxdd","oos_hit"]:
        if key in df.columns:
            s = pd.to_numeric(df[key], errors="coerce").dropna()
            if s.size:
                if key.lower().startswith("hit"):
                    out["hit"] = float(s.mean())
                elif "sharpe" in key.lower():
                    out["sharpe"] = float(s.mean())
                elif "dd" in key.lower():
                    out["maxDD"] = float(s.mean())
    return out

def top_by(df, col="sharpe", n=12):
    if col not in df.columns: return []
    d = df.copy()
    d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.sort_values(col, ascending=False).head(n)
    rows = []
    asset_col = "asset" if "asset" in d.columns else ("symbol" if "symbol" in d.columns else d.columns[0])
    for r in d.itertuples():
        rows.append([getattr(r, asset_col),
                     f"{getattr(r, 'hit', float('nan')):.3f}" if 'hit' in d.columns else "—",
                     f"{getattr(r, 'sharpe', float('nan')):.3f}" if 'sharpe' in d.columns else "—",
                     f"{getattr(r, 'maxDD', float('nan')):.3f}" if 'maxDD' in d.columns else
                     (f"{getattr(r, 'maxdd', float('nan')):.3f}" if 'maxdd' in d.columns else "—")])
    return rows

def main():
    base = load_wf_table("walk_forward_table.csv")
    plus = load_wf_table("walk_forward_table_plus.csv")

    # Baseline summary
    if base is not None:
        s = summarize(base)
        table_print("WALK-FORWARD (BASELINE)", [
            ["Assets", s.get("assets","—")],
            ["Hit (avg)", f"{s.get('hit'):0.3f}" if s.get("hit") is not None else "—"],
            ["Sharpe (avg)", f"{s.get('sharpe'):0.3f}" if s.get("sharpe") is not None else "—"],
            ["MaxDD (avg)", f"{s.get('maxDD'):0.3f}" if s.get("maxDD") is not None else "—"],
        ], header=["Metric","Value"])
        table_print("TOP BY SHARPE (BASELINE)", top_by(base, "sharpe"),
                    header=["Asset","Hit","Sharpe","MaxDD"])

    # WF+ summary
    if plus is not None:
        s = summarize(plus)
        table_print("WALK-FORWARD PLUS (ADD-ONS APPLIED)", [
            ["Assets", s.get("assets","—")],
            ["Hit (avg)", f"{s.get('hit'):0.3f}" if s.get("hit") is not None else "—"],
            ["Sharpe (avg)", f"{s.get('sharpe'):0.3f}" if s.get("sharpe") is not None else "—"],
            ["MaxDD (avg)", f"{s.get('maxDD'):0.3f}" if s.get("maxDD") is not None else "—"],
        ], header=["Metric","Value"])
        table_print("TOP BY SHARPE (WF+)", top_by(plus, "sharpe"),
                    header=["Asset","Hit","Sharpe","MaxDD"])

    # Council / Reflexive / Hive quick looks
    council = read_json(RUNS/"council.json", {"final_weights": {}}).get("final_weights", {})
    c_pairs = sorted(council.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
    table_print("COUNCIL (TOP |WEIGHT|)", [[k, f"{v:+.4f}"] for k,v in c_pairs], header=["Symbol","Weight"])

    reflex = read_json(RUNS/"reflexive.json", {"weights": {}, "exposure_scaler": None})
    r_pairs = sorted((reflex.get("weights") or {}).items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
    esc = reflex.get("exposure_scaler")
    cap = f"(exposure scaler ≈ {esc:.2f})" if isinstance(esc,(int,float)) else ""
    table_print(f"REFLEXIVE TOP 10 {cap}".strip(), [[k, f"{v:+.3f}"] for k,v in r_pairs],
                header=["Symbol","Weight"])

    hive = read_json(RUNS/"hive.json", {"hives": {}}).get("hives", {})
    hrows = [[h, ", ".join(hive[h][:6]) + (" …" if len(hive[h])>6 else "")] for h in sorted(hive)]
    table_print("HIVE / ECOSYSTEM", hrows, header=["Hive","First symbols"])

    # Final path hints
    out_html = ROOT / "report_all.html"
    if exists(out_html):
        print(f"\nReport: {out_html.as_posix()}")

if __name__ == "__main__":
    main()
