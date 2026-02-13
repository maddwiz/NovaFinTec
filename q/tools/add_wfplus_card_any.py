#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import html

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"

def exists(p: Path) -> bool:
    try: return p.exists()
    except: return False

def load_tbl(name):
    p = RUNS / name
    if not exists(p): return None
    try: return pd.read_csv(p)
    except Exception: return None

def insert_before_body(html_in: str, block: str) -> str:
    return html_in.replace("</body>", block + "\n</body>") if "</body>" in html_in else html_in + block

def make_card():
    base = load_tbl("walk_forward_table.csv")
    plus = load_tbl("walk_forward_table_plus.csv")
    if plus is None: return ""
    def summ(df):
        d={}
        if df is None: return {"assets":"—","hit":"—","sh":"—","dd":"—"}
        d["assets"]=len(df)
        def mean(col):
            if col not in df.columns: return None
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            return float(s.mean()) if len(s) else None
        d["hit"]=mean("hit") if "hit" in df.columns else None
        d["sh"]=mean("sharpe") if "sharpe" in df.columns else None
        d["dd"]=mean("maxDD") if "maxDD" in df.columns else (mean("maxdd") if "maxdd" in df.columns else None)
        return d
    sb, sp = summ(base), summ(plus)
    def fmt(x): return ("—" if x is None else f"{x:.3f}")
    rows = [
        ("Assets", sp["assets"], sb["assets"]),
        ("Hit (avg)", fmt(sp["hit"]), fmt(sb["hit"])),
        ("Sharpe (avg)", fmt(sp["sh"]), fmt(sb["sh"])),
        ("MaxDD (avg)", fmt(sp["dd"]), fmt(sb["dd"])),
    ]
    table_rows = "\n".join(
        f"<tr><td>{html.escape(k)}</td><td style='text-align:right'>{v}</td><td style='text-align:right'>{b}</td></tr>"
        for k,v,b in rows
    )
    return f"""
<section style="border:1px solid #ddd;padding:12px;margin:16px 0;border-radius:8px">
  <h2 style="margin:0 0 8px 0">Walk-Forward PLUS (with add-ons)</h2>
  <p style="margin:4px 0 8px 0;color:#666">Baseline vs Add-ons applied. Source: runs_plus/walk_forward_table*.csv</p>
  <table style="width:100%;border-collapse:collapse;font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;font-size:14px">
    <thead><tr><th style="text-align:left">Metric</th><th style="text-align:right">WF+</th><th style="text-align:right">Baseline</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
</section>
"""

def inject_into(path: Path, card: str):
    if not exists(path): 
        print(f"skip {path.name}: not found"); 
        return
    if not card:
        print("skip: no WF+ table found to inject")
        return
    html_in = path.read_text()
    if "Walk-Forward PLUS" in html_in:
        print(f"already has WF+ card: {path.name}")
        return
    path.write_text(insert_before_body(html_in, card))
    print(f"✅ Injected WF+ card into {path.name}")

if __name__ == "__main__":
    card = make_card()
    inject_into(ROOT / "report_all.html", card)
    inject_into(ROOT / "report_best_plus.html", card)
