#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import html

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
INP  = ROOT / "report_all.html"
OUT  = ROOT / "report_all.html"  # in-place

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
        d["hit"]=pd.to_numeric(df.get("hit"), errors="coerce").mean() if "hit" in df.columns else None
        d["sh"]=pd.to_numeric(df.get("sharpe"), errors="coerce").mean() if "sharpe" in df.columns else None
        mcol="maxDD" if "maxDD" in df.columns else ("maxdd" if "maxdd" in df.columns else None)
        d["dd"]=pd.to_numeric(df.get(mcol), errors="coerce").mean() if mcol else None
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

if __name__ == "__main__":
    if not exists(INP):
        raise SystemExit("report_all.html not found. Run tools/run_all_plus.py first.")
    card = make_card()
    if not card:
        raise SystemExit("No walk_forward_table_plus.csv to add.")
    html_in = INP.read_text()
    html_out = insert_before_body(html_in, card)
    OUT.write_text(html_out)
    print("✅ Injected WF+ card into report_all.html")
