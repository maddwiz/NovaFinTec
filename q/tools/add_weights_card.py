#!/usr/bin/env python3
# tools/add_weights_card.py
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
FILES = [ROOT / "report_best_plus.html", ROOT / "report_all.html"]
CARD_START="<!--WEIGHTS_CARD_START-->"
CARD_END="<!--WEIGHTS_CARD_END-->"

def upsert(html, block):
    if CARD_START in html and CARD_END in html:
        pre = html.split(CARD_START)[0]; post = html.split(CARD_END,1)[1]
        return pre + block + post
    return html.replace("</body>", block + "\n</body>") if "</body>" in html else html + block

if __name__ == "__main__":
    p = RUNS/"portfolio_weights.csv"
    if not p.exists():
        raise SystemExit("Missing runs_plus/portfolio_weights.csv")
    df = pd.read_csv(p).sort_values("weight", ascending=False).reset_index(drop=True)
    # HTML table
    rows = "\n".join(
        f"<tr><td>{a}</td><td style='text-align:right'>{w:0.4f}</td></tr>"
        for a, w in zip(df["asset"], df["weight"])
    )
    block = f"""{CARD_START}
<section style="border:2px solid #999;padding:14px;margin:18px 0;border-radius:10px">
  <h2 style="margin:0 0 8px 0">Portfolio Weights (Top {len(df)})</h2>
  <table style="width:100%;border-collapse:collapse">
    <thead><tr><th style="text-align:left">Asset</th><th style="text-align:right">Weight</th></tr></thead>
    <tbody>
      {rows}
    </tbody>
  </table>
  <p style="color:#666;margin-top:6px">Equal-weight with per-asset cap, re-normalized.</p>
</section>
{CARD_END}"""
    for f in FILES:
        if not f.exists():
            print("skip", f.name); 
            continue
        f.write_text(upsert(f.read_text(), block))
        print("✅ Upserted WEIGHTS card in", f.name)
