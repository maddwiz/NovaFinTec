#!/usr/bin/env python3
# tools/add_robust_card.py
# Inserts a simple robustness table (OOS Sharpe / OOS MaxDD vs split) into report HTML.

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"
FILES = [ROOT/"report_best_plus.html", ROOT/"report_all.html"]
START="<!--ROBUST_CARD_START-->"
END  ="<!--ROBUST_CARD_END-->"

def upsert(html, block):
    if START in html and END in html:
        pre = html.split(START)[0]; post = html.split(END,1)[1]
        return pre + block + post
    return html.replace("</body>", block + "\n</body>") if "</body>" in html else html + block

if __name__ == "__main__":
    p = RUNS/"robust_time_splits.csv"
    if not p.exists():
        raise SystemExit("Missing runs_plus/robust_time_splits.csv (run tools/robust_time_splits.py first).")
    df = pd.read_csv(p)

    # Build a tiny HTML table focusing on OOS Sharpe and OOS MaxDD
    cols = [c for c in ["split",
                        "main_oos_sharpe","reg_oos_sharpe","dna_oos_sharpe",
                        "main_oos_maxdd","reg_oos_maxdd","dna_oos_maxdd"]
            if c in df.columns]
    tbl = df[cols].copy()
    tbl["split"] = tbl["split"].map(lambda x: f"{x:0.2f}")
    html_table = tbl.to_html(index=False, border=0, justify="left")

    block = f"""{START}
<section style="border:2px solid #557;padding:14px;margin:18px 0;border-radius:10px">
  <h2 style="margin:0 0 8px 0">Robustness: IS/OOS Split Sensitivity</h2>
  <p style="color:#666;margin:6px 0 12px 0">OOS Sharpe / OOS MaxDD across different IS/OOS splits. We want Regime+DNA to hold roughly steady.</p>
  {html_table}
</section>
{END}"""

    for f in FILES:
        if not f.exists(): 
            print("skip", f.name)
            continue
        f.write_text(upsert(f.read_text(), block))
        print("✅ Upserted ROBUSTNESS card in", f.name)
