#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
FILES = [ROOT/"report_all.html", ROOT/"report_best_plus.html"]
CARD_START="<!--META_TUNING_CARD_START-->"
CARD_END="<!--META_TUNING_CARD_END-->"

def upsert(html_in, block):
    if CARD_START in html_in and CARD_END in html_in:
        pre = html_in.split(CARD_START)[0]
        post = html_in.split(CARD_END,1)[1]
        return pre+block+post
    return html_in.replace("</body>", block+"\n</body>") if "</body>" in html_in else html_in+block

if __name__=="__main__":
    p = RUNS / "meta_tuning.csv"
    if not p.exists(): raise SystemExit("meta_tuning.csv not found. Run tune_meta.py first.")
    df = pd.read_csv(p).dropna(subset=["sharpe"]).copy()
    df["dd_score"]=df["maxDD"]
    df=df.sort_values(["sharpe","dd_score","hit"], ascending=[False,False,False]).head(10)
    rows=[]
    for _,r in df.iterrows():
        rows.append(f"<tr><td>{r['META_STRENGTH']:.2f}</td><td>{r['META_SIGN_THRESH']:.2f}</td><td>{int(r['META_REQUIRE_AGREE'])}</td><td style='text-align:right'>{r['sharpe']:.3f}</td><td style='text-align:right'>{r['maxDD']:.3f}</td><td style='text-align:right'>{r['hit']:.3f}</td></tr>")
    table="\n".join(rows)
    block=f"""{CARD_START}
<section style="border:1px solid #ddd;padding:12px;margin:16px 0;border-radius:8px">
  <h2 style="margin:0 0 8px 0">Meta Tuning Results (Top 10)</h2>
  <table style="width:100%;border-collapse:collapse;font-size:14px">
    <thead><tr><th>META_STRENGTH</th><th>META_SIGN_THRESH</th><th>AGREE</th><th style="text-align:right">Sharpe</th><th style="text-align:right">MaxDD</th><th style="text-align:right">Hit</th></tr></thead>
    <tbody>{table}</tbody>
  </table>
</section>
{CARD_END}"""
    for f in FILES:
        if not f.exists(): 
            print(f"skip {f.name}: not found"); 
            continue
        f.write_text(upsert(f.read_text(), block))
        print(f"✅ Upserted Meta Tuning card in {f.name}")
