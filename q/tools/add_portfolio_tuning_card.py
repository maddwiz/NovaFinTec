#!/usr/bin/env python3
"""
add_portfolio_tuning_card.py

- Reads runs_plus/portfolio_tuning.csv
- Shows the TOP 10 rows by Sharpe (then MaxDD, then Hit) as a small table
- Inserts or refreshes a card in both reports
"""

from pathlib import Path
import pandas as pd
import html

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
FILES = [ROOT/"report_all.html", ROOT/"report_best_plus.html"]

CARD_START = "<!--PORTFOLIO_TUNING_CARD_START-->"
CARD_END   = "<!--PORTFOLIO_TUNING_CARD_END-->"

def upsert(html_in: str, block: str) -> str:
    if CARD_START in html_in and CARD_END in html_in:
        pre = html_in.split(CARD_START)[0]
        post = html_in.split(CARD_END, 1)[1]
        return pre + block + post
    if "</body>" in html_in:
        return html_in.replace("</body>", block + "\n</body>")
    return html_in + "\n" + block

def main():
    p = RUNS / "portfolio_tuning.csv"
    if not p.exists():
        raise SystemExit("portfolio_tuning.csv not found. Run tune_portfolio.py first.")
    df = pd.read_csv(p)
    df = df.dropna(subset=["sharpe"]).copy()
    df["dd_score"] = df["maxDD"]
    df = df.sort_values(["sharpe","dd_score","hit"], ascending=[False, False, False]).head(10)

    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"<tr><td>{r['CAP_PER']:.3f}</td><td>{r['HIVE_CAP']:.2f}</td><td>{r['LOOKBACK']:.0f}</td>"
            f"<td>{r['COST_BPS']:.1f}</td><td style='text-align:right'>{r['sharpe']:.3f}</td>"
            f"<td style='text-align:right'>{r['maxDD']:.3f}</td><td style='text-align:right'>{r['hit']:.3f}</td></tr>"
        )
    table = "\n".join(rows)
    block = f"""{CARD_START}
<section style="border:1px solid #ddd;padding:12px;margin:16px 0;border-radius:8px">
  <h2 style="margin:0 0 8px 0">Portfolio Tuning Results (Top 10)</h2>
  <p style="margin:4px 0 8px 0;color:#666">Source: runs_plus/portfolio_tuning.csv</p>
  <table style="width:100%;border-collapse:collapse;font-size:14px">
    <thead>
      <tr>
        <th style="text-align:left">CAP_PER</th>
        <th style="text-align:left">HIVE_CAP</th>
        <th style="text-align:left">LOOKBACK</th>
        <th style="text-align:left">COST_BPS</th>
        <th style="text-align:right">Sharpe</th>
        <th style="text-align:right">MaxDD</th>
        <th style="text-align:right">Hit</th>
      </tr>
    </thead>
    <tbody>
      {table}
    </tbody>
  </table>
</section>
{CARD_END}"""

    for f in FILES:
        if not f.exists():
            print(f"skip {f.name}: not found")
            continue
        html_in = f.read_text()
        f.write_text(upsert(html_in, block))
        print(f"✅ Upserted Portfolio Tuning card in {f.name}")

if __name__ == "__main__":
    main()
