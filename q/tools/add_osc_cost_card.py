#!/usr/bin/env python3
# tools/add_osc_cost_card.py
from pathlib import Path
import json as _json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
FILES = [ROOT / "report_best_plus.html", ROOT / "report_all.html"]
CARD_START="<!--OSC_COST_CARD_START-->"
CARD_END="<!--OSC_COST_CARD_END-->"

def upsert(html, block):
    if CARD_START in html and CARD_END in html:
        pre = html.split(CARD_START)[0]; post = html.split(CARD_END,1)[1]
        return pre + block + post
    return html.replace("</body>", block + "\n</body>") if "</body>" in html else html + block

if __name__ == "__main__":
    s = RUNS / "osc_portfolio_costed_summary.json"
    c = RUNS / "osc_portfolio_costed.csv"
    if not s.exists() or not c.exists():
        raise SystemExit("Run run_osc_portfolio_costed.py first.")
    meta = _json.loads(s.read_text())
    df = pd.read_csv(c, parse_dates=["DATE"])

    png = RUNS / "osc_costed_eq.png"
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(df["DATE"], df["eq_gross"], label="Gross")
        plt.plot(df["DATE"], df["eq_net"],   label="Net")
        plt.legend()
        plt.title("Oscillator Equity (Gross vs Net)")
        plt.xlabel("Date"); plt.ylabel("Equity")
        plt.tight_layout()
        plt.savefig(png, dpi=120)
    except Exception:
        png=None

    g = meta.get("gross", {})
    n = meta.get("net",   {})
    params = meta.get("params", {})

    block = f"""{CARD_START}
<section style="border:1px solid #ddd;padding:12px;margin:16px 0;border-radius:8px">
  <h2 style="margin:0 0 8px 0">Technical Overlay — COSTED</h2>
  <p style="color:#666;margin:4px 0 8px 0">Net includes turnover cost = {params.get('cost_bps',0):.1f} bps × daily turnover.</p>
  <p><b>Gross Sharpe:</b> {g.get('sharpe',0):.3f} &nbsp; <b>Hit:</b> {g.get('hit',0):.3f} &nbsp; <b>MaxDD:</b> {g.get('maxdd',0):.3f}<br>
     <b> Net  Sharpe:</b> {n.get('sharpe',0):.3f} &nbsp; <b>Hit:</b> {n.get('hit',0):.3f} &nbsp; <b>MaxDD:</b> {n.get('maxdd',0):.3f}</p>
  {"<img src='runs_plus/osc_costed_eq.png' style='max-width:100%'>" if png else ""}
  <p style="color:#666;margin:4px 0 0 0">Assets used: {params.get('assets_used',0)} | Max dPos: {params.get('max_dpos',0)} | Smooth: {params.get('smooth_span',0)} | MomWin: {params.get('mom_win',0)}</p>
</section>
{CARD_END}"""

    for f in FILES:
        if not f.exists():
            print(f"skip {f.name}: not found"); 
            continue
        f.write_text(upsert(f.read_text(), block))
        print(f"✅ Upserted Oscillator COSTED card in {f.name}")
