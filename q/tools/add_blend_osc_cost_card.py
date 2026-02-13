#!/usr/bin/env python3
# tools/add_blend_osc_cost_card.py
from pathlib import Path
import json as _json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
FILES = [ROOT / "report_best_plus.html", ROOT / "report_all.html"]
CARD_START="<!--BLEND_OSC_COST_CARD_START-->"
CARD_END="<!--BLEND_OSC_COST_CARD_END-->"

def upsert(html, block):
    if CARD_START in html and CARD_END in html:
        pre = html.split(CARD_START)[0]; post = html.split(CARD_END,1)[1]
        return pre + block + post
    return html.replace("</body>", block + "\n</body>") if "</body>" in html else html + block

if __name__ == "__main__":
    s = RUNS / "blend_main_osc_costed_summary.json"
    c = RUNS / "blend_main_osc_costed.csv"
    if not s.exists() or not c.exists():
        raise SystemExit("Run blend_main_with_osc_costed.py first.")
    meta = _json.loads(s.read_text())
    df = pd.read_csv(c, parse_dates=["DATE"])

    png = RUNS / "blend_main_osc_costed.png"
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(df["DATE"], df["eq"])
        plt.title(f"Blended Equity: Main × Osc (NET, alpha={meta.get('alpha',0):.2f})")
        plt.xlabel("Date"); plt.ylabel("Equity")
        plt.tight_layout()
        plt.savefig(png, dpi=120)
    except Exception:
        png=None

    block = f"""{CARD_START}
<section style="border:1px solid #ddd;padding:12px;margin:16px 0;border-radius:8px">
  <h2 style="margin:0 0 8px 0">Blended Portfolio: Main × Oscillator (NET)</h2>
  <p style="color:#666;margin:4px 0 8px 0">Static blend uses NET oscillator (after turnover cost).</p>
  <p><b>Best alpha (weight on Main):</b> {meta.get('alpha',0):.2f}<br>
     <b>Sharpe:</b> {meta.get('sharpe',0):.3f} &nbsp; 
     <b>Hit:</b> {meta.get('hit',0):.3f} &nbsp; 
     <b>MaxDD:</b> {meta.get('maxdd',0):.3f}</p>
  {"<img src='runs_plus/blend_main_osc_costed.png' style='max-width:100%'>" if png else ""}
</section>
{CARD_END}"""
    for f in FILES:
        if not f.exists():
            print(f"skip {f.name}: not found"); 
            continue
        f.write_text(upsert(f.read_text(), block))
        print(f"✅ Upserted Blend(Main×Osc NET) card in {f.name}")
