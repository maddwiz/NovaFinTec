#!/usr/bin/env python3
# tools/add_vol_card.py
from pathlib import Path
import json as _json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
FILES = [ROOT / "report_best_plus.html", ROOT / "report_all.html"]
CARD_START="<!--VOL_CARD_START-->"
CARD_END="<!--VOL_CARD_END-->"

def upsert(html, block):
    if CARD_START in html and CARD_END in html:
        pre = html.split(CARD_START)[0]; post = html.split(CARD_END, 1)[1]
        return pre + block + post
    return html.replace("</body>", block + "\n</body>") if "</body>" in html else html + block

if __name__ == "__main__":
    s = RUNS / "vol_overlay_summary.json"
    c = RUNS / "vol_overlay.csv"
    if not s.exists() or not c.exists():
        raise SystemExit("Run run_vol_overlay.py first.")
    meta = _json.loads(s.read_text())
    df = pd.read_csv(c)

    png = RUNS / "vol_overlay.png"
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(pd.to_datetime(df["DATE"]), df["eq"])
        plt.title("Volatility Overlay Equity")
        plt.xlabel("Date"); plt.ylabel("Equity")
        plt.tight_layout()
        plt.savefig(png, dpi=120)
    except Exception:
        png = None

    block = f"""{CARD_START}
<section style="border:1px solid #ddd;padding:12px;margin:16px 0;border-radius:8px">
  <h2 style="margin:0 0 8px 0">Volatility Regime Overlay</h2>
  <p style="color:#666;margin:4px 0 8px 0">Sleeve that reduces exposure during high-volatility regimes.</p>
  <p><b>Sharpe:</b> {meta.get('sharpe',0):.3f} &nbsp; 
     <b>Hit:</b> {meta.get('hit',0):.3f} &nbsp; 
     <b>MaxDD:</b> {meta.get('maxdd',0):.3f}</p>
  {"<img src='runs_plus/vol_overlay.png' style='max-width:100%'>" if png else ""}
</section>
{CARD_END}"""
    for f in FILES:
        if not f.exists(): 
            print(f"skip {f.name}: not found")
            continue
        f.write_text(upsert(f.read_text(), block))
        print(f"✅ Upserted Volatility card in {f.name}")
