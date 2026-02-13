#!/usr/bin/env python3
# tools/add_osc_card.py  (fixed var name + robust formatting)
from pathlib import Path
import json as _json
import pandas as pd
import math

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
FILES = [ROOT / "report_best_plus.html", ROOT / "report_all.html"]
CARD_START="<!--OSC_CARD_START-->"
CARD_END="<!--OSC_CARD_END-->"

def upsert(html, block):
    if CARD_START in html and CARD_END in html:
        pre = html.split(CARD_START)[0]
        post = html.split(CARD_END, 1)[1]
        return pre + block + post
    return html.replace("</body>", block + "\n</body>") if "</body>" in html else html + block

def safe_num(x, default=0.0):
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default

if __name__ == "__main__":
    summ = RUNS / "osc_portfolio_summary.json"
    ser  = RUNS / "osc_portfolio.csv"
    if not summ.exists() or not ser.exists():
        raise SystemExit("Run run_osc_portfolio.py first.")

    meta = _json.loads(summ.read_text())
    df = pd.read_csv(ser)

    sh = safe_num(meta.get("sharpe", 0.0))
    hi = safe_num(meta.get("hit", 0.0))
    dd = safe_num(meta.get("maxdd", 0.0))

    eq_png = RUNS / "osc_equity.png"
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(pd.to_datetime(df["DATE"]), df["eq"])
        plt.title("Oscillator Portfolio Equity")
        plt.xlabel("Date"); plt.ylabel("Equity")
        plt.tight_layout()
        plt.savefig(eq_png, dpi=120)
    except Exception:
        eq_png = None

    block = f"""{CARD_START}
<section style="border:1px solid #ddd;padding:12px;margin:16px 0;border-radius:8px">
  <h2 style="margin:0 0 8px 0">Technical Overlay (Momentum Sign)</h2>
  <p style="color:#666;margin:4px 0 8px 0">Equal-weight sleeve with safety caps to avoid unrealistic math.</p>
  <p><b>Sharpe:</b> {sh:.3f} &nbsp; <b>Hit:</b> {hi:.3f} &nbsp; <b>MaxDD:</b> {dd:.3f}</p>
  {"<img src='runs_plus/osc_equity.png' style='max-width:100%'>" if eq_png else ""}
</section>
{CARD_END}"""

    for f in FILES:
        if not f.exists():
            print("skip", f.name)
            continue
        f.write_text(upsert(f.read_text(), block))
        print(f"✅ Upserted Oscillator card in", f.name)
