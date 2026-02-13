#!/usr/bin/env python3
# tools/add_final_card.py  (v3: shows guard + real OOS)
from pathlib import Path
import json as _json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
FILES = [ROOT / "report_best_plus.html", ROOT / "report_all.html"]
CARD_START="<!--FINAL_PORTFOLIO_CARD_START-->"
CARD_END="<!--FINAL_PORTFOLIO_CARD_END-->"

def upsert(html, block):
    if CARD_START in html and CARD_END in html:
        pre = html.split(CARD_START)[0]; post = html.split(CARD_END,1)[1]
        return pre + block + post
    return html.replace("</body>", block + "\n</body>") if "</body>" in html else html + block

if __name__ == "__main__":
    s = RUNS / "final_portfolio_summary.json"
    c = RUNS / "final_portfolio.csv"
    if not s.exists() or not c.exists():
        raise SystemExit("Run tools/triple_blend.py first.")
    meta = _json.loads(s.read_text())
    df = pd.read_csv(c, parse_dates=["DATE"])

    png = RUNS / "final_portfolio_eq.png"
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(df["DATE"], df["eq"])
        w = meta.get("weights", {})
        ttl = f"Final Portfolio Equity (w_main={w.get('w_main',0):.2f}, w_vol={w.get('w_vol',0):.2f}, w_osc={w.get('w_osc',0):.2f})"
        plt.title(ttl)
        plt.xlabel("Date"); plt.ylabel("Equity")
        plt.tight_layout()
        plt.savefig(png, dpi=120)
    except Exception:
        png = None

    w   = meta.get("weights", {})
    IS  = meta.get("in_sample", {})
    OS  = meta.get("out_sample", {})
    SS  = meta.get("sleeve_sharpes_test", {})
    SP  = meta.get("split", {})
    GRD = meta.get("guard","")

    guard_html = ""
    if GRD:
        guard_html = f"<p style='margin:6px 0;color:#b00020'><b>Guard:</b> {GRD.replace('_',' ')}</p>"

    block = f"""{CARD_START}
<section style="border:2px solid #444;padding:14px;margin:18px 0;border-radius:10px">
  <h2 style="margin:0 0 8px 0">FINAL PORTFOLIO (Triple Blend, robust OOS)</h2>
  <p style="color:#666;margin:4px 0 10px 0">Weights chosen on TRAIN (first 75%), evaluated on TEST (last 25%). Train end: {SP.get('train_until','?')}.</p>
  <p><b>Weights</b> — Main: {w.get('w_main',0):.2f} &nbsp; Vol: {w.get('w_vol',0):.2f} &nbsp; Osc: {w.get('w_osc',0):.2f}</p>
  <p><b>In-Sample (Train):</b> Sharpe {IS.get('sharpe',0):.3f} | Hit {IS.get('hit',0):.3f} | MaxDD {IS.get('maxdd',0):.3f}<br>
     <b>Out-of-Sample (Test):</b> Sharpe {OS.get('sharpe',0):.3f} | Hit {OS.get('hit',0):.3f} | MaxDD {OS.get('maxdd',0):.3f}</p>
  <p style="color:#666;margin:4px 0 8px 0">Sleeve Sharpes (Test, unscaled) — Main: {SS.get('main',0):.3f} | Vol NET: {SS.get('vol_net',0):.3f} | Osc NET: {SS.get('osc_net',0):.3f}</p>
  {guard_html}
  {"<img src='runs_plus/final_portfolio_eq.png' style='max-width:100%'>" if png else ""}
</section>
{CARD_END}"""
    for f in FILES:
        if not f.exists():
            print("skip", f.name)
            continue
        f.write_text(upsert(f.read_text(), block))
        print(f"✅ Upserted FINAL PORTFOLIO card in", f.name)
