#!/usr/bin/env python3
# tools/add_regime_card.py
from pathlib import Path
import json as _json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
FILES = [ROOT / "report_best_plus.html", ROOT / "report_all.html"]
CARD_START="<!--REGIME_CARD_START-->"
CARD_END="<!--REGIME_CARD_END-->"

def upsert(html, block):
    if CARD_START in html and CARD_END in html:
        pre = html.split(CARD_START)[0]; post = html.split(CARD_END,1)[1]
        return pre + block + post
    return html.replace("</body>", block + "\n</body>") if "</body>" in html else html + block

if __name__ == "__main__":
    sump = RUNS / "regime_summary.json"
    wts = RUNS / "regime_weights.csv"
    if not sump.exists() or not wts.exists():
        raise SystemExit("Run tools/make_regime.py first.")

    # tiny preview (regime timeline)
    png = RUNS / "regime_preview.png"
    try:
        import matplotlib.pyplot as plt
        df = pd.read_csv(wts, parse_dates=["DATE"])
        if not df.empty:
            # map regime to levels
            levels = {"CALM_TREND":1, "CALM_CHOP":0, "HIGHVOL_TREND":2, "CRISIS":3}
            y = df["regime"].map(levels).fillna(0)
            plt.figure()
            plt.plot(df["DATE"], y)
            plt.title("Regime timeline (0=CalmChop,1=CalmTrend,2=HighVolTrend,3=Crisis)")
            plt.xlabel("Date"); plt.ylabel("Regime code")
            plt.tight_layout()
            plt.savefig(png, dpi=120)
    except Exception:
        png = None

    meta = _json.loads(sump.read_text())
    cur = meta.get("current", {})
    weights = cur.get("suggested_weights", {})
    wtxt = ", ".join([f"{k}={weights.get(k,0):.2f}" for k in ["w_main","w_vol","w_osc","w_sym","w_reflex"]])

    block = f"""{CARD_START}
<section style="border:2px solid #444;padding:14px;margin:18px 0;border-radius:10px">
  <h2 style="margin:0 0 8px 0">Adaptive Regime Switcher (observer)</h2>
  <p style="color:#666;margin:4px 0 8px 0">Detects regimes and proposes sleeve weights (NOT applied yet).</p>
  <p><b>Current:</b> {cur.get('date','?')} → <b>{cur.get('regime','?')}</b> &nbsp; <b>Suggested:</b> {wtxt}</p>
  {"<img src='runs_plus/regime_preview.png' style='max-width:100%'>" if png else ""}
</section>
{CARD_END}"""
    for f in FILES:
        if not f.exists(): 
            print("skip", f.name); 
            continue
        f.write_text(upsert(f.read_text(), block))
        print(f"✅ Upserted REGIME card in", f.name)
