#!/usr/bin/env python3
# tools/add_reflexive_card.py
# Adds a Reflexive Feedback card to HTML reports.

from pathlib import Path
import json as _json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
FILES = [ROOT / "report_best_plus.html", ROOT / "report_all.html"]
CARD_START="<!--REFLEXIVE_CARD_START-->"
CARD_END="<!--REFLEXIVE_CARD_END-->"

def upsert(html, block):
    if CARD_START in html and CARD_END in html:
        pre = html.split(CARD_START)[0]; post = html.split(CARD_END,1)[1]
        return pre + block + post
    return html.replace("</body>", block + "\n</body>") if "</body>" in html else html + block

if __name__ == "__main__":
    sigp = RUNS / "reflexive_signal.csv"
    sump = RUNS / "reflexive_summary.json"
    if not sigp.exists() or not sump.exists():
        raise SystemExit("Run tools/make_reflexive.py first.")

    # tiny visualization (cumulating a tiny fraction of signal)
    png = RUNS / "reflexive_signal_eq.png"
    try:
        import matplotlib.pyplot as plt
        df = pd.read_csv(sigp, parse_dates=["DATE"])
        if not df.empty:
            if "ALL" in df["ASSET"].unique():
                dfa = df[df["ASSET"]=="ALL"].sort_values("DATE")
            else:
                dfa = df.groupby("DATE", as_index=False)["reflex_signal"].mean().sort_values("DATE")
            eq = (1.0 + dfa["reflex_signal"].clip(-0.02,0.02)).cumprod()
            plt.figure()
            plt.plot(dfa["DATE"], eq)
            plt.title("Reflexive Signal (cumulated tiny returns from signal)")
            plt.xlabel("Date"); plt.ylabel("Index from 1.0")
            plt.tight_layout()
            plt.savefig(png, dpi=120)
    except Exception:
        png = None

    meta = _json.loads(sump.read_text())
    block = f"""{CARD_START}
<section style="border:2px solid #444;padding:14px;margin:18px 0;border-radius:10px">
  <h2 style="margin:0 0 8px 0">Reflexive Feedback</h2>
  <p style="color:#666;margin:4px 0 8px 0">Dream latents → compressed 1D signal per asset (rolling z → tanh).</p>
  <p><b>Rows:</b> {meta.get('rows',0)} &nbsp; <b>Assets:</b> {len(meta.get('assets',[]))} &nbsp; <b>Span:</b> {meta.get('date_min','?')} → {meta.get('date_max','?')}</p>
  {"<img src='runs_plus/reflexive_signal_eq.png' style='max-width:100%'>" if png else ""}
</section>
{CARD_END}"""
    for f in FILES:
        if not f.exists():
            print("skip", f.name)
            continue
        f.write_text(upsert(f.read_text(), block))
        print(f"✅ Upserted REFLEXIVE card in", f.name)
