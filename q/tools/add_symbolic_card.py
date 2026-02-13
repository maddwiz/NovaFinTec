#!/usr/bin/env python3
# tools/add_symbolic_card.py
# Adds a "Symbolic / Affective" card to your HTML reports.

from pathlib import Path
import json as _json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
FILES = [ROOT / "report_best_plus.html", ROOT / "report_all.html"]
CARD_START="<!--SYMBOLIC_CARD_START-->"
CARD_END="<!--SYMBOLIC_CARD_END-->"

def upsert(html, block):
    if CARD_START in html and CARD_END in html:
        pre = html.split(CARD_START)[0]; post = html.split(CARD_END,1)[1]
        return pre + block + post
    return html.replace("</body>", block + "\n</body>") if "</body>" in html else html + block

if __name__ == "__main__":
    evp = RUNS / "symbolic_events.csv"
    sigp = RUNS / "symbolic_signal.csv"
    sump = RUNS / "symbolic_summary.json"
    if not evp.exists() or not sigp.exists() or not sump.exists():
        raise SystemExit("Run tools/make_symbolic.py first.")

    # small preview chart
    png = RUNS / "symbolic_signal_eq.png"
    try:
        import matplotlib.pyplot as plt
        df = pd.read_csv(sigp, parse_dates=["DATE"])
        if not df.empty:
            # show aggregate "ALL" if present, else average over assets
            if "ALL" in df["ASSET"].unique():
                dfa = df[df["ASSET"]=="ALL"].sort_values("DATE")
            else:
                dfa = df.groupby("DATE", as_index=False)["sym_signal"].mean().sort_values("DATE")
            eq = (1.0 + dfa["sym_signal"].clip(-0.02,0.02)).cumprod()
            plt.figure()
            plt.plot(dfa["DATE"], eq)
            plt.title("Symbolic Signal (cumulated tiny returns from signal)")
            plt.xlabel("Date"); plt.ylabel("Index from 1.0")
            plt.tight_layout()
            plt.savefig(png, dpi=120)
    except Exception:
        png = None

    meta = _json.loads(sump.read_text())
    top = meta.get("top_words", {})
    tops = ", ".join(list(top.keys())[:12]) if top else "(none)"

    block = f"""{CARD_START}
<section style="border:2px solid #444;padding:14px;margin:18px 0;border-radius:10px">
  <h2 style="margin:0 0 8px 0">Symbolic / Affective Ingestion</h2>
  <p style="color:#666;margin:4px 0 8px 0">News/comments → tokens → sentiment &amp; affect → daily signal. Simple lexicon baseline (no external API).</p>
  <p><b>Rows:</b> {meta.get('rows',0)} &nbsp; <b>Assets:</b> {len(meta.get('assets',[]))} &nbsp; <b>Span:</b> {meta.get('date_min','?')} → {meta.get('date_max','?')}</p>
  <p><b>Top tokens:</b> {tops}</p>
  {"<img src='runs_plus/symbolic_signal_eq.png' style='max-width:100%'>" if png else ""}
</section>
{CARD_END}"""
    for f in FILES:
        if not f.exists():
            print("skip", f.name)
            continue
        f.write_text(upsert(f.read_text(), block))
        print(f"✅ Upserted SYMBOLIC card in", f.name)
