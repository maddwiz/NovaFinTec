#!/usr/bin/env python3
# tools/add_hive_card.py
# Adds a Hive / Ecosystem card to the HTML reports.

from pathlib import Path
import json as _json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
FILES = [ROOT / "report_best_plus.html", ROOT / "report_all.html"]
CARD_START="<!--HIVE_CARD_START-->"
CARD_END="<!--HIVE_CARD_END-->"

def upsert(html, block):
    if CARD_START in html and CARD_END in html:
        pre = html.split(CARD_START)[0]; post = html.split(CARD_END,1)[1]
        return pre + block + post
    return html.replace("</body>", block + "\n</body>") if "</body>" in html else html + block

if __name__ == "__main__":
    summ = RUNS / "hive_summary.json"
    sigp = RUNS / "hive_signals.csv"
    if not summ.exists() or not sigp.exists():
        raise SystemExit("Run tools/make_hive.py first.")

    # Tiny preview chart: average hive_signal over time
    png = RUNS / "hive_preview.png"
    try:
        import matplotlib.pyplot as plt
        df = pd.read_csv(sigp, parse_dates=["DATE"])
        if not df.empty:
            dfa = df.groupby("DATE", as_index=False)["hive_signal"].mean().sort_values("DATE")
            dfa["eq"] = (1.0 + dfa["hive_signal"].clip(-0.02,0.02)).cumprod()
            plt.figure()
            plt.plot(dfa["DATE"], dfa["eq"])
            plt.title("Hive Layer (avg hive_signal → tiny cum index)")
            plt.xlabel("Date"); plt.ylabel("Index from 1.0")
            plt.tight_layout()
            plt.savefig(png, dpi=120)
    except Exception:
        png = None

    meta = _json.loads(summ.read_text())
    tops = meta.get("top_recent", []) or []
    top_html = ""
    if tops:
        top_html = "<ol style='margin:6px 0 0 18px'>"
        for t in tops:
            top_html += f"<li>{t['HIVE']}: median {t['median_signal']:.3f}, last {t['last_signal']:.3f}, members≈{t['avg_members']:.1f}</li>"
        top_html += "</ol>"

    block = f"""{CARD_START}
<section style="border:2px solid #444;padding:14px;margin:18px 0;border-radius:10px">
  <h2 style="margin:0 0 8px 0">Hive / Ecosystem</h2>
  <p style="color:#666;margin:4px 0 8px 0">Groups assets into hives and builds a combined symbolic+reflexive signal per hive (read-only preview).</p>
  <p><b>Hives:</b> {len(meta.get('hives',[]))} &nbsp; <b>Rows:</b> {meta.get('rows',0)} &nbsp; <b>Span:</b> {meta.get('date_min','?')} → {meta.get('date_max','?')}</p>
  {top_html if top_html else "<p>No hive leaders yet.</p>"}
  {"<img src='runs_plus/hive_preview.png' style='max-width:100%'>" if png else ""}
</section>
{CARD_END}"""
    for f in FILES:
        if not f.exists():
            print("skip", f.name)
            continue
        f.write_text(upsert(f.read_text(), block))
        print(f"✅ Upserted HIVE card in", f.name)
