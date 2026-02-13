#!/usr/bin/env python3
# tools/add_cross_overlay_card.py
# Adds the Cross-Domain Dream Overlays card.

from pathlib import Path
import json as _json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
FILES = [ROOT / "report_best_plus.html", ROOT / "report_all.html"]
CARD_START="<!--CROSS_OVERLAY_CARD_START-->"
CARD_END="<!--CROSS_OVERLAY_CARD_END-->"

def upsert(html, block):
    if CARD_START in html and CARD_END in html:
        pre = html.split(CARD_START)[0]; post = html.split(CARD_END,1)[1]
        return pre + block + post
    return html.replace("</body>", block + "\n</body>") if "</body>" in html else html + block

if __name__ == "__main__":
    summ = RUNS / "cross_overlay_summary.json"
    corrp = RUNS / "cross_overlay.csv"
    if not summ.exists() or not corrp.exists():
        raise SystemExit("Run tools/make_cross_overlay.py first.")

    # small chart: last-domain corr line (or average)
    png = RUNS / "cross_overlay_preview.png"
    try:
        import matplotlib.pyplot as plt
        df = pd.read_csv(corrp, parse_dates=["DATE"])
        if not df.empty:
            # average over domains to preview
            dfa = df.groupby("DATE", as_index=False)["corr"].mean().sort_values("DATE")
            plt.figure()
            plt.plot(dfa["DATE"], dfa["corr"])
            plt.title("Cross-Domain Overlay: Avg Rolling Corr (Finance vs. External)")
            plt.xlabel("Date"); plt.ylabel("Corr (126d)")
            plt.tight_layout()
            plt.savefig(png, dpi=120)
    except Exception:
        png = None

    meta = _json.loads(summ.read_text())
    tops = meta.get("top_recent", []) or []
    top_html = ""
    if tops:
        top_html = "<ul style='margin:6px 0 0 18px'>"
        for t in tops:
            top_html += f"<li>{t['DOMAIN']}: median {t['corr_median']:.2f}, last {t['corr_last']:.2f}</li>"
        top_html += "</ul>"

    block = f"""{CARD_START}
<section style="border:2px solid #444;padding:14px;margin:18px 0;border-radius:10px">
  <h2 style="margin:0 0 8px 0">Cross-Domain Dream Overlays</h2>
  <p style="color:#666;margin:4px 0 8px 0">Rolling correlations between Finance Dream Index and external domain indices (window=126d).</p>
  <p><b>Domains:</b> {len(meta.get('domains',[]))} &nbsp; <b>Rows:</b> {meta.get('rows',0)}</p>
  {top_html if top_html else "<p>No recent domain leaders.</p>"}
  {"<img src='runs_plus/cross_overlay_preview.png' style='max-width:100%'>" if png else ""}
</section>
{CARD_END}"""
    for f in FILES:
        if not f.exists():
            print("skip", f.name)
            continue
        f.write_text(upsert(f.read_text(), block))
        print(f"✅ Upserted CROSS OVERLAY card in", f.name)
