#!/usr/bin/env python3
# tools/add_overlay_card.py
# Inserts a compare card: Vol-Target vs Vol-Target+Overlay

from pathlib import Path
import json as _json

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"
FILES = [ROOT/"report_best_plus.html", ROOT/"report_all.html"]
START="<!--OVERLAY_CARD_START-->"
END  ="<!--OVERLAY_CARD_END-->"

def f3(x):
    try: return f"{float(x):.3f}"
    except: return "?"

def upsert(html, block):
    if START in html and END in html:
        pre = html.split(START)[0]; post = html.split(END,1)[1]
        return pre + block + post
    return html.replace("</body>", block + "\n</body>") if "</body>" in html else html + block

if __name__ == "__main__":
    vt  = _json.loads((RUNS/"final_portfolio_vt_summary.json").read_text())
    vto = _json.loads((RUNS/"final_portfolio_vt_overlay_summary.json").read_text())

    block = f"""{START}
<section style="border:2px solid #a85;padding:14px;margin:18px 0;border-radius:10px">
  <h2 style="margin:0 0 8px 0">Overlay (Breadth + Calmness)</h2>
  <div style="display:flex;gap:24px;flex-wrap:wrap">
    <div style="min-width:260px">
      <h3>Vol-Target</h3>
      <p>OOS Sharpe {f3(vt['out_sample']['sharpe'])}<br>
         OOS MaxDD {f3(vt['out_sample']['maxdd'])}</p>
    </div>
    <div style="min-width:260px">
      <h3>Vol-Target + Overlay</h3>
      <p>OOS Sharpe {f3(vto['out_sample']['sharpe'])}<br>
         OOS MaxDD {f3(vto['out_sample']['maxdd'])}</p>
      <p style="color:#666">Overlay alpha is 0.70–1.30 from breadth (% up) and calmness (low realized vol).</p>
    </div>
  </div>
</section>
{END}"""

    for f in FILES:
        if not f.exists():
            print("skip", f.name); 
            continue
        f.write_text(upsert(f.read_text(), block))
        print("✅ Upserted OVERLAY card in", f.name)
