#!/usr/bin/env python3
# tools/add_vol_target_card.py
# Adds a Main vs Vol-Target comparison card to your reports.

from pathlib import Path
import json as _json

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"
FILES = [ROOT/"report_best_plus.html", ROOT/"report_all.html"]
START="<!--VOL_TARGET_CARD_START-->"
END  ="<!--VOL_TARGET_CARD_END-->"

def f3(x):
    try: return f"{float(x):.3f}"
    except: return "?"

def upsert(html, block):
    if START in html and END in html:
        pre = html.split(START)[0]; post = html.split(END,1)[1]
        return pre + block + post
    return html.replace("</body>", block + "\n</body>") if "</body>" in html else html + block

if __name__ == "__main__":
    main  = _json.loads((RUNS/"final_portfolio_summary.json").read_text())
    vt    = _json.loads((RUNS/"final_portfolio_vt_summary.json").read_text())

    block = f"""{START}
<section style="border:2px solid #589;padding:14px;margin:18px 0;border-radius:10px">
  <h2 style="margin:0 0 8px 0">Volatility Targeting (Main Portfolio)</h2>
  <div style="display:flex;gap:24px;flex-wrap:wrap">
    <div style="min-width:260px">
      <h3>Main</h3>
      <p>OOS Sharpe {f3(main['out_sample']['sharpe'])}<br>
         OOS MaxDD {f3(main['out_sample']['maxdd'])}</p>
    </div>
    <div style="min-width:260px">
      <h3>Vol-Targeted</h3>
      <p>OOS Sharpe {f3(vt['out_sample']['sharpe'])}<br>
         OOS MaxDD {f3(vt['out_sample']['maxdd'])}</p>
      <p style="color:#666">Target vol {f3(vt['target_ann'])}, window {int(vt['roll_days'])}d, bounds [{f3(vt['scale_bounds'][0])}, {f3(vt['scale_bounds'][1])}]</p>
    </div>
  </div>
  <p style="color:#666;margin-top:6px">Daily scaling toward a fixed annualized vol. Bounded to avoid whipsawing.</p>
</section>
{END}"""

    for f in FILES:
        if not f.exists():
            print("skip", f.name); 
            continue
        f.write_text(upsert(f.read_text(), block))
        print("✅ Upserted VOL-TARGET card in", f.name)
