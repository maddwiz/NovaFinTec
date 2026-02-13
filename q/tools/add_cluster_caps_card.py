#!/usr/bin/env python3
# tools/add_cluster_caps_card.py
# Adds a comparison card (Main vs ClusterCap) to report_best_plus.html and report_all.html

from pathlib import Path
import json as _json

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"
FILES = [ROOT/"report_best_plus.html", ROOT/"report_all.html"]
START="<!--CLUSTER_CAPS_CARD_START-->"
END  ="<!--CLUSTER_CAPS_CARD_END-->"

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
    clust = _json.loads((RUNS/"final_portfolio_cluster_summary.json").read_text())

    block = f"""{START}
<section style="border:2px solid #7a5;padding:14px;margin:18px 0;border-radius:10px">
  <h2 style="margin:0 0 8px 0">Cluster-Capped Portfolio (Diversification Control)</h2>
  <div style="display:flex;gap:24px;flex-wrap:wrap">
    <div style="min-width:260px">
      <h3>Main (Current)</h3>
      <p>OOS Sharpe {f3(main['out_sample']['sharpe'])}<br>
         OOS MaxDD {f3(main['out_sample']['maxdd'])}</p>
    </div>
    <div style="min-width:260px">
      <h3>ClusterCap</h3>
      <p>OOS Sharpe {f3(clust['out_sample']['sharpe'])}<br>
         OOS MaxDD {f3(clust['out_sample']['maxdd'])}</p>
      <p style="color:#666">Per-asset cap {f3(clust['cap_per'])}, per-cluster cap {f3(clust['cluster_cap'])}, corr≥{f3(clust['corr_thresh'])}</p>
    </div>
  </div>
  <p style="color:#666;margin-top:6px">We cap clusters defined by |correlation| ≥ threshold, then renormalize. This reduces hidden concentration risk.</p>
</section>
{END}"""

    for f in FILES:
        if not f.exists():
            print("skip", f.name)
            continue
        f.write_text(upsert(f.read_text(), block))
        print("✅ Upserted CLUSTER-CAPS card in", f.name)
