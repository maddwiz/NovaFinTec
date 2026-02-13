#!/usr/bin/env python3
# tools/add_costs_card.py
from pathlib import Path
import json as _json

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"
FILES = [ROOT/"report_best_plus.html", ROOT/"report_all.html"]
START="<!--COSTS_CARD_START-->"
END  ="<!--COSTS_CARD_END-->"

def f3(x):
    try: return f"{float(x):.3f}"
    except: return "?"

def upsert(html, block):
    if START in html and END in html:
        pre = html.split(START)[0]; post = html.split(END,1)[1]
        return pre + block + post
    return html.replace("</body>", block + "\n</body>") if "</body>" in html else html + block

if __name__ == "__main__":
    # raw (gross)
    main_g = _json.loads((RUNS/"final_portfolio_summary.json").read_text())
    reg_g  = _json.loads((RUNS/"final_portfolio_regime_summary.json").read_text())
    dna_g  = _json.loads((RUNS/"final_portfolio_regime_dna_summary.json").read_text())
    # net (after costs)
    main_n = _json.loads((RUNS/"final_portfolio_summary_costs.json").read_text())
    reg_n  = _json.loads((RUNS/"final_portfolio_regime_summary_costs.json").read_text())
    dna_n  = _json.loads((RUNS/"final_portfolio_regime_dna_summary_costs.json").read_text())

    def lines(g, n):
        return (
            f"Gross OOS Sharpe {f3(g['out_sample']['sharpe'])} → "
            f"Net {f3(n['out_sample']['sharpe'])}<br>"
            f"Gross OOS MaxDD {f3(g['out_sample']['maxdd'])} → "
            f"Net {f3(n['out_sample']['maxdd'])}"
        )

    block = f"""{START}
<section style="border:2px solid #0a7;padding:14px;margin:18px 0;border-radius:10px">
  <h2 style="margin:0 0 8px 0">Costs Impact (Mgmt Fee + Slippage)</h2>
  <div style="display:flex;gap:24px;flex-wrap:wrap">
    <div style="min-width:240px"><h3>Main</h3><p>{lines(main_g, main_n)}</p></div>
    <div style="min-width:240px"><h3>Regime</h3><p>{lines(reg_g, reg_n)}</p></div>
    <div style="min-width:240px"><h3>Regime+DNA</h3><p>{lines(dna_g, dna_n)}</p></div>
  </div>
  <p style="color:#666;margin-top:6px">Costs applied: 1%/yr fee + 2bps·|ret| slippage proxy. Adjust in tools/apply_costs.py.</p>
</section>
{END}"""

    for f in FILES:
        if not f.exists():
            print("skip", f.name); 
            continue
        f.write_text(upsert(f.read_text(), block))
        print("✅ Upserted COSTS card in", f.name)
