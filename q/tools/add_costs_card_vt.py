#!/usr/bin/env python3
# tools/add_costs_card_vt.py
# Inserts a Costs card comparing Main / Regime / DNA / Vol-Target (Gross vs Net)

from pathlib import Path
import json as _json

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"
FILES = [ROOT/"report_best_plus.html", ROOT/"report_all.html"]
START="<!--COSTS_CARD_VT_START-->"
END  ="<!--COSTS_CARD_VT_END-->"

def f3(x):
    try: return f"{float(x):.3f}"
    except: return "?"

def block_line(name, g, n):
    return (f"<div style='min-width:220px'><h3>{name}</h3>"
            f"<p>OOS Sharpe {f3(g['out_sample']['sharpe'])} → Net {f3(n['out_sample']['sharpe'])}<br>"
            f"OOS MaxDD {f3(g['out_sample']['maxdd'])} → Net {f3(n['out_sample']['maxdd'])}</p></div>")

def upsert(html, block):
    if START in html and END in html:
        pre = html.split(START)[0]; post = html.split(END,1)[1]
        return pre + block + post
    return html.replace("</body>", block + "\n</body>") if "</body>" in html else html + block

if __name__ == "__main__":
    main_g = _json.loads((RUNS/"final_portfolio_summary.json").read_text())
    reg_g  = _json.loads((RUNS/"final_portfolio_regime_summary.json").read_text())
    dna_g  = _json.loads((RUNS/"final_portfolio_regime_dna_summary.json").read_text())
    vt_g   = _json.loads((RUNS/"final_portfolio_vt_summary.json").read_text())

    # Net (after costs). The first three assume you already ran tools/apply_costs.py
    main_n = _json.loads((RUNS/"final_portfolio_summary_costs.json").read_text())
    reg_n  = _json.loads((RUNS/"final_portfolio_regime_summary_costs.json").read_text())
    dna_n  = _json.loads((RUNS/"final_portfolio_regime_dna_summary_costs.json").read_text())
    vt_n   = _json.loads((RUNS/"final_portfolio_vt_summary_costs.json").read_text())

    block = f"""{START}
<section style="border:2px solid #0a7;padding:14px;margin:18px 0;border-radius:10px">
  <h2 style="margin:0 0 8px 0">Costs Impact (Gross → Net)</h2>
  <div style="display:flex;gap:24px;flex-wrap:wrap">
    {block_line("Main", main_g, main_n)}
    {block_line("Regime", reg_g, reg_n)}
    {block_line("Regime + DNA", dna_g, dna_n)}
    {block_line("Vol-Target", vt_g, vt_n)}
  </div>
  <p style="color:#666;margin-top:6px">Model: 1%/yr mgmt + 2bps·|ret| slippage proxy. Adjust in tools/apply_costs.py and tools/apply_costs_vt.py.</p>
</section>
{END}"""

    for f in FILES:
        if not f.exists(): 
            print("skip", f.name); 
            continue
        f.write_text(upsert(f.read_text(), block))
        print("✅ Upserted COSTS (4-way) card in", f.name)
