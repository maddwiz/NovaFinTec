#!/usr/bin/env python3
"""
add_portfolio_card.py  (refresh version)

- Reads runs_plus/portfolio_summary.json
- Updates (or inserts) a "Portfolio (WF+)" card in both report_all.html and report_best_plus.html
- If a previous Portfolio card exists, it is REPLACED so numbers stay current.
"""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
FILES = [ROOT / "report_all.html", ROOT / "report_best_plus.html"]

CARD_START = "<!--PORTFOLIO_CARD_START-->"
CARD_END   = "<!--PORTFOLIO_CARD_END-->"

def exists(p):
    try: return p.exists()
    except: return False

def make_block(s):
    cap = float(s.get("cap_per_asset", s.get("cap", 0.0)))
    cost = float(s.get("cost_bps", 0.0))
    sharpe = float(s.get("sharpe", 0.0))
    hit = float(s.get("hit", 0.0))
    maxdd = float(s.get("maxDD", 0.0))
    return f"""{CARD_START}
<section style="border:1px solid #ddd;padding:12px;margin:16px 0;border-radius:8px">
  <h2 style="margin:0 0 8px 0">Portfolio (WF+)</h2>
  <p style="margin:4px 0 8px 0;color:#666">Inverse-vol weights (cap {cap:.2f}), turnover cost {cost:.1f} bps.</p>
  <table style="width:100%;border-collapse:collapse;font-size:14px">
    <tbody>
      <tr><td>Sharpe</td><td style="text-align:right">{sharpe:.3f}</td></tr>
      <tr><td>Hit</td><td style="text-align:right">{hit:.3f}</td></tr>
      <tr><td>MaxDD</td><td style="text-align:right">{maxdd:.3f}</td></tr>
    </tbody>
  </table>
</section>
{CARD_END}"""

def upsert_card(html_in: str, card_block: str) -> str:
    # If card with markers exists, replace it
    if CARD_START in html_in and CARD_END in html_in:
        pre = html_in.split(CARD_START)[0]
        post = html_in.split(CARD_END, 1)[1]
        return pre + card_block + post
    # If an older card (without markers) exists, replace the first section containing "Portfolio (WF+)"
    needle = "<h2 style=\"margin:0 0 8px 0\">Portfolio (WF+)</h2>"
    if needle in html_in:
        # find the section start
        sec_start = html_in.rfind("<section", 0, html_in.find(needle))
        if sec_start == -1:
            sec_start = html_in.find(needle)  # fallback
        # find the section end
        sec_end = html_in.find("</section>", html_in.find(needle))
        if sec_end != -1:
            sec_end += len("</section>")
            return html_in[:sec_start] + card_block + html_in[sec_end:]
    # Otherwise, insert before </body>
    if "</body>" in html_in:
        return html_in.replace("</body>", card_block + "\n</body>")
    return html_in + "\n" + card_block

if __name__ == "__main__":
    summ_p = RUNS / "portfolio_summary.json"
    if not exists(summ_p):
        raise SystemExit("portfolio_summary.json not found. Run: python tools/build_portfolio_plus.py first.")
    s = json.loads(summ_p.read_text())
    block = make_block(s)

    for f in FILES:
        if not exists(f):
            print(f"skip {f.name}: not found"); 
            continue
        html_in = f.read_text()
        html_out = upsert_card(html_in, block)
        f.write_text(html_out)
        print(f"✅ Refreshed Portfolio card in {f.name}")
