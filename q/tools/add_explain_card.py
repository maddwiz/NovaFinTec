#!/usr/bin/env python3
# tools/add_explain_card.py
from pathlib import Path
import json as _json

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
FILES = [ROOT / "report_best_plus.html", ROOT / "report_all.html"]
CARD_START="<!--EXPLAIN_CARD_START-->"
CARD_END="<!--EXPLAIN_CARD_END-->"

def upsert(html, block):
    if CARD_START in html and CARD_END in html:
        pre = html.split(CARD_START)[0]; post = html.split(CARD_END,1)[1]
        return pre + block + post
    return html.replace("</body>", block + "\n</body>") if "</body>" in html else html + block

if __name__ == "__main__":
    p = RUNS / "explain_latest.json"
    if not p.exists():
        raise SystemExit("Run tools/make_explain.py first.")

    ex = _json.loads(p.read_text())
    bullets = ex.get("bullets", [])
    li = "".join([f"<li>{b}</li>" for b in bullets]) or "<li>No explanation available.</li>"

    block = f"""{CARD_START}
<section style="border:2px solid #444;padding:14px;margin:18px 0;border-radius:10px">
  <h2 style="margin:0 0 8px 0">Explanation / Provenance</h2>
  <ul style="margin:6px 0 0 18px">{li}</ul>
  <p style="color:#666;margin-top:8px">Observer-mode explanations derived from regime, symbolic, reflexive, and final portfolio state.</p>
</section>
{CARD_END}"""
    for f in FILES:
        if not f.exists():
            print("skip", f.name)
            continue
        f.write_text(upsert(f.read_text(), block))
        print(f"✅ Upserted EXPLANATION card in", f.name)
