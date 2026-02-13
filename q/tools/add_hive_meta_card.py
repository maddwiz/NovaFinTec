#!/usr/bin/env python3
from pathlib import Path
import json, html

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
IN_FILES = [ROOT/"report_all.html", ROOT/"report_best_plus.html"]

def read_json(p, default):
    try: return json.loads(Path(p).read_text())
    except Exception: return default

def insert_before_body(html_in: str, block: str) -> str:
    return html_in.replace("</body>", block + "\n</body>") if "</body>" in html_in else html_in + block

def make_card():
    hive = read_json(RUNS/"hive.json", {"hives": {}}).get("hives", {})
    hivew = read_json(RUNS/"hive_council.json", {"hive_weights": {}}).get("hive_weights", {})
    meta  = read_json(RUNS/"meta_council.json", {"asset_weights": {}}).get("asset_weights", {})

    # Build simple lists
    hive_rows = "".join(f"<tr><td>{html.escape(h)}</td><td style='text-align:right'>{hivew.get(h,0):+.3f}</td><td>{', '.join(map(html.escape, (hive.get(h) or [])[:6]))}{' …' if len(hive.get(h,[]))>6 else ''}</td></tr>"
                        for h in sorted(hive.keys()))
    meta_top = sorted(meta.items(), key=lambda kv: abs(kv[1]), reverse=True)[:12]
    meta_rows = "".join(f"<tr><td>{html.escape(s)}</td><td style='text-align:right'>{v:+.3f}</td></tr>"
                        for s,v in meta_top)

    return f"""
<section style="border:1px solid #ddd;padding:12px;margin:16px 0;border-radius:8px">
  <h2 style="margin:0 0 8px 0">Hive Councils & Meta-Council</h2>
  <div style="display:flex;gap:24px;flex-wrap:wrap">
    <div style="flex:1;min-width:280px">
      <h3 style="margin:6px 0 6px 0">Hive Weights</h3>
      <table style="width:100%;border-collapse:collapse;font-size:14px">
        <thead><tr><th style="text-align:left">Hive</th><th style="text-align:right">Weight</th><th style="text-align:left">Members</th></tr></thead>
        <tbody>{hive_rows}</tbody>
      </table>
    </div>
    <div style="flex:1;min-width:240px">
      <h3 style="margin:6px 0 6px 0">Meta-Council: Top |weight|</h3>
      <table style="width:100%;border-collapse:collapse;font-size:14px">
        <thead><tr><th style="text-align:left">Symbol</th><th style="text-align:right">Weight</th></tr></thead>
        <tbody>{meta_rows}</tbody>
      </table>
    </div>
  </div>
  <p style="color:#666;margin-top:6px">Weights normalized with per-hive cap; asset weights inherit hive weight and member proportions.</p>
</section>
"""

if __name__ == "__main__":
    card = make_card()
    for p in IN_FILES:
        if not p.exists():
            print(f"skip {p.name}: not found"); continue
        txt = p.read_text()
        if "Hive Councils & Meta-Council" in txt:
            print(f"already has Hive/Meta card: {p.name}")
            continue
        p.write_text(insert_before_body(txt, card))
        print(f"✅ Injected Hive/Meta card into {p.name}")
