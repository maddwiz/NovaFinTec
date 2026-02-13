#!/usr/bin/env python3
from pathlib import Path
import json, html

SRC = Path("report_plus.html")           # your existing report
COUNCIL = Path("runs_plus/council.json") # what you just created
DST = Path("report_council.html")        # new file we'll write

if not SRC.exists():
    raise SystemExit("report_plus.html not found (run your normal report first).")
if not COUNCIL.exists():
    raise SystemExit("runs_plus/council.json not found (run tools/make_council.py first).")

data = json.loads(COUNCIL.read_text())
weights = data.get("final_weights", {})
# Sort by absolute weight, top 20 rows
pairs = sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)[:20]

rows = "\n".join(
    f"<tr><td>{html.escape(sym)}</td>"
    f"<td style='text-align:right'>{w:+.4f}</td></tr>"
    for sym, w in pairs
)

card = f"""
<section style="border:1px solid #ddd;padding:12px;margin:16px 0;border-radius:8px">
  <h2 style="margin:0 0 8px 0">Council (final weights)</h2>
  <table style="width:100%;border-collapse:collapse;font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;font-size:14px">
    <thead>
      <tr><th style="text-align:left;padding:4px 0">Symbol</th>
          <th style="text-align:right;padding:4px 0">Weight</th></tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
  <p style="color:#666;font-size:12px;margin-top:8px">Source: runs_plus/council.json</p>
</section>
"""

html_in = SRC.read_text()
html_out = html_in.replace("</body>", card + "\n</body>") if "</body>" in html_in else (html_in + card)
DST.write_text(html_out)
print("✅ Wrote", DST.as_posix())
