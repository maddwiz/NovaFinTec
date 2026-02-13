#!/usr/bin/env python3
from pathlib import Path

SRC = Path("report_plus.html")
IMG = Path("runs_plus/heartbeat.png")
DST = Path("report_heartbeat.html")

if not SRC.exists():
    raise SystemExit("report_plus.html not found (run your normal report first).")
if not IMG.exists():
    raise SystemExit("runs_plus/heartbeat.png not found (run tools/make_heartbeat.py first).")

card = f"""
<section style="border:1px solid #ddd;padding:12px;margin:16px 0;border-radius:8px">
  <h2 style="margin:0 0 8px 0">Heartbeat (BPM)</h2>
  <img src="{IMG.as_posix()}" width="720" alt="Heartbeat Metabolism (BPM)">
  <p style="color:#666;font-size:12px;margin-top:8px">Source: runs_plus/heartbeat.json</p>
</section>
"""

html_in = SRC.read_text()
html_out = html_in.replace("</body>", card + "\n</body>") if "</body>" in html_in else (html_in + card)
DST.write_text(html_out)
print("✅ Wrote", DST.as_posix())
