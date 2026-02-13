#!/usr/bin/env python3
import subprocess, sys, json, html
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
RUNS  = ROOT / "runs_plus"

BASE_REPORT = ROOT / "report_plus.html"
OUT_REPORT  = ROOT / "report_council.html"   # keep same name you’ve been using

def run(cmd):
    print("▶", " ".join(cmd))
    subprocess.check_call(cmd)

def safe_exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False

def append_card(html_in: str, card_html: str) -> str:
    if "</body>" in html_in:
        return html_in.replace("</body>", card_html + "\n</body>")
    return html_in + card_html

def main():
    # 1) Build the base report (your normal builder)
    run([sys.executable, str(TOOLS / "build_report_plus.py")])

    # 2) Generate artifacts (Council, DNA, Heartbeat)
    run([sys.executable, str(TOOLS / "make_council.py")])
    run([sys.executable, str(TOOLS / "make_dna_drift.py")])
    run([sys.executable, str(TOOLS / "make_heartbeat.py")])

    # 3) Load base report
    if not BASE_REPORT.exists():
        print("ERROR: base report not found:", BASE_REPORT)
        sys.exit(1)
    html_out = BASE_REPORT.read_text()

    # 4) Council card (if available)
    council_json = RUNS / "council.json"
    if safe_exists(council_json):
        data = json.loads(council_json.read_text())
        weights = data.get("final_weights", {})
        pairs = sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)[:20]
        rows = "\n".join(
            f"<tr><td>{html.escape(sym)}</td>"
            f"<td style='text-align:right'>{w:+.4f}</td></tr>"
            for sym, w in pairs
        )
        council_card = f"""
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
        html_out = append_card(html_out, council_card)

    # 5) DNA card (if available)
    dna_png = RUNS / "dna_drift.png"
    if safe_exists(dna_png):
        dna_card = f"""
<section style="border:1px solid #ddd;padding:12px;margin:16px 0;border-radius:8px">
  <h2 style="margin:0 0 8px 0">DNA Drift</h2>
  <img src="{dna_png.as_posix()}" width="720" alt="DNA Drift (avg across symbols)">
  <p style="color:#666;font-size:12px;margin-top:8px">Source: runs_plus/dna_drift.json</p>
</section>
"""
        html_out = append_card(html_out, dna_card)

    # 6) Heartbeat card (if available)
    hb_png = RUNS / "heartbeat.png"
    if safe_exists(hb_png):
        hb_card = f"""
<section style="border:1px solid #ddd;padding:12px;margin:16px 0;border-radius:8px">
  <h2 style="margin:0 0 8px 0">Heartbeat (BPM)</h2>
  <img src="{hb_png.as_posix()}" width="720" alt="Heartbeat Metabolism (BPM)">
  <p style="color:#666;font-size:12px;margin-top:8px">Source: runs_plus/heartbeat.json</p>
</section>
"""
        html_out = append_card(html_out, hb_card)

    # 7) Write the combined report (same familiar name)
    OUT_REPORT.write_text(html_out)
    print("✅ Wrote", OUT_REPORT.as_posix())

    # 8) Open it (macOS)
    try:
        run(["open", OUT_REPORT.as_posix()])
    except Exception as e:
        print("Note: could not open automatically:", e)
        print("You can open it with: open", OUT_REPORT.as_posix())

if __name__ == "__main__":
    main()
