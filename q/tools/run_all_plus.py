#!/usr/bin/env python3
import subprocess, sys, json, html
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
RUNS  = ROOT / "runs_plus"

BASE_REPORT = ROOT / "report_plus.html"
OUT_REPORT  = ROOT / "report_all.html"

def run(cmd):
    print("▶", " ".join(cmd))
    subprocess.check_call(cmd)

def exists(p: Path) -> bool:
    try: return p.exists()
    except Exception: return False

def insert_before_body(html_in: str, card_html: str) -> str:
    if not card_html:
        return html_in
    if "</body>" in html_in:
        return html_in.replace("</body>", card_html + "\n</body>")
    return html_in + card_html

def card_council() -> str:
    p = RUNS / "council.json"
    if not exists(p): return ""
    data = json.loads(p.read_text())
    weights = data.get("final_weights", {}) or {}
    pairs = sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)[:20]
    rows = "\n".join(
        f"<tr><td>{html.escape(sym)}</td><td style='text-align:right'>{w:+.4f}</td></tr>"
        for sym, w in pairs
    ) or "<tr><td colspan='2' style='color:#888'>No council weights yet</td></tr>"
    return f"""
<section style="border:1px solid #ddd;padding:12px;margin:16px 0;border-radius:8px">
  <h2 style="margin:0 0 8px 0">Council (final weights)</h2>
  <table style="width:100%;border-collapse:collapse;font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;font-size:14px">
    <thead><tr><th style="text-align:left">Symbol</th><th style="text-align:right">Weight</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <p style="color:#666;font-size:12px;margin-top:8px">Source: runs_plus/council.json</p>
</section>
"""

def card_img(title: str, img_path: Path, src_note: str) -> str:
    if not exists(img_path): return ""
    return f"""
<section style="border:1px solid #ddd;padding:12px;margin:16px 0;border-radius:8px">
  <h2 style="margin:0 0 8px 0">{html.escape(title)}</h2>
  <img src="{img_path.as_posix()}" width="720" alt="{html.escape(title)}">
  <p style="color:#666;font-size:12px;margin-top:8px">Source: {html.escape(src_note)}</p>
</section>
"""

def card_symbolic() -> str:
    p = RUNS / "symbolic.json"
    if not exists(p): return ""
    js = json.loads(p.read_text())
    scores = js.get("symbolic", {}) or {}
    heads  = js.get("headlines", []) or []
    pairs = sorted(scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:20]
    rows = "\n".join(
        f"<tr><td>{html.escape(sym)}</td><td style='text-align:right'>{val:+.3f}</td></tr>"
        for sym, val in pairs
    ) or "<tr><td colspan='2' style='color:#888'>No scored symbols yet</td></tr>"
    head_rows = "\n".join(
        f"<li>{html.escape(str(h.get('date') or ''))} — "
        f"<a href='{html.escape(str(h.get('url') or '#'))}'>{html.escape(h.get('title',''))}</a>"
        f" <span style='color:#888'>({html.escape(str(h.get('source') or ''))}, score {float(h.get('score',0)):+.0f})</span></li>"
        for h in heads[-10:]
    ) or "<li style='color:#888'>No headlines found</li>"
    return f"""
<section style="border:1px solid #ddd;padding:12px;margin:16px 0;border-radius:8px">
  <h2 style="margin:0 0 8px 0">Symbolic / Affective Ingestion</h2>
  <table style="width:100%;border-collapse:collapse;font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;font-size:14px">
    <thead><tr><th style="text-align:left">Symbol</th><th style="text-align:right">Score</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <h3 style="margin:12px 0 6px 0;font-size:14px">Recent Headlines</h3>
  <ul>{head_rows}</ul>
  <p style="color:#666;font-size:12px;margin-top:8px">Source: runs_plus/symbolic.json</p>
</section>
"""

def card_reflexive() -> str:
    pjson = RUNS / "reflexive.json"
    ppng  = RUNS / "reflexive.png"
    if not exists(pjson): return ""
    data = json.loads(pjson.read_text())
    weights = data.get("weights", {}) or {}
    scaler  = data.get("exposure_scaler", 1.0)
    pairs = sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)[:20]
    rows = "\n".join(
        f"<tr><td>{html.escape(sym)}</td><td style='text-align:right'>{val:+.3f}</td></tr>"
        for sym, val in pairs
    ) or "<tr><td colspan='2' style='color:#888'>No reflexive weights yet</td></tr>"
    img_block = f"<img src='{ppng.as_posix()}' width='720' alt='Reflexive Weights'>" if exists(ppng) else ""
    return f"""
<section style="border:1px solid #ddd;padding:12px;margin:16px 0;border-radius:8px">
  <h2 style="margin:0 0 8px 0">Reflexive Feedback (Q eats its own signals)</h2>
  <p style="margin:4px 0 8px 0;color:#666">Blend of DNA state-change + Symbolic affect, exposure gated by Heartbeat (scaler ≈ {scaler:.2f}).</p>
  {img_block}
  <table style="width:100%;border-collapse:collapse;margin-top:8px;font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;font-size:14px">
    <thead><tr><th style="text-align:left">Symbol</th><th style='text-align:right'>Reflexive Weight</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <p style="color:#666;font-size:12px;margin-top:8px">Sources: runs_plus/dna_drift.json, runs_plus/symbolic.json, runs_plus/heartbeat.json</p>
</section>
"""

def card_hive() -> str:
    pjson = RUNS / "hive.json"
    ppng  = RUNS / "hive.png"
    if not exists(pjson): return ""
    data = json.loads(pjson.read_text())
    hives = data.get("hives", {}) or {}
    top   = data.get("top_by_hive", {}) or {}
    # Build cluster table
    blocks = []
    for h in sorted(hives):
        symlist = ", ".join(hives[h]) if hives[h] else "(none)"
        top_rows = ""
        if top.get(h):
            tr = "".join(f"<tr><td>{html.escape(sym)}</td><td style='text-align:right'>{float(w):+0.3f}</td></tr>"
                         for sym, w in top[h].items())
            top_rows = f"""
            <table style="width:100%;border-collapse:collapse;margin-top:6px">
              <thead><tr><th style="text-align:left">Top by |Council|</th><th style="text-align:right">Weight</th></tr></thead>
              <tbody>{tr}</tbody>
            </table>
            """
        blocks.append(f"""
        <section style="border:1px dashed #ccc;padding:8px;margin:8px 0;border-radius:6px">
          <h3 style="margin:0 0 6px 0;font-size:14px">{html.escape(h)}</h3>
          <div style="font-size:13px;color:#444">{html.escape(symlist)}</div>
          {top_rows}
        </section>
        """)
    img_block = f"<img src='{ppng.as_posix()}' width='720' alt='Hive PCA'>" if exists(ppng) else ""
    return f"""
<section style="border:1px solid #ddd;padding:12px;margin:16px 0;border-radius:8px">
  <h2 style="margin:0 0 8px 0">Hive / Ecosystem</h2>
  <p style="margin:4px 0 8px 0;color:#666">Clusters from PCA(returns) over last window; shows composition and top-|council| per hive.</p>
  {img_block}
  {''.join(blocks)}
  <p style="color:#666;font-size:12px;margin-top:8px">Sources: data/*.csv, runs_plus/council.json</p>
</section>
"""

def card_nested_wf() -> str:
    p = RUNS / "nested_wf_summary.json"
    if not exists(p):
        return ""
    data = json.loads(p.read_text())
    top = data.get("top_configs", []) or []
    rows = "\n".join(
        f"<tr><td>{html.escape(str(item.get('config', '')))}</td><td style='text-align:right'>{int(item.get('count', 0))}</td></tr>"
        for item in top[:8]
    ) or "<tr><td colspan='2' style='color:#888'>No config usage yet</td></tr>"
    return f"""
<section style="border:1px solid #ddd;padding:12px;margin:16px 0;border-radius:8px">
  <h2 style="margin:0 0 8px 0">Nested WF (Purged)</h2>
  <p style="margin:4px 0 8px 0;color:#666">
    Assets: {int(data.get('assets', 0))} |
    Avg OOS Sharpe: {float(data.get('avg_oos_sharpe') or 0.0):.3f} |
    Avg OOS MaxDD: {float(data.get('avg_oos_maxDD') or 0.0):.3f}
  </p>
  <table style="width:100%;border-collapse:collapse;font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;font-size:14px">
    <thead><tr><th style="text-align:left">Top Config</th><th style="text-align:right">Chosen Count</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <p style="color:#666;font-size:12px;margin-top:8px">Source: runs_plus/nested_wf_summary.json</p>
</section>
"""

def main():
    # base
    run([sys.executable, str(TOOLS / "build_report_plus.py")])

    # artifacts
    run([sys.executable, str(TOOLS / "make_council.py")])
    run([sys.executable, str(TOOLS / "make_dna_drift.py")])
    run([sys.executable, str(TOOLS / "make_heartbeat.py")])
    run([sys.executable, str(TOOLS / "make_symbolic.py")])
    run([sys.executable, str(TOOLS / "make_reflexive.py")])
    run([sys.executable, str(TOOLS / "make_hive.py")])
    run([sys.executable, str(TOOLS / "make_hive_council.py")])
    run([sys.executable, str(TOOLS / "nested_wf_lite.py")])

    # assemble
    html_out = BASE_REPORT.read_text()
    html_out = insert_before_body(html_out, card_council())
    html_out = insert_before_body(html_out, card_img("DNA Drift", RUNS / "dna_drift.png", "runs_plus/dna_drift.json"))
    html_out = insert_before_body(html_out, card_img("Heartbeat (BPM)", RUNS / "heartbeat.png", "runs_plus/heartbeat.json"))
    html_out = insert_before_body(html_out, card_symbolic())
    html_out = insert_before_body(html_out, card_reflexive())
    html_out = insert_before_body(html_out, card_hive())
    html_out = insert_before_body(html_out, card_nested_wf())

    OUT_REPORT.write_text(html_out)
    print("✅ Wrote", OUT_REPORT.as_posix())
    try:
        subprocess.check_call(["open", OUT_REPORT.as_posix()])
    except Exception:
        print("Open:", OUT_REPORT.as_posix())

if __name__ == "__main__":
    main()
