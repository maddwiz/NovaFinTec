# tools/build_report_plus.py — cards show in-sample + walk-forward stats
import json, datetime as dt
from pathlib import Path
import html

RUNS = Path("runs_plus")
OUT  = Path("report_plus.html")

def read_json(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return {}

def read_summary(asset_dir: Path):
    m = read_json(asset_dir/"summary.json")
    hit    = m.get("hit_rate", m.get("hit", 0.0))
    sharpe = m.get("sharpe", 0.0)
    maxdd  = m.get("max_dd", m.get("maxDD", 0.0))
    weights = m.get("weights", {})
    dna    = m.get("dna", "dna:NA")
    bpm    = m.get("heartbeat_bpm_latest", "-")
    return hit, sharpe, maxdd, weights, dna, bpm

def read_wf(asset_name: str):
    p = RUNS/"walk_forward"/f"{asset_name}.json"
    if not p.exists(): return None
    w = read_json(p)
    return (w.get("hit",0.0), w.get("sharpe",0.0), w.get("max_dd",0.0))

def alarm_badge(asset_dir: Path):
    arr = read_json(asset_dir/"alarms.json") if (asset_dir/"alarms.json").exists() else []
    if not arr: return ""
    last = arr[-1]
    msg = f"{last.get('type','alarm')} · {last.get('date','-')} · z={last.get('z','-')}"
    return f'<span class="badge">ALERT</span> <span class="small">{html.escape(msg)}</span>'

def news_card():
    p = RUNS/"news.json"
    if not p.exists(): return ""
    try: items = json.loads(p.read_text())
    except Exception: items = []
    if not items: return ""
    lis=[]
    for it in items[-10:][::-1]:
        title = html.escape(it.get("title","(no title)"))
        url   = html.escape(it.get("url",""))
        src   = html.escape(it.get("source",""))
        date  = html.escape(it.get("date",""))
        lis.append(f'<li><a href="{url}" target="_blank" rel="noopener">{title}</a> <span class="small">({src} · {date})</span></li>')
    return f"<div class='card'><div class='card-title'>Trusted News</div><ul>{''.join(lis)}</ul></div>"

def wf_card_table():
    p = RUNS/"walk_forward_table.json"
    if not p.exists(): return ""
    try: rows = json.loads(p.read_text())
    except Exception: rows = []
    if not rows: return ""
    rows = sorted(rows, key=lambda r: (r.get('sharpe',0), r.get('hit',0)), reverse=True)[:12]
    trs=["<tr><th>Asset</th><th>Hit</th><th>Sharpe</th><th>MaxDD</th></tr>"]
    for r in rows:
        a  = html.escape(r.get("asset","?"))
        h,sh,dd = r.get("hit",0.0), r.get("sharpe",0.0), r.get("max_dd",0.0)
        trs.append(f"<tr><td>{a}</td><td>{h:.3f}</td><td>{sh:.3f}</td><td>{dd:.3f}</td></tr>")
    return f"<div class='card'><div class='card-title'>Walk-Forward (Top 12 by Sharpe)</div><table class='tbl'>{''.join(trs)}</table><div class='small'>Full CSV: runs_plus/walk_forward_table.csv</div></div>"

def card_html(asset_dir: Path):
    a = asset_dir.name
    hit_is,sh_is,dd_is,weights,dna,bpm = read_summary(asset_dir) if (asset_dir/"summary.json").exists() else (0,0,0,{}, "dna:NA","-")
    wf = read_wf(a)
    sig_tag = f'<img src="runs_plus/{a}/signals.png" class="img">' if (asset_dir/"signals.png").exists() else "<div class='muted'>signals.png missing</div>"
    dream_mp4 = (asset_dir/"dream.mp4").exists()
    dream_tag = f'<video class="vid" controls loop muted src="runs_plus/{a}/dream.mp4"></video>' if dream_mp4 else (f'<img src="runs_plus/{a}/dream.gif" class="img">' if (asset_dir/"dream.gif").exists() else "<div class='muted'>dream.gif missing</div>")
    overlay_tag = f'<a class="btn" href="runs_plus/{a}/overlay.gif" target="_blank" rel="noopener">Open Overlay</a>' if (asset_dir/"overlay.gif").exists() else "<span class='muted'>no overlay</span>"
    wtxt = f"mom={weights.get('mom',0):.2f}, mr={weights.get('mr',0):.2f}, carry={weights.get('carry',0):.2f}"
    notes = (asset_dir/"notes.txt").read_text() if (asset_dir/"notes.txt").exists() else "None"
    alarm = alarm_badge(asset_dir)

    wf_row = ""
    if wf:
        h,sh,dd = wf
        wf_row = f"<div class='mt4'><b>walk-forward</b> hit {h:.3f} · sharpe {sh:.3f} · maxDD {dd:.3f}</div>"

    return f"""
<div class="card">
  <div class="card-title">{html.escape(a)} {alarm}</div>
  <div class="grid">
    <div>
      <div><b>in-sample</b> hit {hit_is:.3f} · sharpe {sh_is:.3f} · maxDD {dd_is:.3f}</div>
      {wf_row}
      <div class="mt4"><b>weights</b> {wtxt}</div>
      <div class="mt4"><b>DNA</b> {html.escape(str(dna))} · <b>BPM</b> {html.escape(str(bpm))}</div>
      <div class="mt8"><b>Notes</b><br>{html.escape(notes)}</div>
      <div class="mt8">{overlay_tag}</div>
    </div>
    <div>{sig_tag}</div>
    <div>{dream_tag}</div>
  </div>
</div>
"""

def style():
    return """
<style>
body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Ubuntu,Helvetica,Arial,sans-serif;margin:16px;color:#222}
.small{font-weight:500;color:#666;font-size:13px}
.muted{color:#888}
.card{border:1px solid #e3e3e3;border-radius:10px;padding:14px;margin:12px 0;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,0.04)}
.card-title{font-size:18px;font-weight:700;margin-bottom:8px}
.grid{display:grid;grid-template-columns:1.2fr 1fr 1fr;gap:12px;align-items:start}
.badge{display:inline-block;background:#e91515;color:#fff;border-radius:6px;padding:2px 6px;font-size:12px;margin-left:6px}
.tbl{border-collapse:collapse;width:100%}
.tbl th,.tbl td{border:1px solid #eee;padding:6px 8px;text-align:right}
.tbl th:first-child,.tbl td:first-child{text-align:left}
.img{max-width:100%;border:1px solid #ddd;border-radius:6px;margin-top:6px}
.vid{max-width:100%;border:1px solid #ddd;border-radius:6px;margin-top:6px}
.btn{display:inline-block;background:#0b6efd;color:#fff;text-decoration:none;padding:6px 10px;border-radius:6px}
.mt4{margin-top:4px}.mt8{margin-top:8px}
.footer{color:#777;margin-top:20px;font-size:13px}
</style>
"""

def build():
    cards=[card_html(s.parent) for s in sorted(RUNS.glob("*/summary.json"))]
    html_doc=f"""<!doctype html><html><head><meta charset='utf-8'><title>Q Report+</title>{style()}</head><body>
  <h1>Q Report+ <span class='small'>generated {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</span></h1>
  {news_card()}
  {wf_card_table()}
  {''.join(cards)}
  <div class="footer">Cards show in-sample and (if available) walk-forward for each asset.</div>
</body></html>"""
    OUT.write_text(html_doc)
    print("wrote", OUT)

if __name__=="__main__":
    build()
