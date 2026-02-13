import json, pathlib, datetime as dt

RUNS = pathlib.Path("runs_plus")
ASSETS = ["IWM","RSP","LQD_TR","HYG_TR"]

def _safe_read_json(p: pathlib.Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def card(asset: str) -> str:
    summary_path = RUNS / asset / "summary.json"
    alarms_path  = RUNS / asset / "alarms.json"

    m = _safe_read_json(summary_path) or {}
    alarms = _safe_read_json(alarms_path) or []

    dream = f"runs_plus/{asset}/dream.gif"
    sigs  = f"runs_plus/{asset}/signals.png"

    # show at most 5 alarms
    alarms_html = "<br>".join(
        f"{a.get('date','?')}: z={a.get('z',0):.2f}, drift={a.get('drift',0):.3f}"
        for a in alarms[:5]
    ) or "None"

    w = m.get("weights", {})
    hit = m.get("hit_rate", float("nan"))
    sh  = m.get("sharpe",  float("nan"))
    dd  = m.get("max_dd",  float("nan"))
    bpm = m.get("heartbeat_bpm_latest", "-")
    dna = m.get("dna", "dna:NA")

    return f"""
    <div class="card">
      <h3>{asset}</h3>
      <p><b>hit</b> {hit:.3f} &nbsp; <b>sharpe</b> {sh:.3f} &nbsp; <b>maxDD</b> {dd:.4f}</p>
      <p><b>weights</b> mom={w.get('mom',0):.2f}, mr={w.get('mr',0):.2f}, carry={w.get('carry',0):.2f}</p>
      <p><b>DNA</b> {dna} &nbsp; <b>BPM</b> {bpm}</p>
      <p><b>Drift alarms</b><br>{alarms_html}</p>
      <div class="row">
        <img src="{dream}" class="img">
        <img src="{sigs}" class="img">
      </div>
    </div>
    """

page = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Q Report</title>
<style>
body {{ font-family: -apple-system, Arial, sans-serif; margin: 24px; }}
h1 {{ margin-top: 0; }}
.card {{ border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin-bottom: 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }}
.row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
.img {{ width: 360px; border: 1px solid #eee; border-radius: 8px; }}
.small {{ color: #666; font-size: 12px; }}
</style>
</head>
<body>
<h1>Q Report <span class="small">generated {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</span></h1>
{"".join(card(a) for a in ASSETS)}
</body>
</html>
"""

pathlib.Path("report.html").write_text(page)
print("wrote report.html")
