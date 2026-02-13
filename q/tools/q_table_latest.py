import json, pathlib

ASSETS = ["IWM", "RSP", "LQD_TR", "HYG_TR"]

def read_summary(run_dir: pathlib.Path):
    sj = run_dir / "summary.json"
    if sj.exists():
        try:
            return json.loads(sj.read_text())
        except Exception:
            return None
    return None

rows = []
for a in ASSETS:
    run_dir = pathlib.Path("runs") / a
    m = read_summary(run_dir)
    if m:
        rows.append((a, m.get("hit_rate"), m.get("sharpe"), m.get("max_dd")))

print("ASSET      hit     sharpe    maxDD")
for a, hit, sh, dd in rows:
    try:
        print(f"{a:9s}  {float(hit):.3f}  {float(sh):.3f}  {float(dd):.4f}")
    except Exception:
        print(f"{a:9s}  nan     nan       nan")
