import pathlib, json

def load_metrics(run_path):
    # Look for summary.json first, then metrics.json
    for fname in ["summary.json", "metrics.json"]:
        f = run_path / fname
        if f.exists():
            try:
                return json.loads(f.read_text())
            except json.JSONDecodeError:
                return None
    return None

print("ASSET      hit    sharpe   maxDD")
for run in sorted(pathlib.Path("runs").iterdir()):
    if run.is_dir():
        m = load_metrics(run)
        if m and "asset" in m:
            hit = m.get("hit_rate", float("nan"))
            sharpe = m.get("sharpe", float("nan"))
            max_dd = m.get("max_dd", float("nan"))
            print(f"{m['asset']:<9}  {hit:.3f}  {sharpe:.3f}  {max_dd:.4f}")
