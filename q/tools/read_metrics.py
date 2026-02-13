import json, sys, pathlib, re

run = pathlib.Path(sys.argv[1])
# 1) prefer summary.json (written by the runner)
sj = run / "summary.json"
if sj.exists():
    print(sj.read_text())
    sys.exit(0)

# 2) fallback: try to pull a JSON block from stdout.log
sl = run / "stdout.log"
if sl.exists():
    txt = sl.read_text()
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if m:
        print(m.group(0))
        sys.exit(0)

# 3) if nothing found, print a stub
print(json.dumps({"asset": run.name, "hit_rate": None, "sharpe": None, "max_dd": None}))
