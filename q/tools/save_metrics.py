import re, sys, pathlib, json
out = pathlib.Path(sys.argv[1])
txt = (out/"stdout.log").read_text()
m = re.search(r"\{.*\}", txt, flags=re.S)
(out/"metrics.json").write_text(m.group(0) if m else '{"note":"no-json"}')
print("saved:", out/"metrics.json")
