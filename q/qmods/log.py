import pathlib, json, datetime as dt

def append_growth_log(entry: dict, log_file: pathlib.Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"- {ts} | asset={entry.get('asset')} | hit={entry.get('hit_rate'):.3f} | sh={entry.get('sharpe'):.3f} | dna={entry.get('dna')} | drift={entry.get('dna_drift_pct')}\n"
    with log_file.open("a") as f:
        f.write(line)
