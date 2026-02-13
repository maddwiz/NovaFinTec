import csv, json, pathlib, re
from urllib.parse import urlparse

DATA   = pathlib.Path("data")
RUNS   = pathlib.Path("runs_plus")
CONFIG = pathlib.Path("config/trusted_sources.json")
NEWS_CSV = DATA/"news.csv"   # expected columns: Date,Source,URL,Headline,Tickers

def _load_allowlist():
    if CONFIG.exists():
        try:
            return set(json.loads(CONFIG.read_text()).get("allow", []))
        except Exception:
            pass
    return set()

def _domain_ok(src: str, allow: set[str]) -> bool:
    # allow if empty allowlist (no filter) or domain suffix matches
    try:
        netloc = urlparse(src).netloc or src
        netloc = netloc.lower()
        return (not allow) or any(netloc.endswith(dom) for dom in allow)
    except Exception:
        return False

def _norm_tickers(s: str):
    if not s: return []
    toks = re.split(r"[,\s;/]+", s.strip().upper())
    return [t for t in toks if t]

def main():
    if not NEWS_CSV.exists():
        print("No data/news.csv. Skipping news ingestion.")
        return
    allow = _load_allowlist()

    with NEWS_CSV.open() as f:
        rdr = csv.DictReader(f)
        per_asset = {}
        for row in rdr:
            date = (row.get("Date") or "").strip()
            src  = (row.get("Source") or "").strip()
            url  = (row.get("URL") or "").strip()
            head = (row.get("Headline") or "").strip()
            tick = _norm_tickers(row.get("Tickers") or "")

            if not date or not head:
                continue
            if not _domain_ok(src or url, allow):
                continue

            for t in tick:
                # match asset folder name (case-insensitive)
                # e.g., SPY, QQQ, IWM, RSP, LQD_TR, HYG_TR
                tkey = t.upper()
                per_asset.setdefault(tkey, []).append({
                    "date": date,
                    "source": src or url,
                    "headline": head,
                    "url": url
                })

    # write per-asset events.json where we have runs_plus/<ASSET>
    for ap in RUNS.glob("*"):
        if not ap.is_dir(): continue
        a = ap.name.upper()
        events = per_asset.get(a, [])
        if events:
            (ap/"events.json").write_text(json.dumps(events[:50], indent=2))
            print(f"Wrote {ap}/events.json ({len(events)} items)")
        else:
            # ensure empty file exists
            (ap/"events.json").write_text("[]")

if __name__ == "__main__":
    main()
