#!/usr/bin/env python3
# qmods/symbolic.py
# Minimal "Symbolic / Affective Ingestion" engine.
# Reads news/comments CSV/JSON, turns text into daily signals per asset.
#
# Inputs (any that exist will be read, others are skipped):
#   data_news/*.csv     with columns: date, asset(optional), text
#   data_news/*.jsonl   with keys: date, asset(optional), text
#   runs_plus/trusted_news.csv  (same columns)
#
# Outputs:
#   runs_plus/symbolic_events.csv     (raw scored rows)
#   runs_plus/symbolic_signal.csv     (daily signal per asset)
#   runs_plus/symbolic_summary.json   (top words, data counts)

from pathlib import Path
import json, math, re, csv
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
NEWS_DIR = ROOT / "data_news"

SAFE_ASSET = "ALL"

STOP = set("""
a an the and or of for to in on at from by with without into over under as is are was were be been being
this that these those it its they them he she we you i our your their my me us his her
""".split())

POS = set("""
beat beats breakout bullish improvement improved improving upside expansion accelerate tailwind strong stronger strength
surge record high highs growth positive profit profitable profits margin margins upgrade outperform winner
""".split())

NEG = set("""
miss misses warning bearish downgrade downgrade cut cuts decline declining declined risk risks caution cautious weak weakness
selloff drawdown loss losses negative downgrade downgrade halt halted lawsuit failure fail failed
""".split())

FEAR = set("""
war wars conflict conflicts strike strikes default defaults crisis crises crash crashed crisis contagion fear panic panic-selling
sanction sanctions recession recessionary inflation stagflation volatility volatile uncertainty uncertain
""".split())

NEGATION = {"no", "not", "never", "none", "without", "hardly", "barely", "neither", "nor"}
BOOST = {
    "very": 1.25,
    "extremely": 1.55,
    "massive": 1.45,
    "strongly": 1.30,
    "sharply": 1.30,
    "significantly": 1.25,
}
DAMP = {
    "slightly": 0.70,
    "somewhat": 0.80,
    "mildly": 0.75,
    "partly": 0.85,
}

WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+")
UPPER_TICK_RE = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z])?\b")
CASHTAG_RE = re.compile(r"\$([A-Z]{1,5}(?:\.[A-Z])?)\b")

def _read_possible_sources():
    rows = []
    nev = RUNS / "news_events.csv"
    if nev.exists():
        try:
            df = pd.read_csv(nev)
            rows.extend(_rows_from_df(df))
        except Exception:
            pass
    njs = RUNS / "news.json"
    if njs.exists():
        try:
            obj = json.loads(njs.read_text())
            items = obj if isinstance(obj, list) else obj.get("headlines", [])
            for h in items:
                rows.append(
                    {
                        "date": h.get("date") or h.get("timestamp"),
                        "asset": h.get("asset", SAFE_ASSET),
                        "text": " ".join([str(h.get("title", "")), str(h.get("text", ""))]).strip(),
                    }
                )
        except Exception:
            pass
    # trusted_news.csv if present
    tn = RUNS / "trusted_news.csv"
    if tn.exists():
        try:
            df = pd.read_csv(tn)
            rows.extend(_rows_from_df(df))
        except Exception:
            pass
    # CSVs in data_news/
    if NEWS_DIR.exists():
        for p in NEWS_DIR.glob("*.csv"):
            try:
                df = pd.read_csv(p)
                rows.extend(_rows_from_df(df))
            except Exception:
                continue
        # JSONL in data_news/
        for p in NEWS_DIR.glob("*.jsonl"):
            try:
                with p.open() as f:
                    for line in f:
                        if not line.strip(): continue
                        obj = json.loads(line)
                        rows.append({
                            "date": obj.get("date"),
                            "asset": obj.get("asset", SAFE_ASSET),
                            "text": obj.get("text","")
                        })
            except Exception:
                continue
    return rows

def _rows_from_df(df: pd.DataFrame):
    lowers = {c.lower(): c for c in df.columns}
    dcol = lowers.get("date") or lowers.get("timestamp") or lowers.get("time") or list(df.columns)[0]
    acol = lowers.get("asset")
    tcol = lowers.get("text") or lowers.get("headline") or lowers.get("title") or lowers.get("body")
    out = []
    for _, r in df.iterrows():
        date = r.get(dcol)
        asset = r.get(acol) if acol else SAFE_ASSET
        text = r.get(tcol, "")
        out.append({"date": date, "asset": str(asset) if pd.notna(asset) else SAFE_ASSET, "text": str(text or "")})
    return out

def _tokenize(text: str):
    toks = [w.lower() for w in WORD_RE.findall(text)]
    return [t for t in toks if t not in STOP]

def _score_text(text: str):
    raw = WORD_RE.findall(str(text or ""))
    if not raw:
        return 0.0, 0.0, 0.0, [], {"pos": 0.0, "neg": 0.0, "fear": 0.0, "len": 0}

    toks = [w.lower() for w in raw]
    c = Counter([t for t in toks if t not in STOP])

    pos_score = 0.0
    neg_score = 0.0
    fear_score = 0.0
    neg_window = 0
    pending_mult = 1.0

    for tok_raw, tok in zip(raw, toks):
        if tok in BOOST:
            pending_mult *= BOOST[tok]
            continue
        if tok in DAMP:
            pending_mult *= DAMP[tok]
            continue
        if tok in NEGATION:
            neg_window = 3
            pending_mult = 1.0
            continue

        mult = pending_mult
        pending_mult = 1.0
        if tok_raw.isupper() and len(tok_raw) >= 3:
            mult *= 1.10

        if tok in POS:
            sc = 1.0 * mult
            if neg_window > 0:
                sc = -0.80 * sc
            if sc >= 0:
                pos_score += sc
            else:
                neg_score += -sc

        if tok in NEG:
            sc = 1.0 * mult
            if neg_window > 0:
                sc = -0.80 * sc
            if sc >= 0:
                neg_score += sc
            else:
                pos_score += -sc

        if tok in FEAR:
            sc = 1.0 * mult
            if neg_window > 0:
                sc *= 0.50
            fear_score += sc

        if neg_window > 0:
            neg_window -= 1

    n = max(1, len(toks))
    sent = (pos_score - neg_score) / math.sqrt(n + 1.0)
    affect = fear_score / math.sqrt(n + 1.0)

    punct = min(4, str(text).count("!"))
    punct_mult = 1.0 + 0.05 * punct
    confidence_mass = (abs(pos_score) + abs(neg_score) + fear_score)
    confidence = np.clip((confidence_mass / math.sqrt(n + 1.0)) * punct_mult, 0.0, 1.0)

    top = [w for w, _ in c.most_common(5)]
    aux = {"pos": float(pos_score), "neg": float(neg_score), "fear": float(fear_score), "len": int(n)}
    return float(sent), float(affect), float(confidence), top, aux


def _load_asset_universe():
    assets = set()
    for p in (ROOT / "data").glob("*.csv"):
        sym = p.stem.replace("_prices", "").upper().strip()
        if sym:
            assets.add(sym)
            assets.add(sym.replace(".", ""))
    return assets

def _build_asset_aliases(known_assets: set[str]):
    aliases = {}
    for sym in sorted(known_assets):
        if not sym:
            continue
        aliases[sym.lower()] = sym
    names = RUNS / "asset_names.csv"
    if names.exists():
        try:
            df = pd.read_csv(names)
            lowers = {c.lower(): c for c in df.columns}
            sym_col = lowers.get("asset") or lowers.get("symbol") or lowers.get("ticker")
            name_col = lowers.get("name") or lowers.get("asset_name")
            if sym_col and name_col:
                for _, row in df.iterrows():
                    sym = str(row.get(sym_col, "")).upper().strip()
                    if sym not in known_assets:
                        continue
                    nm = str(row.get(name_col, "")).lower().strip()
                    if nm:
                        aliases[nm] = sym
        except Exception:
            pass
    return aliases


def _infer_assets_from_text(text: str, known_assets: set[str], aliases: dict[str, str] | None = None):
    if not known_assets:
        return []
    found = []
    txt = str(text or "")
    for tok in CASHTAG_RE.findall(txt):
        t = tok.upper()
        if t in known_assets:
            found.append(t)
        else:
            td = t.replace(".", "")
            if td in known_assets:
                found.append(td)
    for tok in UPPER_TICK_RE.findall(str(text or "")):
        t = tok.upper()
        if t in known_assets:
            found.append(t)
            continue
        td = t.replace(".", "")
        if td in known_assets:
            found.append(td)
    if aliases:
        low = txt.lower()
        for k, sym in aliases.items():
            if len(k) < 4:
                continue
            if k in low:
                found.append(sym)
    uniq = []
    seen = set()
    for t in found:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    return uniq

def build_events():
    rows = _read_possible_sources()
    if not rows:
        return pd.DataFrame(columns=["DATE","ASSET","text","sent","affect","confidence","pos","neg","fear","len","top_words"])
    known_assets = _load_asset_universe()
    aliases = _build_asset_aliases(known_assets)
    # clean / score
    out = []
    for r in rows:
        try:
            d = pd.to_datetime(r["date"], errors="coerce")
            if pd.isna(d): 
                continue
            raw_asset = str(r.get("asset") or SAFE_ASSET).upper().strip()
            text = r.get("text") or ""
            sent, affect, confidence, top, aux = _score_text(text)
            assets = [raw_asset]
            if raw_asset in ("", SAFE_ASSET) or (known_assets and raw_asset not in known_assets):
                inferred = _infer_assets_from_text(text, known_assets, aliases=aliases)
                assets = inferred if inferred else [SAFE_ASSET]

            for asset in assets:
                out.append({
                    "DATE": d.normalize(),
                    "ASSET": str(asset).upper(),
                    "text": text,
                    "sent": sent,
                    "affect": affect,
                    "confidence": confidence,
                    "pos": aux["pos"], "neg": aux["neg"], "fear": aux["fear"], "len": aux["len"],
                    "top_words": ",".join(top)
                })
        except Exception:
            continue
    df = pd.DataFrame(out).sort_values(["DATE","ASSET"])
    return df

def _tanh_zscore(s: pd.Series, win=63):
    x = s.astype(float).rolling(win, min_periods=max(5, win//3)).agg(['mean','std'])
    mu = x['mean']; sd = x['std'].replace(0, np.nan)
    z = (s - mu) / sd
    z = z.replace([np.inf,-np.inf], np.nan).fillna(0.0)
    return np.tanh(z)

def build_daily_signal(events: pd.DataFrame):
    if events.empty:
        return pd.DataFrame(columns=["DATE","ASSET","sym_signal","sym_sent","sym_affect"])
    # aggregate per day/asset (mean)
    events = events.copy()
    events["DATE"] = pd.to_datetime(events["DATE"], errors="coerce")
    events = events.dropna(subset=["DATE"]).sort_values(["DATE", "ASSET"])
    events["confidence"] = pd.to_numeric(events.get("confidence", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if len(events):
        days_old = (events["DATE"].max() - events["DATE"]).dt.days.clip(lower=0).astype(float)
        decay = np.exp(-np.log(2.0) * (days_old / 30.0))
    else:
        decay = 1.0
    events["w"] = np.maximum(events["confidence"], 0.10) * decay

    def _wavg(g, col):
        w = g["w"].values.astype(float)
        x = pd.to_numeric(g[col], errors="coerce").fillna(0.0).values.astype(float)
        ws = float(w.sum())
        return float(np.dot(w, x) / (ws + 1e-12))

    events["sent_w"] = events["sent"] * events["w"]
    events["affect_w"] = events["affect"] * events["w"]
    grp = events.groupby(["DATE", "ASSET"], as_index=False).agg(
        w_sum=("w", "sum"),
        sent_w=("sent_w", "sum"),
        affect_w=("affect_w", "sum"),
        confidence=("confidence", "mean"),
        events_n=("text", "count"),
    )
    grp["sent"] = grp["sent_w"] / (grp["w_sum"] + 1e-12)
    grp["affect"] = grp["affect_w"] / (grp["w_sum"] + 1e-12)
    grp["confidence"] = grp["confidence"].clip(0.0, 1.0)
    # map to bounded signal with rolling standardization
    sigs = []
    for asset, g in grp.groupby("ASSET"):
        g = g.sort_values("DATE").copy()
        g["sym_sent"] = _tanh_zscore(g["sent"])
        g["sym_affect"] = _tanh_zscore(g["affect"])
        # combine: sentiment minus affect (fear as risk)
        raw = g["sym_sent"] - 0.5*g["sym_affect"]
        g["sym_signal"] = np.clip(raw * (0.50 + 0.50 * g["confidence"]), -1.0, 1.0)
        g["sym_regime"] = np.tanh(_tanh_zscore(g["events_n"].astype(float), win=63))
        g["ASSET"] = asset
        sigs.append(g[["DATE","ASSET","sym_signal","sym_sent","sym_affect","sym_regime","confidence","events_n"]])
    out = pd.concat(sigs, ignore_index=True).sort_values(["DATE","ASSET"])
    return out

def summarize(events: pd.DataFrame):
    info = {
        "rows": int(len(events)),
        "assets": sorted(list(map(str, events["ASSET"].unique()))) if not events.empty else [],
        "date_min": str(events["DATE"].min().date()) if not events.empty else None,
        "date_max": str(events["DATE"].max().date()) if not events.empty else None,
        "top_words": {}
    }
    if not events.empty:
        # top words overall (crude)
        bag = Counter()
        for s in events["text"].astype(str).tolist():
            bag.update(_tokenize(s))
        info["top_words"] = dict(bag.most_common(20))
    return info

def build_symbolic_latent(sig: pd.DataFrame):
    if sig.empty:
        return pd.DataFrame(columns=["DATE", "symbolic_latent", "symbolic_confidence", "symbolic_event_intensity"])
    s = sig.copy()
    s["DATE"] = pd.to_datetime(s["DATE"], errors="coerce")
    s = s.dropna(subset=["DATE"])
    s["confidence"] = pd.to_numeric(s.get("confidence", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    s["events_n"] = pd.to_numeric(s.get("events_n", 1.0), errors="coerce").fillna(1.0).clip(lower=0.0)
    s["w"] = np.maximum(s["confidence"], 0.10)
    s["wx"] = s["w"] * pd.to_numeric(s["sym_signal"], errors="coerce").fillna(0.0)
    out = s.groupby("DATE", as_index=False).agg(
        w_sum=("w", "sum"),
        wx=("wx", "sum"),
        symbolic_confidence=("confidence", "mean"),
        symbolic_event_intensity=("events_n", "sum"),
    )
    out["symbolic_latent"] = np.clip(out["wx"] / (out["w_sum"] + 1e-12), -1.0, 1.0)
    return out[["DATE", "symbolic_latent", "symbolic_confidence", "symbolic_event_intensity"]].sort_values("DATE")

def run_symbolic():
    RUNS.mkdir(parents=True, exist_ok=True)
    ev = build_events()
    ev.to_csv(RUNS/"symbolic_events.csv", index=False)
    sig = build_daily_signal(ev)
    sig.to_csv(RUNS/"symbolic_signal.csv", index=False)
    latent = build_symbolic_latent(sig)
    latent.to_csv(RUNS/"symbolic_latent.csv", index=False)
    info = summarize(ev)
    info["latent_rows"] = int(len(latent))
    (RUNS/"symbolic_summary.json").write_text(json.dumps(info, indent=2))
    return ev, sig, info

if __name__ == "__main__":
    ev, sig, info = run_symbolic()
    print("Symbolic rows:", len(ev), "| days×assets:", len(sig))
    print("Span:", info.get("date_min"), "→", info.get("date_max"))
    print("Assets:", ", ".join(info.get("assets", [])) or "(none)")
