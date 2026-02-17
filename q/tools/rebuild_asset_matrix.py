#!/usr/bin/env python3
# Rebuilds aligned dates + asset return matrix from data/*.csv and data_new/*.csv
# Outputs to runs_plus/:
#   dates.csv                [T] ISO dates
#   asset_names.csv          [N] asset names (file stems)
#   asset_returns.csv        [T x N] daily simple returns
#
# Robust to headers: looks for DATE and (Adj Close|Close|close)

import csv, datetime as dt
import json
import os
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"; RUNS.mkdir(exist_ok=True)
DATA_DIRS = [ROOT/"data", ROOT/"data_new"]
RET_CLIP_ABS = float(np.clip(float(os.getenv("Q_ASSET_RET_CLIP", "0.35")), 0.01, 5.0))


def _sanitize_returns(r: np.ndarray) -> tuple[np.ndarray, int]:
    arr = np.asarray(r, float).ravel()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    clipped = np.clip(arr, -0.95, RET_CLIP_ABS)
    n_clip = int(np.sum(np.abs(clipped - arr) > 1e-12))
    return clipped, n_clip

def read_one(fp: Path):
    # returns (dates list[str], close list[float]) or (None, None) if bad
    try:
        with fp.open() as f:
            rdr = csv.DictReader(f)
            # Normalize header keys
            rows = []
            for row in rdr:
                rows.append({k.strip().upper(): v for k, v in row.items()})
            if not rows:
                return None, None
            # pick date + price columns
            date_key = None
            for dk in ["DATE","TIMESTAMP","TIME","DT"]:
                if dk in rows[0]:
                    date_key = dk; break
            price_key = None
            for pk in ["ADJ CLOSE","CLOSE","PX_LAST","PRICE","VALUE","ADJ_CLOSE","ADJ. CLOSE"]:
                if pk in rows[0]:
                    price_key = pk; break
            if date_key is None or price_key is None:
                return None, None
            ds, ps = [], []
            for r in rows:
                d = r.get(date_key, "").strip()
                v = r.get(price_key, "").replace(",","").strip()
                if not d or not v: continue
                ds.append(d)
                try:
                    ps.append(float(v))
                except:
                    # skip bad lines
                    ds.pop()
            # sort by date ascending
            def to_dt(x):
                for fmt in ("%Y-%m-%d","%m/%d/%Y","%d-%b-%Y","%Y/%m/%d"):
                    try: return dt.datetime.strptime(x, fmt)
                    except: pass
                return None
            pairs = [(to_dt(d), d, p) for d,p in zip(ds,ps)]
            pairs = [(t,d,p) for (t,d,p) in pairs if t is not None]
            pairs.sort(key=lambda x: x[0])
            if len(pairs) < 10:
                return None, None
            dates = [d for _,d,_ in pairs]
            prices = np.array([p for *_,p in pairs], float)
            # returns (T-1,)
            rets = np.diff(prices) / (prices[:-1] + 1e-12)
            rets, _n_clip = _sanitize_returns(rets)
            dates = dates[1:]  # align to returns
            return dates, rets
    except Exception:
        return None, None

if __name__ == "__main__":
    series = []
    names = []
    date_sets = []
    clip_events = 0

    for d in DATA_DIRS:
        if not d.exists(): continue
        for fp in sorted(d.glob("*.csv")):
            dates, r = read_one(fp)
            if dates is None or r is None: continue
            rr, n_clip = _sanitize_returns(r)
            series.append((dates, rr))
            names.append(fp.stem)
            date_sets.append(set(dates))
            clip_events += int(n_clip)

    if not series:
        print("(!) No usable CSVs found in data/ or data_new/."); raise SystemExit(0)

    # intersect dates across all series for clean alignment
    common = set.intersection(*date_sets)
    if not common:
        print("(!) No overlapping dates across assets."); raise SystemExit(0)

    # build ordered common date list
    # pick the longest date array as reference for ordering
    ref_dates = max(series, key=lambda x: len(x[0]))[0]
    ordered = [d for d in ref_dates if d in common]
    if len(ordered) < 50:
        print("(!) Too few overlapping dates after intersection."); raise SystemExit(0)

    # build matrix [T x N]
    T = len(ordered); N = len(series)
    M = np.full((T, N), np.nan, float)
    for j, (dates, r) in enumerate(series):
        idx = {d:i for i,d in enumerate(dates)}
        for t, d in enumerate(ordered):
            i = idx.get(d)
            if i is not None:
                M[t, j] = r[i]

    # drop rows with any NaNs to keep it simple
    mask = ~np.any(np.isnan(M), axis=1)
    M = M[mask]
    dates_out = [d for d, keep in zip(ordered, mask) if keep]

    # save outputs
    np.savetxt(RUNS/"asset_returns.csv", M, delimiter=",")
    (RUNS/"asset_names.csv").write_text("\n".join(names), encoding="utf-8")
    (RUNS/"dates.csv").write_text("\n".join(dates_out), encoding="utf-8")
    (RUNS/"asset_returns_info.json").write_text(
        json.dumps(
            {
                "rows": int(M.shape[0]),
                "assets": int(M.shape[1]),
                "asset_return_clip_abs": float(RET_CLIP_ABS),
                "clip_events": int(clip_events),
                "asset_return_min": float(np.min(M)),
                "asset_return_max": float(np.max(M)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"✅ Wrote runs_plus/asset_returns.csv [{M.shape[0]} x {M.shape[1]}], asset_names.csv, dates.csv")
