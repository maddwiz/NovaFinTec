from __future__ import annotations

import json
import math
import time
from pathlib import Path


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def read_runtime_controls(path: Path) -> dict:
    if not path.exists() or path.is_dir():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def runtime_controls_age_sec(path: Path, now_ts: float | None = None) -> float | None:
    if not path.exists():
        return None
    try:
        st = path.stat()
    except Exception:
        return None
    ts = time.time() if now_ts is None else float(now_ts)
    age = ts - float(st.st_mtime)
    if not math.isfinite(age):
        return None
    return float(max(0.0, age))


def runtime_controls_stale_threshold_sec(
    payload: dict | None,
    *,
    default_loop_seconds: int,
    base_stale_seconds: int,
) -> float:
    p = payload if isinstance(payload, dict) else {}
    loop_seconds = max(5, _safe_int(p.get("loop_seconds"), max(5, int(default_loop_seconds))))
    watchlist_size = max(0, _safe_int(p.get("watchlist_size"), 0))

    # Baseline expects at least ~6 loops worth of updates.
    base_dynamic = loop_seconds * 6
    # Larger watchlists can make a single loop substantially longer.
    watchlist_factor = min(18, watchlist_size // 12)
    with_watchlist = loop_seconds * (6 + watchlist_factor)
    dynamic = max(base_dynamic, with_watchlist)

    floor = max(60, int(base_stale_seconds))
    ceiling = loop_seconds * 24
    return float(max(floor, min(dynamic, ceiling)))


def runtime_controls_stale_info(
    path: Path,
    *,
    default_loop_seconds: int,
    base_stale_seconds: int,
    now_ts: float | None = None,
) -> dict:
    payload = read_runtime_controls(path)
    age = runtime_controls_age_sec(path, now_ts=now_ts)
    threshold = runtime_controls_stale_threshold_sec(
        payload,
        default_loop_seconds=default_loop_seconds,
        base_stale_seconds=base_stale_seconds,
    )
    stale = bool(age is not None and age > threshold)
    return {
        "age_sec": age,
        "threshold_sec": threshold,
        "stale": stale,
        "payload": payload,
    }
