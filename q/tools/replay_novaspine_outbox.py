#!/usr/bin/env python3
# Replay queued NovaSpine outbox batches when NovaSpine API is available.
#
# Reads:
#   runs_plus/novaspine_outbox/*.jsonl
# Writes:
#   runs_plus/novaspine_replay_status.json
#   runs_plus/novaspine_outbox/sent/*.jsonl (on success)

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmods.novaspine_adapter import publish_events  # noqa: E402

RUNS = ROOT / "runs_plus"
OUTBOX = RUNS / "novaspine_outbox"
SENT = OUTBOX / "sent"
RUNS.mkdir(exist_ok=True)
OUTBOX.mkdir(exist_ok=True)
SENT.mkdir(exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


if __name__ == "__main__":
    enabled = str(os.getenv("C3_MEMORY_ENABLE", "0")).strip().lower() in {"1", "true", "yes", "on"}
    backend = str(os.getenv("C3_MEMORY_BACKEND", "novaspine_api")).strip().lower()
    base = str(os.getenv("C3_MEMORY_NOVASPINE_URL", "http://127.0.0.1:8420")).strip() or "http://127.0.0.1:8420"
    token = str(os.getenv("C3AE_API_TOKEN", "")).strip() or None
    namespace = str(os.getenv("C3_MEMORY_NAMESPACE", "private/nova/actions")).strip() or "private/nova/actions"

    files = sorted([p for p in OUTBOX.glob("*.jsonl") if p.is_file()])
    replayed = 0
    failed = 0
    details = []

    if enabled and backend in {"novaspine_api", "novaspine", "c3ae"}:
        for p in files:
            events = _load_jsonl(p)
            if not events:
                continue
            res = publish_events(
                events=events,
                backend="novaspine_api",
                namespace=namespace,
                outbox_dir=OUTBOX,
                novaspine_base_url=base,
                novaspine_token=token,
            )
            ok = int(res.published) >= len(events) and int(res.failed) == 0
            if ok:
                dest = SENT / p.name
                p.rename(dest)
                replayed += int(len(events))
            else:
                failed += int(len(events))
            details.append(
                {
                    "file": str(p),
                    "events": int(len(events)),
                    "published": int(res.published),
                    "failed": int(res.failed),
                    "error": res.error,
                }
            )

    status = {
        "timestamp_utc": _now_iso(),
        "enabled": bool(enabled),
        "backend": backend,
        "novaspine_url": base,
        "queued_files": int(len(files)),
        "replayed_events": int(replayed),
        "failed_events": int(failed),
        "details": details[:50],
    }
    (RUNS / "novaspine_replay_status.json").write_text(json.dumps(status, indent=2))
    print(f"✅ Wrote {RUNS/'novaspine_replay_status.json'}")
