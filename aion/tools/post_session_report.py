#!/usr/bin/env python3
"""
Build a concise end-of-session report for paper-live operations.

Outputs:
  - state/post_session_report.json
  - state/post_session_report.md

The script refreshes telemetry/performance summaries first, then composes a
single operator-facing snapshot with the key numbers.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aion import config as cfg  # type: ignore
from aion.exec.performance_report import main as run_performance_report  # type: ignore
from aion.exec.telemetry_summary import write_telemetry_summary  # type: ignore


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _fmt_pct(x) -> str:
    try:
        return f"{float(x):.2%}"
    except Exception:
        return "n/a"


def _fmt_num(x, digits: int = 3) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "n/a"


def _build_markdown(payload: dict) -> str:
    readiness = payload.get("readiness", {})
    checks = readiness.get("checks", {}) if isinstance(readiness.get("checks"), dict) else {}
    strict = checks.get("strict_oos_net", {}) if isinstance(checks.get("strict_oos_net"), dict) else {}
    telemetry = payload.get("telemetry_summary", {})
    perf = payload.get("performance_report", {})
    trade = perf.get("trade_metrics", {}) if isinstance(perf.get("trade_metrics"), dict) else {}
    equity = perf.get("equity_metrics", {}) if isinstance(perf.get("equity_metrics"), dict) else {}
    soft = readiness.get("soft_warnings", []) if isinstance(readiness.get("soft_warnings"), list) else []
    hard = readiness.get("hard_blockers", []) if isinstance(readiness.get("hard_blockers"), list) else []

    lines = [
        "# AION Post-Session Report",
        "",
        f"Generated (UTC): {payload.get('generated_at_utc', 'n/a')}",
        "",
        "## Readiness",
        f"- Gate status: {'PASS' if bool(readiness.get('ok', False)) else 'FAIL'}",
        f"- Hard blockers: {len(hard)}",
        f"- Soft warnings: {len(soft)}",
        "",
        "## Q Validation Snapshot",
        f"- OOS Sharpe: {_fmt_num(strict.get('sharpe'))}",
        f"- OOS Hit Rate: {_fmt_pct(strict.get('hit_rate'))}",
        f"- OOS Max Drawdown: {_fmt_pct(strict.get('max_drawdown'))}",
        "",
        "## Session Trades",
        f"- Closed trades: {int(_safe_float(trade.get('closed_trades'), 0))}",
        f"- Win rate: {_fmt_pct(trade.get('winrate'))}",
        f"- Profit factor: {_fmt_num(trade.get('profit_factor'), 2)}",
        f"- Net PnL: {_fmt_num(trade.get('net_pnl'), 2)}",
        "",
        "## Telemetry",
        f"- Rolling hit rate (20): {_fmt_pct(telemetry.get('rolling_hit_rate'))}",
        f"- Win/Loss ratio: {_fmt_num(telemetry.get('win_loss_ratio'), 2)}",
        f"- Avg slippage (bps): {_fmt_num(telemetry.get('avg_slippage_bps'), 2)}",
        f"- Slippage delta vs est (bps): {_fmt_num(telemetry.get('slippage_delta_bps'), 2)}",
        f"- Best regime: {telemetry.get('most_profitable_regime', 'n/a')}",
        f"- Worst regime: {telemetry.get('worst_regime', 'n/a')}",
        "",
        "## Equity",
        f"- Start equity: {_fmt_num(equity.get('start_equity'), 2)}",
        f"- End equity: {_fmt_num(equity.get('end_equity'), 2)}",
        f"- Session return: {_fmt_pct(equity.get('return_pct'))}",
        f"- Max drawdown: {_fmt_pct(equity.get('max_drawdown'))}",
        "",
        "## Artifacts",
        f"- Readiness: `{payload.get('paths', {}).get('readiness', '')}`",
        f"- Telemetry summary: `{payload.get('paths', {}).get('telemetry_summary', '')}`",
        f"- Performance report: `{payload.get('paths', {}).get('performance_report', '')}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    state_dir = Path(os.getenv("AION_STATE_DIR", str(cfg.STATE_DIR)))
    log_dir = Path(os.getenv("AION_LOG_DIR", str(cfg.LOG_DIR)))
    state_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    readiness_path = state_dir / "paper_live_readiness.json"
    telemetry_path = state_dir / "telemetry_summary.json"
    perf_path = log_dir / "performance_report.json"
    report_json = state_dir / "post_session_report.json"
    report_md = state_dir / "post_session_report.md"

    # Refresh telemetry summary from decision logs.
    decisions_name = str(getattr(cfg, "TELEMETRY_DECISIONS_FILE", "trade_decisions.jsonl"))
    decisions_path = state_dir / decisions_name
    write_telemetry_summary(
        decisions_path=decisions_path,
        output_path=telemetry_path,
        rolling_window=20,
    )

    # Refresh performance report from shadow trade/equity logs.
    try:
        run_performance_report()
    except Exception:
        pass

    payload = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "readiness": _read_json(readiness_path),
        "telemetry_summary": _read_json(telemetry_path),
        "performance_report": _read_json(perf_path),
        "paths": {
            "readiness": str(readiness_path),
            "telemetry_summary": str(telemetry_path),
            "performance_report": str(perf_path),
            "report_json": str(report_json),
            "report_md": str(report_md),
        },
    }

    _write_json(report_json, payload)
    report_md.write_text(_build_markdown(payload), encoding="utf-8")

    print(f"✅ Wrote {report_json}")
    print(f"✅ Wrote {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
