import csv
import errno
import json
import math
import re
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen
from zoneinfo import ZoneInfo

from .. import config as cfg
from .doctor import check_external_overlay
from .runtime_decision import runtime_decision_summary
from .runtime_health import (
    aion_feedback_runtime_info,
    memory_outbox_runtime_info,
    memory_feedback_runtime_info,
    overlay_runtime_status,
    runtime_controls_stale_info,
)

DENVER_TZ = ZoneInfo("America/Denver")


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text())
        return data
    except Exception:
        return default


def _tail_lines(path: Path, limit: int = 20):
    if not path.exists():
        return []
    try:
        lines = path.read_text().splitlines()
        return lines[-max(1, limit):]
    except Exception:
        return []


def _tail_csv(path: Path, limit: int = 30):
    if not path.exists():
        return []
    try:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        return rows[-max(1, limit):]
    except Exception:
        return []


def _tail_jsonl(path: Path, limit: int = 30):
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []
    rows = []
    for ln in lines:
        s = str(ln).strip()
        if not s:
            continue
        try:
            row = json.loads(s)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows[-max(1, limit):]


_INTRADAY_ALIGN_RE = re.compile(r"intraday align\s+([0-9]*\.?[0-9]+)", re.IGNORECASE)


def _to_float(raw, default=None):
    try:
        v = float(raw)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


def _signal_gate_summary(rows: list[dict]) -> dict:
    out = {
        "rows": int(len(rows or [])),
        "considered": 0,
        "passed": 0,
        "blocked_total": 0,
        "blocked_intraday": 0,
        "blocked_mtf": 0,
        "blocked_meta": 0,
        "avg_intraday_score": None,
        "last_intraday_score": None,
        "block_rate": None,
    }
    if not rows:
        return out

    intraday_scores = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        decision = str(row.get("decision", "")).strip().upper()
        reasons = str(row.get("reasons", "")).strip().lower()

        intraday_gate = str(row.get("intraday_gate", "")).strip().lower()
        mtf_gate = str(row.get("mtf_gate", "")).strip().lower()
        meta_gate = str(row.get("meta_gate", "")).strip().lower()

        if not intraday_gate:
            intraday_gate = "block" if "intraday blocked" in reasons else ("pass" if "intraday align" in reasons else "")
        if not mtf_gate:
            mtf_gate = "block" if "mtf blocked" in reasons else ""
        if not meta_gate:
            meta_gate = "block" if "meta-label veto" in reasons else ""

        considered = bool(
            intraday_gate in {"pass", "block"}
            or mtf_gate in {"pass", "block"}
            or meta_gate in {"pass", "block"}
            or decision in {"BUY", "SELL"}
        )
        if considered:
            out["considered"] += 1

        blocked_any = False
        if intraday_gate == "block":
            out["blocked_intraday"] += 1
            blocked_any = True
        if mtf_gate == "block":
            out["blocked_mtf"] += 1
            blocked_any = True
        if meta_gate == "block":
            out["blocked_meta"] += 1
            blocked_any = True
        if blocked_any:
            out["blocked_total"] += 1
        elif decision in {"BUY", "SELL"}:
            out["passed"] += 1

        score = _to_float(row.get("intraday_score"), None)
        if score is None and reasons:
            m = _INTRADAY_ALIGN_RE.search(reasons)
            if m:
                score = _to_float(m.group(1), None)
        if score is not None:
            intraday_scores.append(float(score))

    if intraday_scores:
        out["avg_intraday_score"] = float(sum(intraday_scores) / len(intraday_scores))
        out["last_intraday_score"] = float(intraday_scores[-1])
    if out["considered"] > 0:
        out["block_rate"] = float(out["blocked_total"] / out["considered"])
    return out


def _to_int(raw, default: int):
    try:
        return int(raw)
    except Exception:
        return default


def _parse_timestamp(raw):
    s = str(raw or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except Exception:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def _equity_pnl_summary(path: Path) -> dict:
    rows = _tail_csv(path, limit=5000)
    out = {
        "present": bool(rows),
        "daily_pnl": None,
        "overall_pnl": None,
        "overall_return_pct": None,
    }
    if not rows:
        return out

    def _row_equity(r):
        return _to_float((r or {}).get("equity"), None)

    start_eq = _row_equity(rows[0])
    end_eq = _row_equity(rows[-1])
    if (start_eq is not None) and (end_eq is not None):
        out["overall_pnl"] = float(end_eq - start_eq)
        if abs(start_eq) > 1e-9:
            out["overall_return_pct"] = float((end_eq / start_eq) - 1.0)

    now_denver = datetime.now(DENVER_TZ).date()
    daily_start_eq = None
    for r in rows:
        ts = _parse_timestamp((r or {}).get("timestamp"))
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=DENVER_TZ)
        day_denver = ts.astimezone(DENVER_TZ).date()
        eq = _row_equity(r)
        if (eq is not None) and (day_denver == now_denver):
            daily_start_eq = eq
            break
    if (daily_start_eq is not None) and (end_eq is not None):
        out["daily_pnl"] = float(end_eq - daily_start_eq)

    return out


def _doctor_check(doctor: dict, name: str):
    checks = doctor.get("checks", []) if isinstance(doctor, dict) else []
    if not isinstance(checks, list):
        return None
    for c in checks:
        if not isinstance(c, dict):
            continue
        if str(c.get("name", "")) == str(name):
            return c
    return None


def _mtime(path: Path) -> float | None:
    try:
        return float(path.stat().st_mtime)
    except Exception:
        return None


def _is_aion_dashboard_running(host: str, port: int) -> bool:
    probe_hosts = [host]
    if host in {"0.0.0.0", "::"}:
        probe_hosts.append("127.0.0.1")
    for probe_host in probe_hosts:
        url = f"http://{probe_host}:{port}/api/status"
        try:
            with urlopen(url, timeout=1.0) as resp:
                if resp.status != 200:
                    continue
                payload = json.loads(resp.read().decode("utf-8"))
                if isinstance(payload, dict) and "doctor_ok" in payload and "trading_enabled" in payload:
                    return True
        except Exception:
            continue
    return False


def _status_payload():
    doctor_path = cfg.LOG_DIR / "doctor_report.json"
    doctor = _read_json(doctor_path, {})
    monitor = _read_json(cfg.LOG_DIR / "runtime_monitor.json", {})
    perf = _read_json(cfg.LOG_DIR / "performance_report.json", {})
    profile = _read_json(cfg.STATE_DIR / "strategy_profile.json", {})
    rc_info = runtime_controls_stale_info(
        cfg.STATE_DIR / "runtime_controls.json",
        default_loop_seconds=int(cfg.LOOP_SECONDS),
        base_stale_seconds=int(cfg.OPS_GUARD_TRADE_STALE_SEC),
    )
    runtime_controls = rc_info.get("payload", {})
    ops_guard = _read_json(cfg.OPS_GUARD_STATUS_FILE, {})
    watchlist = _tail_lines(cfg.STATE_DIR / "watchlist.txt", limit=200)
    ext_runtime = overlay_runtime_status(cfg.EXT_SIGNAL_FILE, max_age_hours=float(cfg.EXT_SIGNAL_MAX_AGE_HOURS))

    trade_metrics = perf.get("trade_metrics", {})
    equity_metrics = perf.get("equity_metrics", {})
    signal_rows = _tail_csv(cfg.LOG_DIR / "signals.csv", limit=200)
    signal_gate_summary = _signal_gate_summary(signal_rows)
    pnl_summary = _equity_pnl_summary(cfg.LOG_DIR / "shadow_equity.csv")
    ext = _doctor_check(doctor, "external_overlay") or {}
    ext_details = ext.get("details", {}) if isinstance(ext.get("details"), dict) else {}
    ext_ok = bool(ext.get("ok", True))
    ext_msg = ext.get("msg")
    ext_source = "doctor_report"

    doctor_mtime = _mtime(doctor_path)
    overlay_mtime = _mtime(cfg.EXT_SIGNAL_FILE)
    stale_doctor_vs_overlay = False
    if isinstance(doctor_mtime, float) and isinstance(overlay_mtime, float):
        stale_doctor_vs_overlay = bool(math.isfinite(doctor_mtime) and math.isfinite(overlay_mtime) and (overlay_mtime - doctor_mtime) > 120.0)

    if (not isinstance(ext, dict)) or (len(ext) == 0) or stale_doctor_vs_overlay:
        live_ok, live_msg, live_details = check_external_overlay(
            cfg.EXT_SIGNAL_FILE,
            max_age_hours=float(cfg.EXT_SIGNAL_MAX_AGE_HOURS),
            require_runtime_context=True,
        )
        ext_ok = bool(live_ok)
        ext_msg = live_msg
        if isinstance(live_details, dict):
            ext_details = live_details
        ext_source = "live_check"

    risk_flags = ext_details.get("risk_flags", []) if isinstance(ext_details, dict) else []
    if not isinstance(risk_flags, list):
        risk_flags = []
    risk_flags = [str(x).strip().lower() for x in risk_flags if str(x).strip()]
    runtime_flags = ext_runtime.get("risk_flags", []) if isinstance(ext_runtime, dict) else []
    if not isinstance(runtime_flags, list):
        runtime_flags = []
    runtime_flags = [str(x).strip().lower() for x in runtime_flags if str(x).strip()]
    risk_flags = list(dict.fromkeys(risk_flags + runtime_flags))
    fracture_state = "none"
    if "fracture_alert" in risk_flags:
        fracture_state = "alert"
    elif "fracture_warn" in risk_flags:
        fracture_state = "warn"

    guard_running = {}
    if isinstance(ops_guard, dict):
        guard_running = ops_guard.get("running", {})
    if not isinstance(guard_running, dict):
        guard_running = {}
    target_states = []
    for target in cfg.OPS_GUARD_TARGETS:
        item = guard_running.get(str(target), {}) if isinstance(guard_running, dict) else {}
        running = bool(item.get("running")) if isinstance(item, dict) else False
        target_states.append(running)
    ops_guard_ok = bool(target_states) and all(target_states)
    rc_age = rc_info.get("age_sec")
    rc_threshold = float(rc_info.get("threshold_sec", max(60, int(cfg.LOOP_SECONDS * 6))))
    rc_stale = bool(rc_info.get("stale", False))
    decision = runtime_decision_summary(
        runtime_controls if isinstance(runtime_controls, dict) else {},
        ext_runtime,
        risk_flags,
    )
    aion_feedback_runtime = aion_feedback_runtime_info(
        runtime_controls if isinstance(runtime_controls, dict) else {},
        ext_runtime if isinstance(ext_runtime, dict) else {},
    )
    memory_feedback_runtime = memory_feedback_runtime_info(
        runtime_controls if isinstance(runtime_controls, dict) else {},
        ext_runtime if isinstance(ext_runtime, dict) else {},
    )
    memory_outbox_runtime = memory_outbox_runtime_info(
        runtime_controls if isinstance(runtime_controls, dict) else {},
        warn_files=int(getattr(cfg, "MEMORY_OUTBOX_WARN_FILES", 5)),
        alert_files=int(getattr(cfg, "MEMORY_OUTBOX_ALERT_FILES", 20)),
    )

    return {
        "ib": doctor.get("ib", {}),
        "doctor_ok": bool(doctor.get("ok", False)),
        "doctor_remediation": doctor.get("remediation", []),
        "external_overlay_ok": ext_ok,
        "external_overlay_msg": ext_msg,
        "external_overlay_source": ext_source,
        "external_overlay": ext_details,
        "external_overlay_runtime": ext_runtime,
        "external_overlay_risk_flags": risk_flags,
        "external_fracture_state": fracture_state,
        "ops_guard_ok": ops_guard_ok,
        "ops_guard": ops_guard if isinstance(ops_guard, dict) else {},
        "runtime_controls": runtime_controls if isinstance(runtime_controls, dict) else {},
        "aion_feedback_runtime": aion_feedback_runtime,
        "memory_feedback_runtime": memory_feedback_runtime,
        "memory_outbox_runtime": memory_outbox_runtime,
        "runtime_decision": decision,
        "runtime_remediation": decision.get("recommended_actions", []),
        "runtime_controls_age_sec": rc_age,
        "runtime_controls_stale_threshold_sec": rc_threshold,
        "runtime_controls_stale": rc_stale,
        "monitor_ts": monitor.get("ts"),
        "alert_count": len(monitor.get("alerts", [])),
        "system_event_count": len(monitor.get("system_events", [])),
        "watchlist_count": len(watchlist),
        "trade_metrics": trade_metrics,
        "equity_metrics": equity_metrics,
        "signal_gate_summary": signal_gate_summary,
        "pnl_summary": pnl_summary,
        "telemetry_summary": _read_json(cfg.TELEMETRY_SUMMARY_FILE, {}),
        "adaptive_stats": profile.get("adaptive_stats", {}),
        "trading_enabled": bool(profile.get("trading_enabled", True)),
    }


def _html_template():
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AION Dashboard</title>
  <style>
    :root { color-scheme: dark; --bg:#0d1222; --card:#141d36; --text:#d9e3ff; --muted:#8ea3d1; --ok:#2dd4bf; --bad:#fb7185; --accent:#60a5fa; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; background: radial-gradient(1200px 600px at 20% -20%, #1e3a8a, transparent), var(--bg); color: var(--text); }
    .wrap { max-width: 1200px; margin: 0 auto; padding: 20px; }
    .title { display:flex; align-items:center; justify-content:space-between; margin-bottom: 14px; }
    .badge { font-size:12px; padding:4px 8px; border-radius:999px; background:#1f2a4b; color:var(--muted); }
    .grid { display:grid; gap: 12px; grid-template-columns: repeat(12, minmax(0,1fr)); }
    .card { background: linear-gradient(180deg, #182345, #111937); border:1px solid #243563; border-radius: 12px; padding: 12px; box-shadow: 0 8px 24px rgba(0,0,0,.25); }
    .kpi { grid-column: span 3; }
    .wide { grid-column: span 6; }
    .full { grid-column: span 12; }
    .k { color: var(--muted); font-size: 12px; }
    .v { font-size: 24px; font-weight: 700; margin-top: 4px; }
    table { width:100%; border-collapse: collapse; font-size: 12px; }
    th, td { padding: 6px 8px; border-bottom: 1px solid #22345d; text-align:left; }
    th { color: var(--muted); font-weight:600; }
    .ok { color: var(--ok); }
    .bad { color: var(--bad); }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; white-space: pre-wrap; }
    @media (max-width: 900px){ .kpi,.wide { grid-column: span 12; } }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="title">
      <h2 style="margin:0;">AION Live Dashboard</h2>
      <div class="badge" id="ts">loading...</div>
    </div>
    <div class="grid">
      <div class="card kpi"><div class="k">Doctor</div><div class="v" id="doctor_ok">-</div></div>
      <div class="card kpi"><div class="k">Trading Enabled</div><div class="v" id="trading_enabled">-</div></div>
      <div class="card kpi"><div class="k">Watchlist</div><div class="v" id="watchlist_count">-</div></div>
      <div class="card kpi"><div class="k">Closed Trades</div><div class="v" id="closed_trades">-</div></div>
      <div class="card kpi"><div class="k">Daily PnL (Denver)</div><div class="v" id="daily_pnl">-</div></div>
      <div class="card kpi"><div class="k">Overall PnL</div><div class="v" id="overall_pnl">-</div></div>
      <div class="card kpi"><div class="k">Q Overlay</div><div class="v" id="overlay_ok">-</div></div>
      <div class="card kpi"><div class="k">Ops Guard</div><div class="v" id="ops_guard_ok">-</div></div>
      <div class="card kpi"><div class="k">Entry Gates</div><div class="v" id="gate_summary">-</div></div>
      <div class="card kpi"><div class="k">Rolling Hit (20)</div><div class="v" id="telemetry_hit">-</div></div>
      <div class="card kpi"><div class="k">Win/Loss Ratio</div><div class="v" id="telemetry_wlr">-</div></div>
      <div class="card kpi"><div class="k">Best Regime</div><div class="v" id="telemetry_best_regime">-</div></div>
      <div class="card kpi"><div class="k">Worst Regime</div><div class="v" id="telemetry_worst_regime">-</div></div>
      <div class="card kpi"><div class="k">Top Win Signal</div><div class="v" id="telemetry_win_signal">-</div></div>
      <div class="card kpi"><div class="k">Top Loss Signal</div><div class="v" id="telemetry_loss_signal">-</div></div>

      <div class="card wide">
        <div class="k">System Snapshot</div>
        <div class="mono" id="snapshot">-</div>
      </div>
      <div class="card wide">
        <div class="k">Recent Alerts</div>
        <div class="mono" id="alerts">-</div>
      </div>

      <div class="card full">
        <div class="k">Recent Trades</div>
        <table id="trades_tbl"><thead><tr>
          <th>Time</th><th>Symbol</th><th>Side</th><th>Qty</th><th>PnL</th><th>Reason</th>
        </tr></thead><tbody></tbody></table>
      </div>

      <div class="card full">
        <div class="k">Recent Signals</div>
        <table id="signals_tbl"><thead><tr>
          <th>Time</th><th>Symbol</th><th>Decision</th><th>Intra</th><th>MTF</th><th>Meta</th><th>IntraScore</th><th>L</th><th>S</th><th>Patterns</th><th>Indicators</th>
        </tr></thead><tbody></tbody></table>
      </div>

      <div class="card full">
        <div class="k">Recent Decisions</div>
        <table id="decisions_tbl"><thead><tr>
          <th>Time</th><th>Symbol</th><th>Decision</th><th>Confluence</th><th>Regime</th><th>PnL</th><th>Reasons</th>
        </tr></thead><tbody></tbody></table>
      </div>
    </div>
  </div>
  <script>
    async function j(url){ const r = await fetch(url); return r.json(); }
    function txt(el, v){ document.getElementById(el).textContent = v; }
    function cls(el, ok){ const n=document.getElementById(el); n.classList.remove('ok','bad'); n.classList.add(ok?'ok':'bad'); }
    const DENVER_FMT = new Intl.DateTimeFormat('en-US', {
      timeZone: 'America/Denver',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: 'numeric',
      minute: '2-digit',
      second: '2-digit',
      hour12: true
    });
    function formatDenver(v){
      if(v === null || v === undefined || v === '') return '';
      let raw = String(v).trim();
      if(/^\\d{4}-\\d{2}-\\d{2}$/.test(raw)) raw = `${raw}T00:00:00`;
      if(/^\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}$/.test(raw)) raw = raw.replace(' ','T');
      const d = new Date(raw);
      if(Number.isNaN(d.getTime())) return String(v);
      return `${DENVER_FMT.format(d)} MT`;
    }
    function fmtMoney(v){
      const n = Number(v);
      if(!Number.isFinite(n)) return '-';
      return `${n >= 0 ? '+' : '-'}$${Math.abs(n).toFixed(2)}`;
    }
    function renderTable(id, rows, cols){
      const tb = document.querySelector(`#${id} tbody`); tb.innerHTML='';
      for(const r of rows){
        const tr=document.createElement('tr');
        for(const c of cols){
          const td=document.createElement('td');
          let val = (r[c]??'');
          if(c === 'timestamp' || c === 'time'){
            val = formatDenver(val);
          }
          td.textContent=val;
          tr.appendChild(td);
        }
        tb.appendChild(tr);
      }
    }
    async function refresh(){
      const [s,tr,sg,al,dc] = await Promise.all([
        j('/api/status'),
        j('/api/trades?limit=18'),
        j('/api/signals?limit=18'),
        j('/api/alerts?limit=14'),
        j('/api/decisions?limit=18')
      ]);
      txt('ts', formatDenver(new Date().toISOString()));
      txt('doctor_ok', s.doctor_ok ? 'PASS' : 'FAIL'); cls('doctor_ok', !!s.doctor_ok);
      txt('trading_enabled', s.trading_enabled ? 'ON' : 'OFF'); cls('trading_enabled', !!s.trading_enabled);
      txt('watchlist_count', s.watchlist_count ?? 0);
      txt('closed_trades', s.trade_metrics?.closed_trades ?? 0);
      txt('daily_pnl', fmtMoney(s.pnl_summary?.daily_pnl));
      cls('daily_pnl', Number(s.pnl_summary?.daily_pnl || 0) >= 0);
      txt('overall_pnl', fmtMoney(s.pnl_summary?.overall_pnl));
      cls('overall_pnl', Number(s.pnl_summary?.overall_pnl || 0) >= 0);
      txt('overlay_ok', s.external_overlay_ok ? 'OK' : 'WARN'); cls('overlay_ok', !!s.external_overlay_ok);
      txt('ops_guard_ok', s.ops_guard_ok ? 'OK' : 'WARN'); cls('ops_guard_ok', !!s.ops_guard_ok);
      const gates = s.signal_gate_summary || {};
      const considered = gates.considered ?? 0;
      const blocked = gates.blocked_total ?? 0;
      txt('gate_summary', `${blocked}/${considered}`);
      cls('gate_summary', considered === 0 ? true : (blocked / Math.max(considered, 1)) < 0.65);
      const tel = s.telemetry_summary || {};
      const hit = Number.isFinite(Number(tel.rolling_hit_rate)) ? `${(Number(tel.rolling_hit_rate) * 100).toFixed(1)}%` : '-';
      txt('telemetry_hit', hit);
      cls('telemetry_hit', Number(tel.rolling_hit_rate || 0) >= 0.5);
      const wlrNum = Number(tel.win_loss_ratio);
      const wlr = Number.isFinite(wlrNum) ? wlrNum.toFixed(2) : '-';
      txt('telemetry_wlr', wlr);
      cls('telemetry_wlr', Number.isFinite(wlrNum) ? wlrNum >= 1.0 : true);
      txt('telemetry_best_regime', tel.most_profitable_regime || '-');
      cls('telemetry_best_regime', true);
      txt('telemetry_worst_regime', tel.worst_regime || '-');
      cls('telemetry_worst_regime', false);
      txt('telemetry_win_signal', tel.top_win_signal_category || '-');
      cls('telemetry_win_signal', !!tel.top_win_signal_category);
      txt('telemetry_loss_signal', tel.top_loss_signal_category || '-');
      cls('telemetry_loss_signal', false);
      txt('snapshot', JSON.stringify({
        ib: s.ib,
        external_overlay_ok: s.external_overlay_ok,
        external_overlay: s.external_overlay,
        external_overlay_runtime: s.external_overlay_runtime,
        external_overlay_risk_flags: s.external_overlay_risk_flags,
        external_fracture_state: s.external_fracture_state,
        external_overlay_msg: s.external_overlay_msg,
        ops_guard_ok: s.ops_guard_ok,
        ops_guard: s.ops_guard,
        runtime_controls: s.runtime_controls,
        runtime_decision: s.runtime_decision,
        aion_feedback_runtime: s.aion_feedback_runtime,
        memory_feedback_runtime: s.memory_feedback_runtime,
        memory_outbox_runtime: s.memory_outbox_runtime,
        runtime_controls_age_sec: s.runtime_controls_age_sec,
        runtime_controls_stale_threshold_sec: s.runtime_controls_stale_threshold_sec,
        runtime_controls_stale: s.runtime_controls_stale,
        monitor_ts: s.monitor_ts,
        winrate: s.trade_metrics?.winrate,
        expectancy: s.trade_metrics?.expectancy,
        return_pct: s.equity_metrics?.return_pct,
        pnl_summary: s.pnl_summary,
        signal_gate_summary: s.signal_gate_summary,
        telemetry_summary: s.telemetry_summary,
        adaptive: s.adaptive_stats,
        remediation: s.doctor_remediation
      }, null, 2));
      txt('alerts', (al.lines || []).join('\\n') || 'No alerts.');
      renderTable('trades_tbl', tr.rows || [], ['timestamp','symbol','side','qty','pnl','reason']);
      renderTable(
        'signals_tbl',
        sg.rows || [],
        ['timestamp','symbol','decision','intraday_gate','mtf_gate','meta_gate','intraday_score','long_conf','short_conf','pattern_hits','indicator_hits']
      );
      renderTable(
        'decisions_tbl',
        dc.rows || [],
        ['timestamp','symbol','decision','confluence_score','regime','pnl_realized','reasons']
      );
    }
    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = 200):
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query or "")
        limit = _to_int((qs.get("limit") or ["20"])[0], 20)

        if path == "/":
            self._send_html(_html_template())
            return
        if path == "/api/status":
            self._send_json(_status_payload())
            return
        if path == "/api/trades":
            rows = _tail_csv(cfg.LOG_DIR / "shadow_trades.csv", limit=limit)
            self._send_json({"rows": rows})
            return
        if path == "/api/signals":
            rows = _tail_csv(cfg.LOG_DIR / "signals.csv", limit=limit)
            self._send_json({"rows": rows})
            return
        if path == "/api/equity":
            rows = _tail_csv(cfg.LOG_DIR / "shadow_equity.csv", limit=limit)
            self._send_json({"rows": rows})
            return
        if path == "/api/alerts":
            lines = _tail_lines(cfg.LOG_DIR / "alerts.log", limit=limit)
            self._send_json({"lines": lines})
            return
        if path == "/api/decisions":
            p = Path(cfg.STATE_DIR) / str(getattr(cfg, "TELEMETRY_DECISIONS_FILE", "trade_decisions.jsonl"))
            rows = _tail_jsonl(p, limit=limit)
            self._send_json({"rows": rows})
            return
        self._send_json({"error": "not found"}, status=404)

    def log_message(self, format, *args):
        return


def main() -> int:
    host = cfg.DASHBOARD_HOST
    port = int(cfg.DASHBOARD_PORT)
    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            if _is_aion_dashboard_running(host, port):
                print(f"[AION] Dashboard already running at http://{host}:{port}")
                return 0
            print(f"[AION] ERROR: dashboard port {host}:{port} is already in use by another process.")
            return 1
        raise
    print(f"[AION] Dashboard running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
