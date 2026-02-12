import csv
import errno
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

from .. import config as cfg


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


def _to_int(raw, default: int):
    try:
        return int(raw)
    except Exception:
        return default


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
    doctor = _read_json(cfg.LOG_DIR / "doctor_report.json", {})
    monitor = _read_json(cfg.LOG_DIR / "runtime_monitor.json", {})
    perf = _read_json(cfg.LOG_DIR / "performance_report.json", {})
    profile = _read_json(cfg.STATE_DIR / "strategy_profile.json", {})
    watchlist = _tail_lines(cfg.STATE_DIR / "watchlist.txt", limit=200)

    trade_metrics = perf.get("trade_metrics", {})
    equity_metrics = perf.get("equity_metrics", {})

    return {
        "ib": doctor.get("ib", {}),
        "doctor_ok": bool(doctor.get("ok", False)),
        "doctor_remediation": doctor.get("remediation", []),
        "monitor_ts": monitor.get("ts"),
        "alert_count": len(monitor.get("alerts", [])),
        "system_event_count": len(monitor.get("system_events", [])),
        "watchlist_count": len(watchlist),
        "trade_metrics": trade_metrics,
        "equity_metrics": equity_metrics,
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
          <th>Time</th><th>Symbol</th><th>Decision</th><th>L</th><th>S</th><th>Patterns</th><th>Indicators</th>
        </tr></thead><tbody></tbody></table>
      </div>
    </div>
  </div>
  <script>
    async function j(url){ const r = await fetch(url); return r.json(); }
    function txt(el, v){ document.getElementById(el).textContent = v; }
    function cls(el, ok){ const n=document.getElementById(el); n.classList.remove('ok','bad'); n.classList.add(ok?'ok':'bad'); }
    function renderTable(id, rows, cols){
      const tb = document.querySelector(`#${id} tbody`); tb.innerHTML='';
      for(const r of rows){
        const tr=document.createElement('tr');
        for(const c of cols){ const td=document.createElement('td'); td.textContent=(r[c]??''); tr.appendChild(td); }
        tb.appendChild(tr);
      }
    }
    async function refresh(){
      const [s,tr,sg,al] = await Promise.all([
        j('/api/status'),
        j('/api/trades?limit=18'),
        j('/api/signals?limit=18'),
        j('/api/alerts?limit=14')
      ]);
      txt('ts', new Date().toLocaleString());
      txt('doctor_ok', s.doctor_ok ? 'PASS' : 'FAIL'); cls('doctor_ok', !!s.doctor_ok);
      txt('trading_enabled', s.trading_enabled ? 'ON' : 'OFF'); cls('trading_enabled', !!s.trading_enabled);
      txt('watchlist_count', s.watchlist_count ?? 0);
      txt('closed_trades', s.trade_metrics?.closed_trades ?? 0);
      txt('snapshot', JSON.stringify({
        ib: s.ib,
        monitor_ts: s.monitor_ts,
        winrate: s.trade_metrics?.winrate,
        expectancy: s.trade_metrics?.expectancy,
        return_pct: s.equity_metrics?.return_pct,
        adaptive: s.adaptive_stats,
        remediation: s.doctor_remediation
      }, null, 2));
      txt('alerts', (al.lines || []).join('\\n') || 'No alerts.');
      renderTable('trades_tbl', tr.rows || [], ['timestamp','symbol','side','qty','pnl','reason']);
      renderTable('signals_tbl', sg.rows || [], ['timestamp','symbol','decision','long_conf','short_conf','pattern_hits','indicator_hits']);
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
