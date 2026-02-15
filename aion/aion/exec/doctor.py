import csv
import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from ib_insync import IB

from .. import config as cfg


EXPECTED_HEADERS = {
    "shadow_trades.csv": [
        "timestamp",
        "symbol",
        "side",
        "qty",
        "entry",
        "exit",
        "pnl",
        "reason",
        "confidence",
        "regime",
        "stop",
        "target",
        "trail",
        "fill_ratio",
        "slippage_bps",
    ],
    "shadow_equity.csv": ["timestamp", "equity", "cash", "open_pnl", "closed_pnl"],
    "signals.csv": [
        "timestamp",
        "symbol",
        "regime",
        "long_conf",
        "short_conf",
        "decision",
        "meta_prob",
        "mtf_score",
        "pattern_hits",
        "indicator_hits",
        "reasons",
    ],
}


def _candidate_ports() -> list[int]:
    ports = [int(cfg.IB_PORT)]
    for p in getattr(cfg, "IB_PORT_CANDIDATES", []):
        try:
            ports.append(int(p))
        except Exception:
            continue
    uniq = []
    seen = set()
    for p in ports:
        if p <= 0 or p in seen:
            continue
        uniq.append(p)
        seen.add(p)
    return uniq


def _candidate_hosts() -> list[str]:
    hosts = [str(cfg.IB_HOST)]
    for h in getattr(cfg, "IB_HOST_CANDIDATES", []):
        hs = str(h).strip()
        if hs:
            hosts.append(hs)
    uniq = []
    seen = set()
    for h in hosts:
        k = h.lower()
        if k in seen:
            continue
        uniq.append(h)
        seen.add(k)
    return uniq


def check_port(host: str, port: int) -> tuple[bool, str]:
    try:
        with socket.create_connection((host, int(port)), timeout=1.5):
            pass
        return True, f"Connected to {host}:{port}"
    except Exception as exc:
        return False, f"Port check failed for {host}:{port}: {exc}"


def check_ib_handshake(host: str, port: int, base_client_id: int) -> tuple[bool, str]:
    attempts = [base_client_id + 100, base_client_id + 101]
    last_exc = None
    for cid in attempts:
        client = IB()
        try:
            ok = client.connect(host, port, clientId=cid, timeout=6)
            if ok and client.isConnected():
                return True, f"IB API handshake ok ({host}:{port}, clientId={cid})"
        except Exception as exc:
            last_exc = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                client.disconnect()
            except Exception:
                pass
    return False, f"IB API handshake failed ({host}:{port}): {last_exc or 'unknown error'}"


def check_port_candidates(hosts: list[str], ports: list[int]):
    rows = []
    for host in hosts:
        for port in ports:
            ok, msg = check_port(host, int(port))
            rows.append({"host": str(host), "port": int(port), "ok": ok, "msg": msg})
    selected = next(({"host": r["host"], "port": r["port"]} for r in rows if r["ok"]), None)
    return selected is not None, selected, rows


def check_handshake_candidates(hosts: list[str], ports: list[int], base_client_id: int):
    rows = []
    for host in hosts:
        for port in ports:
            ok, msg = check_ib_handshake(host, int(port), base_client_id)
            rows.append({"host": str(host), "port": int(port), "ok": ok, "msg": msg})
    selected = next(({"host": r["host"], "port": r["port"]} for r in rows if r["ok"]), None)
    return selected is not None, selected, rows


def check_csv_schema(path: Path, expected: list[str]) -> tuple[bool, str]:
    if not path.exists():
        return True, f"Missing {path.name} (will be created by runtime)"
    try:
        with path.open(newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
    except Exception as exc:
        return False, f"Cannot read {path.name}: {exc}"

    if header == expected:
        return True, f"Schema ok: {path.name}"
    if path.name == "signals.csv":
        legacy = [
            "timestamp",
            "symbol",
            "regime",
            "long_conf",
            "short_conf",
            "decision",
            "meta_prob",
            "mtf_score",
            "reasons",
        ]
        if header == legacy:
            return True, "Schema ok (legacy): signals.csv will auto-migrate on first runtime write"
    return False, f"Schema mismatch in {path.name}"


def _overlay_age_hours(path: Path):
    try:
        ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return None
    return float((datetime.now(timezone.utc) - ts).total_seconds() / 3600.0)


def _parse_utc_ts(raw):
    s = str(raw or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def check_external_overlay(
    path: Path,
    max_age_hours: float = 12.0,
    require_runtime_context: bool = False,
):
    if not path.exists():
        return False, f"Missing external overlay file: {path}", {"exists": False}

    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        return False, f"Invalid external overlay JSON: {exc}", {"exists": True}

    if not isinstance(payload, dict):
        return False, "External overlay payload must be a JSON object", {"exists": True}

    age_mtime_h = _overlay_age_hours(path)
    age_h = None
    age_source = "unknown"
    generated_at_utc = None
    if isinstance(payload, dict):
        gen_dt = _parse_utc_ts(payload.get("generated_at_utc", payload.get("generated_at")))
        if gen_dt is not None:
            generated_at_utc = gen_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            age_h = float((datetime.now(timezone.utc) - gen_dt).total_seconds() / 3600.0)
            age_source = "payload"
    if age_h is None and age_mtime_h is not None:
        age_h = float(age_mtime_h)
        age_source = "mtime"
    if age_h is None:
        return False, "Cannot determine external overlay file age", {"exists": True}

    signals = payload.get("signals", {})
    sig_count = len(signals) if isinstance(signals, dict) else 0
    degraded = bool(payload.get("degraded_safe_mode", False))
    qg = payload.get("quality_gate", {})
    qg_ok = bool(qg.get("ok", True)) if isinstance(qg, dict) else True
    rc = payload.get("runtime_context", {})
    rc_ok = isinstance(rc, dict) and len(rc) > 0
    risk_flags = rc.get("risk_flags", []) if isinstance(rc, dict) else []
    if not isinstance(risk_flags, list):
        risk_flags = []
    risk_flags = [str(x) for x in risk_flags if str(x).strip()]

    max_age = max(0.5, float(max_age_hours))
    issues = []
    is_stale = False
    if age_h > max_age:
        is_stale = True
        issues.append(f"stale>{max_age:.1f}h ({age_h:.2f}h)")
    if degraded:
        issues.append("degraded_safe_mode=true")
    if not qg_ok:
        issues.append("quality_gate_not_ok")
    if require_runtime_context and not rc_ok:
        issues.append("runtime_context_missing")
    if is_stale and "overlay_stale" not in [str(x).strip().lower() for x in risk_flags]:
        risk_flags.append("overlay_stale")

    ok = len(issues) == 0
    msg = (
        "External overlay healthy"
        if ok
        else "External overlay issues: " + ", ".join(issues)
    )
    details = {
        "exists": True,
        "age_hours": age_h,
        "age_source": age_source,
        "generated_at_utc": generated_at_utc,
        "max_age_hours": max_age,
        "signals": int(sig_count),
        "degraded_safe_mode": degraded,
        "quality_gate_ok": qg_ok,
        "runtime_context_present": rc_ok,
        "risk_flags": risk_flags,
        "source_mode": payload.get("source_mode"),
    }
    return ok, msg, details


def _ib_remediation(checks: list[dict], host: str, ports: list[int], hosts: list[str]) -> list[str]:
    tips = []
    if not checks:
        return tips
    port_check = checks[0] if len(checks) > 0 else {}
    hs_check = checks[1] if len(checks) > 1 else {}
    if port_check.get("ok") and hs_check.get("ok"):
        return tips

    tips.append("Open IB Gateway/TWS and log into PAPER account.")
    tips.append("Enable API access: Global Configuration -> API -> Settings -> 'Enable ActiveX and Socket Clients'.")
    tips.append(f"Verify socket bind and trusted host include {host} (and loopback aliases: {hosts}).")
    tips.append(
        f"Confirm API endpoint is one of hosts {hosts} and ports {ports}; set AION_IB_HOST_CANDIDATES / AION_IB_PORT_CANDIDATES if different."
    )
    tips.append("Disable 'Read-Only API' for paper-trade order simulation.")
    tips.append("Ensure no stale client ID conflict; AION retries multiple client IDs automatically.")
    proc_check = next((c for c in checks if c.get("name") == "ib_process_health"), None)
    if proc_check and not proc_check.get("ok", True):
        msg = str(proc_check.get("msg", ""))
        if "Multiple IB process" in msg:
            tips.append("Close duplicate IB Gateway/TWS instances; keep only one active API session.")
        elif "No listener found" in msg:
            tips.append("IB process is running but API listener is absent: finish login and confirm API socket settings/port in TWS/IBG.")
    return tips


def _overlay_remediation(checks: list[dict], overlay_path: Path) -> list[str]:
    ext = next((c for c in checks if str(c.get("name", "")) == "external_overlay"), None)
    if not isinstance(ext, dict) or bool(ext.get("ok", True)):
        return []
    tips = []
    details = ext.get("details", {}) if isinstance(ext.get("details"), dict) else {}
    if not details.get("exists", False):
        tips.append(f"External overlay missing. Generate it with Q pipeline and ensure path exists: {overlay_path}")
    age_h = details.get("age_hours", None)
    max_age_h = details.get("max_age_hours", None)
    try:
        age_h = float(age_h) if age_h is not None else None
        max_age_h = float(max_age_h) if max_age_h is not None else None
    except Exception:
        age_h = None
        max_age_h = None
    if age_h is not None and max_age_h is not None and age_h > max_age_h:
        tips.append(
            "External overlay is stale. Re-run Q export (`python tools/run_all_in_one_plus.py`) "
            "or lower AION_EXT_SIGNAL_MAX_AGE_HOURS if intentionally slower."
        )
    if bool(details.get("degraded_safe_mode", False)):
        tips.append("Q overlay is in degraded safe mode. Check Q health alerts and resolve upstream gating conditions.")
    if not bool(details.get("quality_gate_ok", True)):
        tips.append("Q quality gate is not OK. Review `runs_plus/health_alerts.json` and `runs_plus/system_health.json` in Q.")
    if not bool(details.get("runtime_context_present", True)):
        tips.append("Overlay runtime_context missing. Ensure Q exporter is updated and writing runtime_context in q_signal_overlay.json.")
    risk_flags = details.get("risk_flags", [])
    if not isinstance(risk_flags, list):
        risk_flags = []
    flags = [str(x).strip().lower() for x in risk_flags if str(x).strip()]
    if "fracture_alert" in flags:
        tips.append("Q regime fracture ALERT is active. Keep AION in defensive mode and reduce max_open_positions until fracture score normalizes.")
    elif "fracture_warn" in flags:
        tips.append("Q regime fracture WARN is active. Consider lower risk-per-trade and tighter concurrency until conditions stabilize.")
    return tips


def check_ib_process_health(ports: list[int]):
    lsof_cmd = "lsof -nP -iTCP -sTCP:LISTEN 2>/dev/null"
    ps_cmd = "ps aux"
    try:
        lsof_out = subprocess.run(["sh", "-lc", lsof_cmd], capture_output=True, text=True, timeout=4).stdout
    except Exception:
        lsof_out = ""
    try:
        ps_out = subprocess.run(["sh", "-lc", ps_cmd], capture_output=True, text=True, timeout=4).stdout
    except Exception:
        ps_out = ""

    listener_lines = []
    for line in (lsof_out or "").splitlines():
        for p in ports:
            token = f":{int(p)}"
            if token in line:
                listener_lines.append(line.strip())
                break

    proc_lines = []
    for line in (ps_out or "").splitlines():
        low = line.lower()
        if ("ib gateway" in low or "ibgateway" in low or "trader workstation" in low or "tws" in low) and "rg -i" not in low:
            proc_lines.append(line.strip())

    warnings = []
    if len(proc_lines) > 1:
        warnings.append(f"Multiple IB process candidates detected ({len(proc_lines)}).")
    if len(listener_lines) > 1:
        warnings.append(f"Multiple listeners found on candidate ports ({len(listener_lines)} entries).")
    if len(listener_lines) == 0:
        warnings.append("No listener found on candidate ports.")

    ok = len(warnings) == 0
    msg = "IB process/listener health looks stable." if ok else " | ".join(warnings)
    details = {"listeners": listener_lines, "processes": proc_lines}
    return ok, msg, details


def main() -> int:
    checks = []
    hosts = _candidate_hosts()
    ports = _candidate_ports()

    ok, selected_port, port_rows = check_port_candidates(hosts, ports)
    msg = (
        f"IB TCP reachability ok on {selected_port['host']}:{selected_port['port']}"
        if ok
        else f"IB TCP reachability failed on candidate endpoints hosts={hosts} ports={ports}"
    )
    checks.append({"ok": ok, "msg": msg, "critical": True, "details": port_rows})

    ok_hs, selected_hs_port, hs_rows = check_handshake_candidates(hosts, ports, int(cfg.IB_CLIENT_ID))
    msg_hs = (
        f"IB API handshake ok on {selected_hs_port['host']}:{selected_hs_port['port']}"
        if ok_hs
        else f"IB API handshake failed on candidate endpoints hosts={hosts} ports={ports}"
    )
    checks.append({"ok": ok_hs, "msg": msg_hs, "critical": True, "details": hs_rows})

    ok_proc, msg_proc, details_proc = check_ib_process_health(ports)
    checks.append({"name": "ib_process_health", "ok": ok_proc, "msg": msg_proc, "critical": False, "details": details_proc})

    required_state = ["watchlist.txt", "watchlist.json"]
    optional_state = ["strategy_profile.json", "runtime_state.json", "meta_model.json"]
    optional_logs = ["killswitch.json"]

    for name in required_state:
        path = cfg.STATE_DIR / name
        exists = path.exists()
        checks.append({"ok": exists, "msg": f"{'Found' if exists else 'Missing'} {path}", "critical": True})

    for name in optional_state:
        path = cfg.STATE_DIR / name
        exists = path.exists()
        checks.append({"ok": True, "msg": f"{'Found' if exists else 'Missing optional'} {path}", "critical": False})

    for name in optional_logs:
        path = cfg.LOG_DIR / name
        exists = path.exists()
        checks.append({"ok": True, "msg": f"{'Found' if exists else 'Missing optional'} {path}", "critical": False})

    for file_name, expected in EXPECTED_HEADERS.items():
        ok, msg = check_csv_schema(cfg.LOG_DIR / file_name, expected)
        checks.append({"ok": ok, "msg": msg, "critical": False})

    if cfg.EXT_SIGNAL_ENABLED:
        ok_ext, msg_ext, ext_details = check_external_overlay(
            cfg.EXT_SIGNAL_FILE,
            max_age_hours=cfg.EXT_SIGNAL_MAX_AGE_HOURS,
            require_runtime_context=cfg.EXT_SIGNAL_REQUIRE_RUNTIME_CONTEXT,
        )
        checks.append(
            {
                "name": "external_overlay",
                "ok": ok_ext,
                "msg": msg_ext,
                "critical": bool(cfg.EXT_SIGNAL_CRITICAL),
                "details": ext_details,
            }
        )
    else:
        checks.append(
            {
                "name": "external_overlay",
                "ok": True,
                "msg": "External overlay disabled by config",
                "critical": False,
            }
        )

    summary = {
        "ok": all(c["ok"] for c in checks if c.get("critical", False)),
        "checks": checks,
        "ib": {
            "configured_host": cfg.IB_HOST,
            "configured_port": cfg.IB_PORT,
            "candidate_hosts": hosts,
            "candidate_ports": ports,
            "recommended_host": (selected_hs_port or selected_port or {}).get("host"),
            "recommended_port": (selected_hs_port or selected_port or {}).get("port"),
        },
        "external_overlay": {
            "enabled": bool(cfg.EXT_SIGNAL_ENABLED),
            "path": str(cfg.EXT_SIGNAL_FILE),
            "max_age_hours": float(cfg.EXT_SIGNAL_MAX_AGE_HOURS),
            "require_runtime_context": bool(cfg.EXT_SIGNAL_REQUIRE_RUNTIME_CONTEXT),
            "critical": bool(cfg.EXT_SIGNAL_CRITICAL),
        },
        "paths": {
            "state": str(cfg.STATE_DIR),
            "logs": str(cfg.LOG_DIR),
            "universe": str(cfg.UNIVERSE_DIR),
        },
        "remediation": _ib_remediation(checks, cfg.IB_HOST, ports, hosts)
        + _overlay_remediation(checks, cfg.EXT_SIGNAL_FILE),
    }

    out = cfg.LOG_DIR / "doctor_report.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
