#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SAFETY="${AION_SAFETY_MODE:-paper-only}"
if [[ "${1:-}" == "paper-only" || "${1:-}" == "live" ]]; then
  SAFETY="$1"
  shift || true
fi

if [[ "$SAFETY" == "paper-only" ]]; then
  export AION_PAPER_MODE="${AION_PAPER_MODE:-1}"
  export AION_BLOCK_LIVE_ORDERS="${AION_BLOCK_LIVE_ORDERS:-1}"
  echo "[SAFETY] Paper-only mode: live order submission BLOCKED"
elif [[ "$SAFETY" == "live" ]]; then
  export AION_PAPER_MODE="${AION_PAPER_MODE:-0}"
  export AION_BLOCK_LIVE_ORDERS="${AION_BLOCK_LIVE_ORDERS:-0}"
  echo "[SAFETY] LIVE mode: orders WILL be submitted to IBKR"
  echo "[SAFETY] Press Ctrl+C within 5 seconds to abort..."
  sleep 5
else
  echo "Usage: ./run_aion.sh [paper-only|live]"
  exit 1
fi

export AION_HOME="${AION_HOME:-$ROOT}"
export AION_STATE_DIR="${AION_STATE_DIR:-$ROOT/state}"
export AION_LOG_DIR="${AION_LOG_DIR:-$ROOT/logs}"
export AION_Q_HOME="${AION_Q_HOME:-$ROOT/../q}"
export IB_HOST="${IB_HOST:-127.0.0.1}"
export IB_PORT="${IB_PORT:-4002}"
export AION_MODE="${AION_MODE:-brain}"
export AION_TASK="${AION_TASK:-trade}"
export AION_TRADING_MODE="${AION_TRADING_MODE:-long_term}"
export IB_CLIENT_ID="${IB_CLIENT_ID:-2731}"
export AION_HIST_USE_RTH="${AION_HIST_USE_RTH:-0}"
export AION_MAX_TRADES_PER_DAY="${AION_MAX_TRADES_PER_DAY:-15}"
export AION_SKIP_DOCTOR="${AION_SKIP_DOCTOR:-1}"
export AION_SKIP_UNIVERSE_SCAN="${AION_SKIP_UNIVERSE_SCAN:-1}"
export AION_EXT_SIGNAL_BLOCK_CRITICAL_FLAGS="${AION_EXT_SIGNAL_BLOCK_CRITICAL_FLAGS:-fracture_alert,drift_alert,exec_risk_hard,hive_stress_alert,hive_crowding_alert,hive_entropy_alert,hive_turnover_alert,memory_turnover_alert,nested_leakage_alert}"
export AION_DEGRADED_STARTUP="${AION_DEGRADED_STARTUP:-1}"
export AION_AUTO_TUNE_ON_START="${AION_AUTO_TUNE_ON_START:-0}"
export AION_AUTO_CLEAN_STALE_WORKERS="${AION_AUTO_CLEAN_STALE_WORKERS:-1}"
export AION_AUTO_RESOLVE_IB_CONFLICT="${AION_AUTO_RESOLVE_IB_CONFLICT:-1}"
export AION_AUTO_RESTART_IB_ON_TIMEOUT="${AION_AUTO_RESTART_IB_ON_TIMEOUT:-1}"
export AION_AUTO_IB_WARMUP="${AION_AUTO_IB_WARMUP:-1}"
export AION_AUTO_REFRESH_Q_OVERLAY="${AION_AUTO_REFRESH_Q_OVERLAY:-1}"
export AION_MEMORY_ENABLE="${AION_MEMORY_ENABLE:-1}"
export AION_MEMORY_BACKEND="${AION_MEMORY_BACKEND:-${C3_MEMORY_BACKEND:-filesystem}}"
export AION_MEMORY_NOVASPINE_URL="${AION_MEMORY_NOVASPINE_URL:-${C3_MEMORY_NOVASPINE_URL:-http://127.0.0.1:8420}}"
export AION_MEMORY_NAMESPACE="${AION_MEMORY_NAMESPACE:-${C3_MEMORY_NAMESPACE:-private/nova/actions}}"
export AION_MEMORY_OUTBOX_DIR="${AION_MEMORY_OUTBOX_DIR:-$AION_LOG_DIR/novaspine_outbox_aion}"
export AION_MEMORY_TOKEN="${AION_MEMORY_TOKEN:-${C3AE_API_TOKEN:-${C3_MEMORY_TOKEN:-}}}"
export AION_MEMORY_FAIL_COOLDOWN_SEC="${AION_MEMORY_FAIL_COOLDOWN_SEC:-120}"

TRADING_MODE_CANON="$(echo "${AION_TRADING_MODE}" | tr '[:upper:]' '[:lower:]')"
if [[ "$TRADING_MODE_CANON" == "day" || "$TRADING_MODE_CANON" == "daytrading" || "$TRADING_MODE_CANON" == "day_trading" || "$TRADING_MODE_CANON" == "intraday" || "$TRADING_MODE_CANON" == "skimmer" || "$TRADING_MODE_CANON" == "day_skimmer" ]]; then
  export AION_TRADING_MODE="day_skimmer"
  export AION_HIST_BAR_SIZE="${AION_HIST_BAR_SIZE:-1 min}"
  export AION_HIST_DURATION="${AION_HIST_DURATION:-3 D}"
  export AION_HIST_USE_RTH="${AION_HIST_USE_RTH:-0}"
  export AION_LOOP_SECONDS="${AION_LOOP_SECONDS:-12}"
  export AION_MAX_TRADES_PER_DAY="${AION_MAX_TRADES_PER_DAY:-30}"
  export AION_MAX_OPEN_POSITIONS="${AION_MAX_OPEN_POSITIONS:-8}"
  export AION_RISK_PER_TRADE="${AION_RISK_PER_TRADE:-0.008}"
  export AION_ENTRY_THRESHOLD_LONG="${AION_ENTRY_THRESHOLD_LONG:-0.63}"
  export AION_ENTRY_THRESHOLD_SHORT="${AION_ENTRY_THRESHOLD_SHORT:-0.63}"
  export AION_SIGNAL_MIN_MARGIN="${AION_SIGNAL_MIN_MARGIN:-0.08}"
  export AION_STOP_ATR_MULT="${AION_STOP_ATR_MULT:-0.90}"
  export AION_TARGET_ATR_MULT="${AION_TARGET_ATR_MULT:-1.20}"
  export AION_TRAIL_ATR_MULT="${AION_TRAIL_ATR_MULT:-0.70}"
  export AION_BREAKEVEN_R="${AION_BREAKEVEN_R:-0.55}"
  export AION_PARTIAL_TAKE_R="${AION_PARTIAL_TAKE_R:-0.80}"
  export AION_MAX_HOLD_CYCLES="${AION_MAX_HOLD_CYCLES:-6}"
  export AION_MTF_CONFIRM_ENABLED="${AION_MTF_CONFIRM_ENABLED:-0}"
  export AION_INTRADAY_CONFIRM_ENABLED="${AION_INTRADAY_CONFIRM_ENABLED:-1}"
  export AION_INTRADAY_MIN_ALIGNMENT_SCORE="${AION_INTRADAY_MIN_ALIGNMENT_SCORE:-0.60}"
  export AION_INTRADAY_OPEN_RANGE_MIN="${AION_INTRADAY_OPEN_RANGE_MIN:-15}"
  export AION_INTRADAY_VOLUME_REL_MIN="${AION_INTRADAY_VOLUME_REL_MIN:-1.12}"
else
  export AION_TRADING_MODE="long_term"
fi

DEFAULT_IB_APP_PREFERRED="$HOME/Applications/IB Gateway 10.43/IB Gateway 10.43.app"
if [[ ! -d "$DEFAULT_IB_APP_PREFERRED" ]]; then
  DEFAULT_IB_APP_PREFERRED="/Applications/IB Gateway 10.43/IB Gateway 10.43.app"
fi
if [[ ! -d "$DEFAULT_IB_APP_PREFERRED" ]]; then
  DEFAULT_IB_APP_PREFERRED="/Applications/IB Gateway 10.39/IB Gateway 10.39.app"
fi
export AION_IB_APP_PREFERRED="${AION_IB_APP_PREFERRED:-$DEFAULT_IB_APP_PREFERRED}"
export AION_IB_APP_CANDIDATES="${AION_IB_APP_CANDIDATES:-$HOME/Applications/IB Gateway 10.43/IB Gateway 10.43.app,/Applications/IB Gateway 10.43/IB Gateway 10.43.app,/Applications/IB Gateway 10.39/IB Gateway 10.39.app}"

mkdir -p "$AION_STATE_DIR" "$AION_LOG_DIR"

if [[ -z "${AION_EXT_SIGNAL_FILE:-}" ]]; then
  if [[ -f "$AION_Q_HOME/runs_plus/q_signal_overlay.json" ]]; then
    export AION_EXT_SIGNAL_FILE="$AION_Q_HOME/runs_plus/q_signal_overlay.json"
  else
    export AION_EXT_SIGNAL_FILE="$AION_STATE_DIR/q_signal_overlay.json"
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "/Users/desmondpottle/aion-venv/bin/python" ]]; then
    PYTHON_BIN="/Users/desmondpottle/aion-venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

echo "[AION] Using Python: $PYTHON_BIN"
echo "[AION] Mode: $AION_MODE"
echo "[AION] Task: $AION_TASK"
echo "[AION] Trading mode: $AION_TRADING_MODE"
echo "[AION] IB endpoint: ${IB_HOST}:${IB_PORT}"
echo "[AION] State dir: $AION_STATE_DIR"
echo "[AION] Log dir: $AION_LOG_DIR"
echo "[AION] Q home: $AION_Q_HOME"
echo "[AION] External overlay: $AION_EXT_SIGNAL_FILE"
echo "[AION] Memory bridge: enable=$AION_MEMORY_ENABLE backend=$AION_MEMORY_BACKEND namespace=$AION_MEMORY_NAMESPACE"

if [[ "$AION_MODE" == "brain" ]]; then
  export AION_UNIVERSE_DIR="${AION_UNIVERSE_DIR:-$ROOT/universe}"
  echo "[AION] Universe dir: $AION_UNIVERSE_DIR"
  cd "$ROOT"

  case "$AION_TASK" in
    trade)
      FORCE_RESTART_TRADE="${AION_FORCE_RESTART_TRADE:-0}"
      ALLOW_OPS_GUARD_CONCURRENT="${AION_ALLOW_OPS_GUARD_CONCURRENT:-0}"
      EXISTING_TRADE_PIDS="$(
        ps -axo pid=,command= | awk '
          /[[:space:]]-m[[:space:]]aion\.exec\.paper_loop([[:space:]]|$)/ {print $1}
        ' || true
      )"
      OPS_GUARD_PIDS="$(
        ps -axo pid=,command= | awk '
          /[[:space:]]-m[[:space:]]aion\.exec\.ops_guard([[:space:]]|$)/ {print $1}
        ' || true
      )"

      if [[ -n "${OPS_GUARD_PIDS//[[:space:]]/}" && "$ALLOW_OPS_GUARD_CONCURRENT" != "1" ]]; then
        echo "[AION] ERROR: ops_guard is running (${OPS_GUARD_PIDS//$'\n'/ })."
        echo "[AION] Refusing direct trade launch to avoid competing supervisors."
        echo "[AION] Stop ops_guard first or set AION_ALLOW_OPS_GUARD_CONCURRENT=1."
        exit 2
      fi

      if [[ -n "${EXISTING_TRADE_PIDS//[[:space:]]/}" ]]; then
        if [[ "$FORCE_RESTART_TRADE" != "1" ]]; then
          echo "[AION] Trade loop already running (${EXISTING_TRADE_PIDS//$'\n'/ })."
          echo "[AION] Refusing duplicate launch. Set AION_FORCE_RESTART_TRADE=1 to recycle."
          exit 0
        fi
        echo "[AION] AION_FORCE_RESTART_TRADE=1; recycling trade loop: ${EXISTING_TRADE_PIDS//$'\n'/ }"
        while IFS= read -r pid; do
          [[ -z "${pid//[[:space:]]/}" ]] && continue
          kill "$pid" 2>/dev/null || true
        done <<< "$EXISTING_TRADE_PIDS"
        sleep 1
      fi

      if [[ "$AION_AUTO_CLEAN_STALE_WORKERS" == "1" ]]; then
        STALE_PIDS="$(
          ps -axo pid=,command= | awk '
            /[[:space:]]-m[[:space:]]aion\.exec\.(universe_scan|doctor|ib_recover|ib_wait_ready)([[:space:]]|$)/ {print $1}
          ' || true
        )"
        if [[ -n "${STALE_PIDS//[[:space:]]/}" ]]; then
          echo "[AION] Clearing stale AION workers: $(echo "$STALE_PIDS" | tr '\n' ' ')"
          while IFS= read -r pid; do
            [[ -z "${pid//[[:space:]]/}" ]] && continue
            kill "$pid" 2>/dev/null || true
          done <<< "$STALE_PIDS"
          sleep 1
        fi
      fi
      WATCHLIST_FILE="$AION_STATE_DIR/watchlist.txt"
      if [[ "$AION_AUTO_REFRESH_Q_OVERLAY" == "1" ]]; then
        if [[ -d "$AION_Q_HOME" && -f "$AION_Q_HOME/tools/export_aion_signal_pack.py" ]]; then
          echo "[AION] Refreshing Q overlay pack..."
          if ! (
            cd "$AION_Q_HOME" &&
            PYTHONPATH="$AION_Q_HOME" "$PYTHON_BIN" tools/export_aion_signal_pack.py \
              --out-json "$AION_EXT_SIGNAL_FILE" \
              --out-csv "$AION_Q_HOME/runs_plus/q_signal_overlay.csv" \
              --allow-degraded
          ); then
            echo "[AION] WARN: Q overlay refresh failed; continuing with existing overlay file."
          fi
        else
          echo "[AION] WARN: Q project not found at $AION_Q_HOME; skipping overlay refresh."
        fi
      fi
      if [[ "$AION_AUTO_IB_WARMUP" == "1" ]]; then
        echo "[AION] Waiting for IB API warmup..."
        if ! "$PYTHON_BIN" -m aion.exec.ib_wait_ready; then
          echo "[AION] WARN: IB warmup window elapsed; proceeding with existing startup policy."
        fi
      fi
      if [[ "$AION_AUTO_RESOLVE_IB_CONFLICT" == "1" ]]; then
        echo "[AION] Running IB conflict auto-recovery..."
        if ! "$PYTHON_BIN" -m aion.exec.ib_recover; then
          echo "[AION] WARN: IB conflict auto-recovery did not fully restore connectivity."
        fi
      fi
      if [[ "$AION_SKIP_DOCTOR" != "1" ]]; then
        echo "[AION] Running preflight doctor..."
        if ! "$PYTHON_BIN" -m aion.exec.doctor; then
          if [[ "$AION_DEGRADED_STARTUP" == "1" && -s "$WATCHLIST_FILE" ]]; then
            echo "[AION] WARN: doctor failed; continuing in degraded startup mode using existing watchlist."
          else
            echo "[AION] ERROR: doctor failed and degraded startup fallback unavailable."
            exit 1
          fi
        fi
      fi
      if [[ "$AION_SKIP_UNIVERSE_SCAN" == "1" && -s "$WATCHLIST_FILE" ]]; then
        echo "[AION] Using existing watchlist; skipping universe scan."
      else
        echo "[AION] Running brain universe scan..."
        if ! "$PYTHON_BIN" -m aion.exec.universe_scan; then
          if [[ "$AION_DEGRADED_STARTUP" == "1" && -s "$WATCHLIST_FILE" ]]; then
            echo "[AION] WARN: universe scan failed; reusing existing watchlist at $WATCHLIST_FILE."
          else
            echo "[AION] ERROR: universe scan failed and no watchlist fallback is available."
            exit 1
          fi
        fi
      fi
      if [[ "$AION_AUTO_TUNE_ON_START" == "1" ]]; then
        echo "[AION] Refreshing adaptive profile..."
        if ! "$PYTHON_BIN" -m aion.exec.adaptive_tuner; then
          echo "[AION] WARN: adaptive profile refresh failed; continuing with existing profile."
        fi
      fi
      if [[ "${AION_ENFORCE_READINESS_PRECHECK:-1}" == "1" ]]; then
        echo "[AION] Running paper-live readiness precheck..."
        if ! "$PYTHON_BIN" "$ROOT/tools/paper_live_readiness.py"; then
          echo "[AION] ERROR: paper-live readiness precheck failed."
          exit 1
        fi
      fi
      echo "[AION] Starting brain paper loop..."
      exec "$PYTHON_BIN" -m aion.exec.paper_loop
      ;;
    backtest)
      echo "[AION] Running walk-forward backtest..."
      exec "$PYTHON_BIN" -m aion.exec.backtest_walkforward
      ;;
    report)
      echo "[AION] Building performance report..."
      exec "$PYTHON_BIN" -m aion.exec.performance_report
      ;;
    tune)
      echo "[AION] Running adaptive tuner..."
      exec "$PYTHON_BIN" -m aion.exec.adaptive_tuner
      ;;
    gate)
      echo "[AION] Running live-promotion gate checks..."
      exec "$PYTHON_BIN" -m aion.exec.promotion_gate
      ;;
    recalibrate)
      echo "[AION] Running full recalibration pipeline..."
      exec "$PYTHON_BIN" -m aion.exec.recalibrate
      ;;
    recover-ib)
      echo "[AION] Running IB conflict recovery..."
      exec "$PYTHON_BIN" -m aion.exec.ib_recover
      ;;
    dashboard)
      echo "[AION] Starting local dashboard..."
      exec "$PYTHON_BIN" -m aion.exec.dashboard
      ;;
    ops-guard)
      echo "[AION] Starting ops guard..."
      exec "$PYTHON_BIN" -m aion.exec.ops_guard
      ;;
    operator)
      echo "[AION] Showing operator status..."
      exec "$PYTHON_BIN" -m aion.exec.operator
      ;;
    doctor)
      echo "[AION] Running environment doctor checks..."
      exec "$PYTHON_BIN" -m aion.exec.doctor
      ;;
    *)
      echo "[AION] ERROR: unknown AION_TASK '$AION_TASK' (trade|backtest|report|tune|gate|recalibrate|recover-ib|dashboard|ops-guard|operator|doctor)."
      exit 1
      ;;
  esac
fi

if [[ "$AION_MODE" == "legacy" ]]; then
  export AION_UNIVERSE_DIR="${AION_UNIVERSE_DIR:-$ROOT/tests/universe}"
  echo "[AION] Universe dir: $AION_UNIVERSE_DIR"
  if [[ "$AION_TASK" != "trade" ]]; then
    echo "[AION] Legacy mode supports only AION_TASK=trade."
    exit 1
  fi
  echo "[AION] Running legacy universe scan..."
  "$PYTHON_BIN" "$ROOT/tests/aion_universe_scan.py"
  echo "[AION] Starting legacy paper loop..."
  exec "$PYTHON_BIN" "$ROOT/tests/aion_paper_loop.py"
fi

echo "[AION] ERROR: unknown AION_MODE '$AION_MODE' (use 'brain' or 'legacy')."
exit 1
