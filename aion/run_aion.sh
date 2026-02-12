#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export AION_HOME="${AION_HOME:-$ROOT}"
export AION_STATE_DIR="${AION_STATE_DIR:-$ROOT/state}"
export AION_LOG_DIR="${AION_LOG_DIR:-$ROOT/logs}"
export IB_HOST="${IB_HOST:-127.0.0.1}"
export IB_PORT="${IB_PORT:-4002}"
export AION_MODE="${AION_MODE:-brain}"
export AION_TASK="${AION_TASK:-trade}"
export AION_SKIP_DOCTOR="${AION_SKIP_DOCTOR:-0}"
export AION_DEGRADED_STARTUP="${AION_DEGRADED_STARTUP:-1}"
export AION_AUTO_TUNE_ON_START="${AION_AUTO_TUNE_ON_START:-1}"
export AION_AUTO_CLEAN_STALE_WORKERS="${AION_AUTO_CLEAN_STALE_WORKERS:-1}"
export AION_AUTO_RESOLVE_IB_CONFLICT="${AION_AUTO_RESOLVE_IB_CONFLICT:-1}"
export AION_AUTO_RESTART_IB_ON_TIMEOUT="${AION_AUTO_RESTART_IB_ON_TIMEOUT:-1}"
export AION_AUTO_IB_WARMUP="${AION_AUTO_IB_WARMUP:-1}"

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
echo "[AION] IB endpoint: ${IB_HOST}:${IB_PORT}"
echo "[AION] State dir: $AION_STATE_DIR"
echo "[AION] Log dir: $AION_LOG_DIR"

if [[ "$AION_MODE" == "brain" ]]; then
  export AION_UNIVERSE_DIR="${AION_UNIVERSE_DIR:-$ROOT/universe}"
  echo "[AION] Universe dir: $AION_UNIVERSE_DIR"
  cd "$ROOT"

  case "$AION_TASK" in
    trade)
      if [[ "$AION_AUTO_CLEAN_STALE_WORKERS" == "1" ]]; then
        STALE_PIDS="$(
          ps -axo pid=,command= | awk '
            /[[:space:]]-m[[:space:]]aion\.exec\.(paper_loop|universe_scan|doctor|ib_recover|ib_wait_ready)([[:space:]]|$)/ {print $1}
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
      echo "[AION] Running brain universe scan..."
      if ! "$PYTHON_BIN" -m aion.exec.universe_scan; then
        if [[ "$AION_DEGRADED_STARTUP" == "1" && -s "$WATCHLIST_FILE" ]]; then
          echo "[AION] WARN: universe scan failed; reusing existing watchlist at $WATCHLIST_FILE."
        else
          echo "[AION] ERROR: universe scan failed and no watchlist fallback is available."
          exit 1
        fi
      fi
      if [[ "$AION_AUTO_TUNE_ON_START" == "1" ]]; then
        echo "[AION] Refreshing adaptive profile..."
        if ! "$PYTHON_BIN" -m aion.exec.adaptive_tuner; then
          echo "[AION] WARN: adaptive profile refresh failed; continuing with existing profile."
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
    doctor)
      echo "[AION] Running environment doctor checks..."
      exec "$PYTHON_BIN" -m aion.exec.doctor
      ;;
    *)
      echo "[AION] ERROR: unknown AION_TASK '$AION_TASK' (trade|backtest|report|tune|gate|recalibrate|recover-ib|dashboard|doctor)."
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
