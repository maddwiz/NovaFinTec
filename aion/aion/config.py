import os
from pathlib import Path


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float_list_env(name: str, default: list[float]) -> list[float]:
    raw = os.getenv(name)
    if not raw:
        return default
    out = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(float(token))
    return out or default


def _int_list_env(name: str, default: list[int]) -> list[int]:
    raw = os.getenv(name)
    if not raw:
        return default
    out = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out or default


def _str_list_env(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    out = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(token)
    return out or default


def _dedupe_ints(values: list[int]) -> list[int]:
    out = []
    seen = set()
    for v in values:
        try:
            iv = int(v)
        except Exception:
            continue
        if iv <= 0 or iv in seen:
            continue
        out.append(iv)
        seen.add(iv)
    return out


def _dedupe_strs(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for v in values:
        sv = str(v).strip()
        if not sv:
            continue
        key = sv.lower()
        if key in seen:
            continue
        out.append(sv)
        seen.add(key)
    return out


AION_HOME = Path(os.getenv("AION_HOME", Path(__file__).resolve().parents[1]))
Q_HOME = Path(os.getenv("AION_Q_HOME", AION_HOME.parent / "q"))
LOG_DIR = Path(os.getenv("AION_LOG_DIR", AION_HOME / "logs"))
STATE_DIR = Path(os.getenv("AION_STATE_DIR", AION_HOME / "state"))
UNIVERSE_DIR = Path(os.getenv("AION_UNIVERSE_DIR", AION_HOME / "universe"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

IB_HOST = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT = int(os.getenv("IB_PORT", "4002"))
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "1731"))
IB_MARKET_DATA_TYPE = int(os.getenv("AION_MARKET_DATA_TYPE", "3"))
IB_REQUEST_MIN_INTERVAL_SEC = float(os.getenv("AION_IB_REQUEST_MIN_INTERVAL_SEC", "0.11"))
IB_RECONNECT_LOG_EVERY = int(os.getenv("AION_IB_RECONNECT_LOG_EVERY", "3"))
IB_HOST_CANDIDATES = _dedupe_strs(_str_list_env("AION_IB_HOST_CANDIDATES", [IB_HOST, "127.0.0.1", "localhost", "::1"]))
IB_PORT_CANDIDATES = _dedupe_ints(_int_list_env("AION_IB_PORT_CANDIDATES", [IB_PORT, 4002, 7497, 7496]))
_DEFAULT_IB_APP_CANDIDATES = [
    str(Path.home() / "Applications/IB Gateway 10.43/IB Gateway 10.43.app"),
    "/Applications/IB Gateway 10.43/IB Gateway 10.43.app",
    "/Applications/IB Gateway 10.39/IB Gateway 10.39.app",
]
IB_APP_CANDIDATES = _dedupe_strs(_str_list_env("AION_IB_APP_CANDIDATES", _DEFAULT_IB_APP_CANDIDATES))
IB_APP_PREFERRED = str(os.getenv("AION_IB_APP_PREFERRED", "")).strip() or (IB_APP_CANDIDATES[0] if IB_APP_CANDIDATES else "")
AUTO_RESOLVE_IB_CONFLICT = _bool_env("AION_AUTO_RESOLVE_IB_CONFLICT", False)
AUTO_RESTART_IB_ON_TIMEOUT = _bool_env("AION_AUTO_RESTART_IB_ON_TIMEOUT", False)
AUTO_IB_WARMUP = _bool_env("AION_AUTO_IB_WARMUP", True)
IB_WARMUP_SECONDS = int(os.getenv("AION_IB_WARMUP_SECONDS", "75"))
IB_WARMUP_POLL_SECONDS = int(os.getenv("AION_IB_WARMUP_POLL_SECONDS", "5"))

SHORTLIST_CAP = int(os.getenv("AION_SHORTLIST_CAP", "50"))
HIST_BAR_SIZE = os.getenv("AION_HIST_BAR_SIZE", "5 mins")
HIST_DURATION = os.getenv("AION_HIST_DURATION", "2 D")
HIST_USE_RTH = _bool_env("AION_HIST_USE_RTH", True)
MAIN_BARS_CACHE_SEC = int(os.getenv("AION_MAIN_BARS_CACHE_SEC", "25"))
MTF_1H_CACHE_SEC = int(os.getenv("AION_MTF_1H_CACHE_SEC", "300"))
MTF_4H_CACHE_SEC = int(os.getenv("AION_MTF_4H_CACHE_SEC", "900"))

# Multi-timeframe confirmation
MTF_CONFIRM_ENABLED = _bool_env("AION_MTF_CONFIRM_ENABLED", True)
MTF_1H_BAR = os.getenv("AION_MTF_1H_BAR", "1 hour")
MTF_1H_DURATION = os.getenv("AION_MTF_1H_DURATION", "20 D")
MTF_4H_BAR = os.getenv("AION_MTF_4H_BAR", "4 hours")
MTF_4H_DURATION = os.getenv("AION_MTF_4H_DURATION", "90 D")
MTF_MIN_ALIGNMENT_SCORE = float(os.getenv("AION_MTF_MIN_ALIGNMENT_SCORE", "0.58"))

EQUITY_START = float(os.getenv("AION_EQUITY_START", "5000"))
RISK_PER_TRADE = float(os.getenv("AION_RISK_PER_TRADE", "0.02"))
MAX_TRADES_PER_DAY = int(os.getenv("AION_MAX_TRADES_PER_DAY", "10"))
MAX_OPEN_POSITIONS = int(os.getenv("AION_MAX_OPEN_POSITIONS", "5"))
MAX_POSITION_NOTIONAL_PCT = float(os.getenv("AION_MAX_POSITION_NOTIONAL_PCT", "0.22"))
MAX_GROSS_LEVERAGE = float(os.getenv("AION_MAX_GROSS_LEVERAGE", "1.6"))
DAILY_DRAWDOWN_LIMIT = float(os.getenv("AION_DAILY_DRAWDOWN_LIMIT", "0.10"))
TOTAL_DRAWDOWN_LIMIT = float(os.getenv("AION_TOTAL_DRAWDOWN_LIMIT", "0.22"))
MAX_CONSECUTIVE_LOSSES = int(os.getenv("AION_MAX_CONSECUTIVE_LOSSES", "5"))
AUTO_RESET_KILLSWITCH_ON_START = _bool_env("AION_AUTO_RESET_KILLSWITCH_ON_START", True)

# Portfolio optimizer
PORTFOLIO_ENABLE = _bool_env("AION_PORTFOLIO_ENABLE", True)
PORTFOLIO_CORR_PENALTY = float(os.getenv("AION_PORTFOLIO_CORR_PENALTY", "0.45"))
PORTFOLIO_VOL_TARGET = float(os.getenv("AION_PORTFOLIO_VOL_TARGET", "0.015"))
PORTFOLIO_MIN_WEIGHT = float(os.getenv("AION_PORTFOLIO_MIN_WEIGHT", "0.04"))
PORTFOLIO_MAX_WEIGHT = float(os.getenv("AION_PORTFOLIO_MAX_WEIGHT", "0.30"))
PORTFOLIO_MAX_CANDIDATES = int(os.getenv("AION_PORTFOLIO_MAX_CANDIDATES", "12"))

# Execution realism
SLIPPAGE_BPS = int(os.getenv("AION_SLIPPAGE_BPS", "5"))
SPREAD_BPS_BASE = float(os.getenv("AION_SPREAD_BPS_BASE", "2.5"))
SPREAD_BPS_VOL_MULT = float(os.getenv("AION_SPREAD_BPS_VOL_MULT", "18.0"))
EXEC_LATENCY_MS = int(os.getenv("AION_EXEC_LATENCY_MS", "250"))
EXEC_QUEUE_IMPACT_BPS = float(os.getenv("AION_EXEC_QUEUE_IMPACT_BPS", "3.0"))
EXEC_PARTIAL_FILL_MIN = float(os.getenv("AION_EXEC_PARTIAL_FILL_MIN", "0.35"))
EXEC_PARTIAL_FILL_MAX = float(os.getenv("AION_EXEC_PARTIAL_FILL_MAX", "1.00"))

RSI_LEN = int(os.getenv("AION_RSI_LEN", "14"))
EMA_FAST = int(os.getenv("AION_EMA_FAST", "12"))
EMA_SLOW = int(os.getenv("AION_EMA_SLOW", "26"))
ATR_LEN = int(os.getenv("AION_ATR_LEN", "14"))
ADX_LEN = int(os.getenv("AION_ADX_LEN", "14"))
BB_LEN = int(os.getenv("AION_BB_LEN", "20"))
BB_STD = float(os.getenv("AION_BB_STD", "2.0"))
STOCH_LEN = int(os.getenv("AION_STOCH_LEN", "14"))
STOCH_SMOOTH = int(os.getenv("AION_STOCH_SMOOTH", "3"))
FIB_TOLERANCE = float(os.getenv("AION_FIB_TOLERANCE", "0.003"))

ENTRY_THRESHOLD_LONG = float(os.getenv("AION_ENTRY_THRESHOLD_LONG", "0.60"))
ENTRY_THRESHOLD_SHORT = float(os.getenv("AION_ENTRY_THRESHOLD_SHORT", "0.60"))
OPPOSITE_EXIT_THRESHOLD = float(os.getenv("AION_OPPOSITE_EXIT_THRESHOLD", "0.62"))
REENTRY_COOLDOWN_CYCLES = int(os.getenv("AION_REENTRY_COOLDOWN_CYCLES", "2"))

CONFLUENCE_LONG_MIN = float(os.getenv("AION_CONFLUENCE_LONG_MIN", "0.46"))
CONFLUENCE_SHORT_MIN = float(os.getenv("AION_CONFLUENCE_SHORT_MIN", "0.46"))
CONFLUENCE_BOOST_MAX = float(os.getenv("AION_CONFLUENCE_BOOST_MAX", "0.12"))
CONFLUENCE_PENALTY_MAX = float(os.getenv("AION_CONFLUENCE_PENALTY_MAX", "0.15"))
SIGNAL_MIN_MARGIN = float(os.getenv("AION_SIGNAL_MIN_MARGIN", "0.05"))

REGIME_TH_SHIFT_TRENDING = float(os.getenv("AION_REGIME_TH_SHIFT_TRENDING", "-0.03"))
REGIME_TH_SHIFT_SQUEEZE = float(os.getenv("AION_REGIME_TH_SHIFT_SQUEEZE", "0.02"))
REGIME_TH_SHIFT_CALM_RANGE = float(os.getenv("AION_REGIME_TH_SHIFT_CALM_RANGE", "0.00"))
REGIME_TH_SHIFT_HIGH_VOL_CHOP = float(os.getenv("AION_REGIME_TH_SHIFT_HIGH_VOL_CHOP", "0.06"))
REGIME_OPP_EXIT_SHIFT_TRENDING = float(os.getenv("AION_REGIME_OPP_EXIT_SHIFT_TRENDING", "-0.02"))
REGIME_OPP_EXIT_SHIFT_SQUEEZE = float(os.getenv("AION_REGIME_OPP_EXIT_SHIFT_SQUEEZE", "0.01"))
REGIME_OPP_EXIT_SHIFT_CALM_RANGE = float(os.getenv("AION_REGIME_OPP_EXIT_SHIFT_CALM_RANGE", "0.00"))
REGIME_OPP_EXIT_SHIFT_HIGH_VOL_CHOP = float(os.getenv("AION_REGIME_OPP_EXIT_SHIFT_HIGH_VOL_CHOP", "0.02"))
REGIME_MARGIN_SHIFT_TRENDING = float(os.getenv("AION_REGIME_MARGIN_SHIFT_TRENDING", "-0.01"))
REGIME_MARGIN_SHIFT_SQUEEZE = float(os.getenv("AION_REGIME_MARGIN_SHIFT_SQUEEZE", "0.01"))
REGIME_MARGIN_SHIFT_CALM_RANGE = float(os.getenv("AION_REGIME_MARGIN_SHIFT_CALM_RANGE", "0.00"))
REGIME_MARGIN_SHIFT_HIGH_VOL_CHOP = float(os.getenv("AION_REGIME_MARGIN_SHIFT_HIGH_VOL_CHOP", "0.03"))

STOP_ATR_MULT = float(os.getenv("AION_STOP_ATR_MULT", "1.35"))
TARGET_ATR_MULT = float(os.getenv("AION_TARGET_ATR_MULT", "2.8"))
TRAIL_ATR_MULT = float(os.getenv("AION_TRAIL_ATR_MULT", "1.1"))
BREAKEVEN_R = float(os.getenv("AION_BREAKEVEN_R", "1.0"))
PARTIAL_TAKE_R = float(os.getenv("AION_PARTIAL_TAKE_R", "1.5"))
PARTIAL_CLOSE_FRACTION = float(os.getenv("AION_PARTIAL_CLOSE_FRACTION", "0.40"))
MAX_HOLD_CYCLES = int(os.getenv("AION_MAX_HOLD_CYCLES", "18"))

REGIME_ADX_TREND_MIN = float(os.getenv("AION_REGIME_ADX_TREND_MIN", "20.0"))
REGIME_ATR_PCT_HIGH = float(os.getenv("AION_REGIME_ATR_PCT_HIGH", "0.045"))
REGIME_ATR_PCT_LOW = float(os.getenv("AION_REGIME_ATR_PCT_LOW", "0.008"))
REGIME_BB_SQUEEZE_PCT = float(os.getenv("AION_REGIME_BB_SQUEEZE_PCT", "0.06"))

# Event/news risk filter
EVENT_FILTER_ENABLED = _bool_env("AION_EVENT_FILTER_ENABLED", True)
EVENT_BLOCK_OPEN_MIN = int(os.getenv("AION_EVENT_BLOCK_OPEN_MIN", "10"))
EVENT_BLOCK_CLOSE_MIN = int(os.getenv("AION_EVENT_BLOCK_CLOSE_MIN", "10"))
EVENT_BLOCK_FILE = Path(os.getenv("AION_EVENT_BLOCK_FILE", str(STATE_DIR / "event_blackouts.json")))

# Meta label model
META_LABEL_ENABLED = _bool_env("AION_META_LABEL_ENABLED", True)
META_LABEL_MIN_PROB = float(os.getenv("AION_META_LABEL_MIN_PROB", "0.52"))
META_LABEL_TRAIN_MIN_SAMPLES = int(os.getenv("AION_META_LABEL_TRAIN_MIN_SAMPLES", "40"))
META_LABEL_LR = float(os.getenv("AION_META_LABEL_LR", "0.05"))
META_LABEL_EPOCHS = int(os.getenv("AION_META_LABEL_EPOCHS", "240"))

LOOP_SECONDS = int(os.getenv("AION_LOOP_SECONDS", "30"))
SWING_LOOKBACK = int(os.getenv("AION_SWING_LOOKBACK", "60"))
MIN_BARS = int(os.getenv("AION_MIN_BARS", "80"))
RESTORE_RUNTIME_STATE = _bool_env("AION_RESTORE_RUNTIME_STATE", True)
RUNTIME_STATE_FILE = Path(os.getenv("AION_RUNTIME_STATE_FILE", str(STATE_DIR / "runtime_state.json")))

# Optional external signal overlay (e.g., Q -> AION bridge)
EXT_SIGNAL_ENABLED = _bool_env("AION_EXT_SIGNAL_ENABLED", True)
_DEFAULT_EXT_SIGNAL_FILE = Q_HOME / "runs_plus" / "q_signal_overlay.json"
if not _DEFAULT_EXT_SIGNAL_FILE.exists():
    _DEFAULT_EXT_SIGNAL_FILE = STATE_DIR / "q_signal_overlay.json"
EXT_SIGNAL_FILE = Path(os.getenv("AION_EXT_SIGNAL_FILE", str(_DEFAULT_EXT_SIGNAL_FILE)))
EXT_SIGNAL_MIN_CONFIDENCE = float(os.getenv("AION_EXT_SIGNAL_MIN_CONFIDENCE", "0.55"))
EXT_SIGNAL_MAX_BIAS = float(os.getenv("AION_EXT_SIGNAL_MAX_BIAS", "0.90"))
EXT_SIGNAL_CONF_BOOST = float(os.getenv("AION_EXT_SIGNAL_CONF_BOOST", "0.10"))
EXT_SIGNAL_THRESHOLD_SHIFT = float(os.getenv("AION_EXT_SIGNAL_THRESHOLD_SHIFT", "0.05"))
EXT_SIGNAL_RUNTIME_MIN_SCALE = float(os.getenv("AION_EXT_RUNTIME_MIN_SCALE", "0.55"))
EXT_SIGNAL_RUNTIME_MAX_SCALE = float(os.getenv("AION_EXT_RUNTIME_MAX_SCALE", "1.05"))
EXT_SIGNAL_RUNTIME_DEGRADED_SCALE = float(os.getenv("AION_EXT_RUNTIME_DEGRADED_SCALE", "0.70"))
EXT_SIGNAL_RUNTIME_QFAIL_SCALE = float(os.getenv("AION_EXT_RUNTIME_QFAIL_SCALE", "0.82"))
EXT_SIGNAL_RUNTIME_FLAG_SCALE = float(os.getenv("AION_EXT_RUNTIME_FLAG_SCALE", "0.90"))
EXT_SIGNAL_MAX_AGE_HOURS = float(os.getenv("AION_EXT_SIGNAL_MAX_AGE_HOURS", "12"))
EXT_SIGNAL_REQUIRE_RUNTIME_CONTEXT = _bool_env("AION_EXT_SIGNAL_REQUIRE_RUNTIME_CONTEXT", False)
EXT_SIGNAL_CRITICAL = _bool_env("AION_EXT_SIGNAL_CRITICAL", False)

WALKFORWARD_DURATION = os.getenv("AION_WF_DURATION", "30 D")
WALKFORWARD_BAR_SIZE = os.getenv("AION_WF_BAR_SIZE", "15 mins")
WALKFORWARD_TRAIN_BARS = int(os.getenv("AION_WF_TRAIN_BARS", "240"))
WALKFORWARD_TEST_BARS = int(os.getenv("AION_WF_TEST_BARS", "80"))
WALKFORWARD_STEP_BARS = int(os.getenv("AION_WF_STEP_BARS", "80"))
WF_THRESHOLDS = _float_list_env("AION_WF_THRESHOLDS", [0.56, 0.60, 0.64])
WF_STOP_MULTS = _float_list_env("AION_WF_STOP_MULTS", [1.2, 1.35, 1.6])
WF_TARGET_MULTS = _float_list_env("AION_WF_TARGET_MULTS", [2.2, 2.8, 3.4])
WF_MAX_SYMBOLS = int(os.getenv("AION_WF_MAX_SYMBOLS", "20"))
WF_STABILITY_MIN_SCORE = float(os.getenv("AION_WF_STABILITY_MIN_SCORE", "0.58"))

ADAPTIVE_LOOKBACK_TRADES = int(os.getenv("AION_ADAPTIVE_LOOKBACK_TRADES", "80"))
ADAPTIVE_WINRATE_FLOOR = float(os.getenv("AION_ADAPTIVE_WINRATE_FLOOR", "0.45"))
ADAPTIVE_EXPECTANCY_FLOOR = float(os.getenv("AION_ADAPTIVE_EXPECTANCY_FLOOR", "0.0"))
ADAPTIVE_THRESHOLD_STEP = float(os.getenv("AION_ADAPTIVE_THRESHOLD_STEP", "0.02"))
ADAPTIVE_THRESHOLD_MAX = float(os.getenv("AION_ADAPTIVE_THRESHOLD_MAX", "0.75"))
ADAPTIVE_THRESHOLD_MIN = float(os.getenv("AION_ADAPTIVE_THRESHOLD_MIN", "0.50"))
REASON_ADAPT_ENABLED = _bool_env("AION_REASON_ADAPT_ENABLED", True)
REASON_ADAPT_MIN_TRADES = int(os.getenv("AION_REASON_ADAPT_MIN_TRADES", "10"))
REASON_ADAPT_MAX_REASONS = int(os.getenv("AION_REASON_ADAPT_MAX_REASONS", "16"))
REASON_ADAPT_MIN_MULT = float(os.getenv("AION_REASON_ADAPT_MIN_MULT", "0.82"))
REASON_ADAPT_MAX_MULT = float(os.getenv("AION_REASON_ADAPT_MAX_MULT", "1.18"))

# Promotion gate for live deployment readiness
PROMOTION_MIN_TRADES = int(os.getenv("AION_PROMOTION_MIN_TRADES", "60"))
PROMOTION_MIN_WINRATE = float(os.getenv("AION_PROMOTION_MIN_WINRATE", "0.52"))
PROMOTION_MIN_PROFIT_FACTOR = float(os.getenv("AION_PROMOTION_MIN_PROFIT_FACTOR", "1.25"))
PROMOTION_MAX_DRAWDOWN = float(os.getenv("AION_PROMOTION_MAX_DRAWDOWN", "0.12"))
PROMOTION_MIN_WF_AVG_PNL = float(os.getenv("AION_PROMOTION_MIN_WF_AVG_PNL", "0.0"))

# Monitoring thresholds
MONITORING_ENABLED = _bool_env("AION_MONITORING_ENABLED", True)
MONITOR_SIGNAL_DRIFT_WINDOW = int(os.getenv("AION_MONITOR_SIGNAL_DRIFT_WINDOW", "120"))
MONITOR_MIN_AVG_CONF = float(os.getenv("AION_MONITOR_MIN_AVG_CONF", "0.54"))
MONITOR_MAX_SLIPPAGE_BPS = float(os.getenv("AION_MONITOR_MAX_SLIPPAGE_BPS", "28.0"))
MONITOR_MAX_HOURLY_DD = float(os.getenv("AION_MONITOR_MAX_HOURLY_DD", "0.035"))
MONITOR_EVENT_WINDOW_MIN = int(os.getenv("AION_MONITOR_EVENT_WINDOW_MIN", "20"))
MONITOR_IB_FAIL_ALERT_COUNT = int(os.getenv("AION_MONITOR_IB_FAIL_ALERT_COUNT", "3"))

# Local web dashboard
DASHBOARD_HOST = os.getenv("AION_DASHBOARD_HOST", "127.0.0.1")
DASHBOARD_PORT = int(os.getenv("AION_DASHBOARD_PORT", "8787"))

# Recalibration schedule metadata/config
RECALIBRATION_MIN_DAYS_BETWEEN = int(os.getenv("AION_RECALIBRATION_MIN_DAYS_BETWEEN", "5"))
