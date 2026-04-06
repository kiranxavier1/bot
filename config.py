"""
config.py
─────────────────────────────────────────────────────────────────────────────
Central configuration — loads .env, validates required keys, exposes typed
constants used by every module in the bot.

v2: All 11 strategy fixes incorporated.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)


def _require(key: str) -> str:
    val = os.getenv(key, "").strip()
    if not val:
        sys.exit(
            f"[CONFIG ERROR] Required environment variable '{key}' is missing.\n"
            f"Copy .env.example → .env and fill in the value."
        )
    return val


def _optional(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


# ── Exchange ──────────────────────────────────────────────────────────────────
BINANCE_API_KEY    = _require("BINANCE_API_KEY")
BINANCE_API_SECRET = _require("BINANCE_API_SECRET")

EXCHANGE_ID        = "binance"
QUOTE_CURRENCY     = "USDT"

# FIX #9 — Primary signal timeframe is 15m; 5m used for precise entry timing.
# Both are streamed via WebSocket so the buffer is always warm.
TIMEFRAMES         = ["5m", "15m"]
SIGNAL_TIMEFRAME   = "15m"     # trendline pattern detection
ENTRY_TIMEFRAME    = "5m"      # precise entry confirmation candle

CANDLE_BUFFER_SIZE = 300       # bumped from 200 → 300 to support 1h EMA-200

# Symbol discovery
TOP_N_PAIRS             = int(_optional("TOP_N_PAIRS", "50"))
DISCOVERY_INTERVAL_SECS = 3600

# ── AI / Claude ───────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = _require("ANTHROPIC_API_KEY")
CLAUDE_MODEL      = _optional("CLAUDE_MODEL", "claude-sonnet-4-6")
MIN_AI_CONFIDENCE = float(_optional("MIN_AI_CONFIDENCE", "0.8"))

# ── Telegram ──────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = _require("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = _require("TELEGRAM_CHAT_ID")

# ── CryptoPanic (optional) ────────────────────────────────────────────────────
CRYPTOPANIC_API_KEY = _optional("CRYPTOPANIC_API_KEY")

# ── Pattern detection (FIX #1, #5, #10) ──────────────────────────────────────
PIVOT_N            = int(_optional("PIVOT_N", "5"))
WATCHER_PROXIMITY  = float(_optional("WATCHER_PROXIMITY", "0.005"))   # 0.5 %

# FIX #10 — require N confirmed touches before the trendline is considered valid
MIN_TRENDLINE_TOUCHES = 3

# FIX #5 — trendline slope bounds (price-change per 15m candle as a fraction)
# Rejects near-flat noise AND unsustainably steep lines.
# Default: slope must be between +0.0002 and +0.005 per candle
TRENDLINE_MIN_SLOPE = float(_optional("TRENDLINE_MIN_SLOPE", "0.0002"))
TRENDLINE_MAX_SLOPE = float(_optional("TRENDLINE_MAX_SLOPE", "0.005"))

# FIX #1 — volume confirmation: confirmation candle must exceed N× avg volume
VOLUME_CONFIRM_MULTIPLIER = float(_optional("VOLUME_CONFIRM_MULTIPLIER", "1.3"))
VOLUME_LOOKBACK           = 20    # periods for average volume baseline

# ── Risk management ───────────────────────────────────────────────────────────
RISK_PER_TRADE = float(_optional("RISK_PER_TRADE", "0.02"))  # 2 % of balance

# FIX #4 — ATR-based stop loss (replaces fixed % / candle-low)
ATR_PERIOD     = 14
ATR_MULTIPLIER = float(_optional("ATR_MULTIPLIER", "1.5"))   # SL = entry − 1.5×ATR

# FIX #11 + USER SWING INSIGHT — target just below previous swing high
# If no valid swing high found, fall back to min RR ratio below
SWING_TP_BUFFER   = 0.005    # sell 0.5 % below the previous swing high
MIN_RR_FALLBACK   = 1.5      # minimum R:R if no structural TP available

# FIX #8 — structure-based break-even: trail SL below the most recent pivot low
# formed AFTER entry, rather than a fixed +5 % price level
BE_PIVOT_LOOKBACK = 10       # scan last N closed candles for post-entry pivot

# Cooldown after SL hit
COOLDOWN_SECONDS = int(_optional("COOLDOWN_SECONDS", str(4 * 3600)))

# FIX #6 — portfolio-level daily loss circuit breaker
MAX_DAILY_LOSS_PCT = float(_optional("MAX_DAILY_LOSS_PCT", "0.03"))  # halt at -3 %

# ── Higher-timeframe trend filter (FIX #2) ────────────────────────────────────
HTF_EMA_PERIOD    = 200      # coin must be above 200-EMA on the HTF
HTF_FILTER_TF     = "1h"     # higher timeframe used for the EMA filter
HTF_CANDLE_LIMIT  = 220      # how many 1h candles to fetch for the EMA

# ── ADX directional filter (FIX #3) ──────────────────────────────────────────
ADX_TREND_THRESHOLD = 25     # ADX must be above this
# DI+ must be > DI− to confirm upward trend direction
ADX_REQUIRE_DIRECTION = True

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = _optional("LOG_LEVEL", "INFO").upper()
LOG_FILE  = _optional("LOG_FILE", "bot.log")

# ── Macro context ─────────────────────────────────────────────────────────────
BTC_SYMBOL    = "BTC/USDT"
BTC_TIMEFRAME = "1h"

# ── News keywords ─────────────────────────────────────────────────────────────
NEGATIVE_KEYWORDS = [
    "hack", "exploit", "scam", "fraud", "rug", "lawsuit",
    "sec", "ban", "delist", "bankrupt", "insolvent", "crash",
    "ponzi", "phishing", "stolen", "breach",
]
