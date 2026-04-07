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

# Primary signal timeframe is 5m (user requested); 15m and 1h used for HTF trend.
TIMEFRAMES         = ["5m", "15m", "1h"]
SIGNAL_TIMEFRAME   = "5m"      # core trading timeframe
ENTRY_TIMEFRAME    = "5m"      # precise entry confirmation candle
HTF_FILTER_TF      = "15m"     # visual trend identification (1h also used)
HTF_SECONDARY_TF   = "1h"      # macro trend

CANDLE_BUFFER_SIZE = 300       # bumped from 200 → 300 to support 1h EMA-200

# Symbol discovery
TOP_N_PAIRS             = int(_optional("TOP_N_PAIRS", "25"))
DISCOVERY_INTERVAL_SECS = 3600

# ── AI / Gemini ───────────────────────────────────────────────────────────────
GEMINI_API_KEY = _require("GEMINI_API_KEY")
GEMINI_MODEL   = _optional("GEMINI_MODEL", "gemini-3-flash-preview")
# Lowered from 0.80 → 0.60: capture more active scalp opportunities dynamically.
# High-volatility coins need more entries, not fewer. The quality checklist and
# continuous learning rules still protect against bad setups.
MIN_AI_CONFIDENCE = float(_optional("MIN_AI_CONFIDENCE", "0.60"))

# ── Telegram ──────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = _optional("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = _optional("TELEGRAM_CHAT_ID", "")

# ── CryptoPanic (optional) ────────────────────────────────────────────────────
CRYPTOPANIC_API_KEY = _optional("CRYPTOPANIC_API_KEY")

# ── Pattern detection (FIX #1, #5, #10) ──────────────────────────────────────
PIVOT_N            = int(_optional("PIVOT_N", "5"))
WATCHER_PROXIMITY  = float(_optional("WATCHER_PROXIMITY", "0.005"))   # 0.5 %

# Required validated touches before trading. 
# 2 touches define the line -> enter trade on the 3rd touch!
MIN_TRENDLINE_TOUCHES = 2

# FIX #5 — trendline slope bounds (price-change per 15m candle as a fraction)
# Rejects near-flat noise AND unsustainably steep lines.
# Default: slope must be between +0.0002 and +0.005 per candle
TRENDLINE_MIN_SLOPE = float(_optional("TRENDLINE_MIN_SLOPE", "0.0002"))
TRENDLINE_MAX_SLOPE = float(_optional("TRENDLINE_MAX_SLOPE", "0.005"))

# FIX #1 — volume confirmation: confirmation candle must exceed N× avg volume
VOLUME_CONFIRM_MULTIPLIER = float(_optional("VOLUME_CONFIRM_MULTIPLIER", "1.3"))
VOLUME_LOOKBACK           = 20    # periods for average volume baseline

# ── Risk management ───────────────────────────────────────────────────────────
RISK_PER_TRADE = float(_optional("RISK_PER_TRADE", "0.02"))  # 2 % of balance (legacy — kept for fallback)

# Percentage-based position sizing — what % of the free balance to deploy per trade.
# Overridden at runtime by the dashboard UI slider (bot_state.trade_allocation_pct).
# Default 30 % means a $12 balance → ~$3.60 margin per trade at 1× leverage.
TRADE_ALLOCATION_PCT = float(_optional("TRADE_ALLOCATION_PCT", "30.0"))  # 30 % of balance

# Futures settings
USE_FUTURES       = True      # Enabled for AWS Static IP
MAX_LEVERAGE      = 20        # Cap AI-selected leverage
DEFAULT_LEVERAGE  = 15


# Minimum free balance (USDT) before the bot will open a new trade.
# Set low enough to work with small balances like $12.
MIN_TRADE_BALANCE = float(_optional("MIN_TRADE_BALANCE", "2.0"))  # $2 minimum

# FIX #4 — ATR-based stop loss (replaces fixed % / candle-low)
ATR_PERIOD     = 14
ATR_MULTIPLIER = float(_optional("ATR_MULTIPLIER", "1.5"))   # SL = entry − 1.5×ATR

# FIX #11 + USER SWING INSIGHT — target just below previous swing high
# If no valid swing high found, fall back to min RR ratio below (User requested 3-4x profit risk)
SWING_TP_BUFFER   = 0.005    # sell 0.5 % below the previous swing high
MIN_RR_FALLBACK   = 3.0      # minimum R:R if no structural TP available

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

# ── VWAP Bounce strategy ──────────────────────────────────────────────────────
# Rolling session length: 96 × 15m = 24 h — mirrors how institutions reset VWAP daily
VWAP_SESSION_CANDLES    = int(_optional("VWAP_SESSION_CANDLES", "96"))
# Arm the VWAP watcher when price is within 0.3% of VWAP
VWAP_PROXIMITY          = float(_optional("VWAP_PROXIMITY", "0.003"))
# Volume confirmation for VWAP bounce (slightly lower than trendline bounce)
VWAP_VOLUME_MULTIPLIER  = float(_optional("VWAP_VOLUME_MULTIPLIER", "1.2"))

# ── RSI Divergence strategy ───────────────────────────────────────────────────
# Number of recent candles to scan for divergence patterns
RSI_DIVERGENCE_LOOKBACK = int(_optional("RSI_DIVERGENCE_LOOKBACK", "20"))
# RSI at the second (lower) price low must be below this level (mildly oversold)
RSI_DIVERGENCE_OVERSOLD = float(_optional("RSI_DIVERGENCE_OVERSOLD", "45"))

# ── Coin selection — volatility filter ────────────────────────────────────────
# Minimum 24h high-low range as % of price; filters stablecoins & low-vol coins
# 3% daily range = coin can actually produce scalping / swing profits
MIN_DAILY_RANGE_PCT = float(_optional("MIN_DAILY_RANGE_PCT", "3.0"))

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
