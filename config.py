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

# Scalping + swing trading only — 5m signal, 15m HTF context. No 1h needed.
TIMEFRAMES         = ["5m", "15m"]
SIGNAL_TIMEFRAME   = "5m"      # core trading timeframe (all entries)
ENTRY_TIMEFRAME    = "5m"      # precise entry confirmation candle
HTF_FILTER_TF      = "15m"     # higher timeframe for trend context

CANDLE_BUFFER_SIZE = 200       # 200 candles sufficient for 15m EMA-50

# Symbol discovery
TOP_N_PAIRS             = int(_optional("TOP_N_PAIRS", "40"))
DISCOVERY_INTERVAL_SECS = 3600

# ── AI / Gemini ────────────────────────────────────────────────────────────
GEMINI_API_KEY = _require("GEMINI_API_KEY")
GEMINI_MODEL   = _optional("GEMINI_MODEL", "gemini-3.1-pro-preview")
# 60%+ win rate mode: AI filters for quality, not just frequency.
# Only high-conviction setups pass — this is the main win-rate lever.
MIN_AI_CONFIDENCE = float(_optional("MIN_AI_CONFIDENCE", "0.50")) 
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

# Trendline slope bounds — wider range to catch more scalp setups on 5m
TRENDLINE_MIN_SLOPE = float(_optional("TRENDLINE_MIN_SLOPE", "0.00005"))
TRENDLINE_MAX_SLOPE = float(_optional("TRENDLINE_MAX_SLOPE", "0.015"))

# Volume confirmation: require at least average volume — filters low-conviction moves
VOLUME_CONFIRM_MULTIPLIER = float(_optional("VOLUME_CONFIRM_MULTIPLIER", "1.0"))
VOLUME_LOOKBACK           = 20    # periods for average volume baseline

# ── Risk management ───────────────────────────────────────────────────────────
RISK_PER_TRADE = float(_optional("RISK_PER_TRADE", "0.02"))  # 2 % of balance (legacy — kept for fallback)

# Percentage-based position sizing — what % of the free balance to deploy per trade.
# Overridden at runtime by the dashboard UI slider (bot_state.trade_allocation_pct).
# Default 30 % means a $12 balance → ~$3.60 margin per trade at 1× leverage.
TRADE_ALLOCATION_PCT = float(_optional("TRADE_ALLOCATION_PCT", "30.0"))  # 30 % of balance

# Futures settings
USE_FUTURES       = True      # Enabled for AWS Static IP
MAX_LEVERAGE             = 20        # Cap AI-selected leverage
DEFAULT_LEVERAGE         = 15
MAX_CONCURRENT_POSITIONS = int(_optional("MAX_CONCURRENT_POSITIONS", "5"))  # capital guard


# Minimum free balance (USDT) before the bot will open a new trade.
# Set low enough to work with small balances like $12.
MIN_TRADE_BALANCE = float(_optional("MIN_TRADE_BALANCE", "2.0"))  # $2 minimum

# ATR-based stop loss — tighter SL means shorter distance to TP at same R:R
ATR_PERIOD     = 14
ATR_MULTIPLIER = float(_optional("ATR_MULTIPLIER", "1.2"))   # SL = entry − 1.2×ATR (tight, proportional to volatility)

# Structural SL — use 15m pivot lows to anchor SL at real support levels
# instead of a pure ATR mathematical level.  Much harder to stop-hunt.
USE_STRUCTURAL_SL     = True
STRUCTURAL_SL_BUFFER  = float(_optional("STRUCTURAL_SL_BUFFER", "0.002"))   # 0.2% below pivot low
SL_MIN_ATR_MULT       = float(_optional("SL_MIN_ATR_MULT", "1.5"))          # floor: SL ≥ 1.5×ATR below entry

# Trailing SL — only activate trailing when sufficiently in profit.
# Prevents TSL from tightening during early entry noise and causing premature exits.
TSL_ACTIVATION_ATR_MULT = float(_optional("TSL_ACTIVATION_ATR_MULT", "1.0"))  # trail only when profit ≥ 1×ATR
TSL_ATR_MULTIPLIER      = float(_optional("TSL_ATR_MULTIPLIER", "2.0"))        # wider trail once active

# TP targeting: 1.5x R:R — tight targets that price actually reaches = high win rate.
# Research: 1.5x R:R with 60%+ win rate is more profitable than 3.5x with 27% win rate.
SWING_TP_BUFFER   = 0.002    # sell 0.2% below previous swing high (tighter = more hits)
MIN_RR_FALLBACK   = 1.5      # fixed fallback R:R — achievable on every 5m scalp

# ATR-based TP multiplier for scalp strategies (vwap_bounce, ema_cross, momentum_scalp)
# TP = entry + ATR × SCALP_TP_ATR_MULT — keeps targets proportional to actual volatility
SCALP_TP_ATR_MULT = float(_optional("SCALP_TP_ATR_MULT", "1.5"))

# Swing strategies use slightly wider TP (2x R:R) — more room to run
SWING_RR = float(_optional("SWING_RR", "2.0"))

# FIX #8 — structure-based break-even: trail SL below the most recent pivot low
# formed AFTER entry, rather than a fixed +5 % price level
BE_PIVOT_LOOKBACK = 10       # scan last N closed candles for post-entry pivot

# Max SL distance as % of entry — prevents oversized losses on sudden volatility spikes.
# If the computed SL is wider than this, the trade is skipped entirely.
MAX_SL_PCT = float(_optional("MAX_SL_PCT", "3.0"))  # 3.0% max SL distance

# Cooldown after SL hit — short cooldown for high-frequency scalping
COOLDOWN_SECONDS = int(_optional("COOLDOWN_SECONDS", "1"))

# Portfolio-level daily loss circuit breaker
MAX_DAILY_LOSS_PCT = float(_optional("MAX_DAILY_LOSS_PCT", "0.50"))  # halt at -50 %

# ── Higher-timeframe trend filter ─────────────────────────────────────────────
# 15m 50-EMA is the trend filter for scalp/swing — fast enough to be relevant on 5m
HTF_EMA_PERIOD    = 50       # coin must be above 50-EMA on the 15m
HTF_CANDLE_LIMIT  = 60       # how many 15m candles to fetch (15h window)

# ── ADX directional filter ────────────────────────────────────────────────────
# ADX ≥ 20 ensures there's real momentum — avoids choppy ranging markets
ADX_TREND_THRESHOLD = 20
ADX_REQUIRE_DIRECTION = True

# ── VWAP Bounce strategy (scalping) ──────────────────────────────────────────
# Rolling session: 288 × 5m = 24 h
VWAP_SESSION_CANDLES    = int(_optional("VWAP_SESSION_CANDLES", "288"))
# Arm the VWAP watcher when price is within 0.4% of VWAP
VWAP_PROXIMITY          = float(_optional("VWAP_PROXIMITY", "0.004"))
# Volume confirmation for VWAP bounce
VWAP_VOLUME_MULTIPLIER  = float(_optional("VWAP_VOLUME_MULTIPLIER", "0.8"))

# ── Coin selection — volatility filter ────────────────────────────────────────
# Minimum 24h high-low range as % of price; filters stablecoins & low-vol coins
# 3% daily range = coin can actually produce scalping / swing profits
MIN_DAILY_RANGE_PCT = float(_optional("MIN_DAILY_RANGE_PCT", "3.0"))

# Dashboard — throttle live price broadcast to avoid WS overload
# P&L updates at most every N seconds per open position
PRICE_UPDATE_THROTTLE_SECS = float(_optional("PRICE_UPDATE_THROTTLE_SECS", "2.0"))

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = _optional("LOG_LEVEL", "INFO").upper()
LOG_FILE  = _optional("LOG_FILE", "bot.log")

# ── Macro context ─────────────────────────────────────────────────────────────
BTC_SYMBOL    = "BTC/USDT"
BTC_TIMEFRAME = "15m"   # 15m BTC trend used as macro filter (matches scalp timeframe)

# ── News keywords ─────────────────────────────────────────────────────────────
NEGATIVE_KEYWORDS = [
    "hack", "exploit", "scam", "fraud", "rug", "lawsuit",
    "sec", "ban", "delist", "bankrupt", "insolvent", "crash",
    "ponzi", "phishing", "stolen", "breach",
]
