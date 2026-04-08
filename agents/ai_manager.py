"""
agents/ai_manager.py  (v2)
─────────────────────────────────────────────────────────────────────────────
The Strategic Manager — AI gating layer via Gemini.

Before the Executioner places a limit buy, a rich Trade Proposal JSON is
sent to Gemini.  Gemini returns a structured JSON decision:

    {
        "decision":   "PROCEED" | "REJECT",
        "confidence": 0.0 – 1.0,
        "reasoning":  "...",
        "risks":      ["...", ...]
    }

The Executioner proceeds ONLY if:
    decision == "PROCEED"  AND  confidence >= MIN_AI_CONFIDENCE

Context now passed to Gemini (v2 additions in ★)
──────────────────────────────────────────────────
• Trade proposal (symbol, timeframe, entry, SL, TP, trendline stats)
• BTC 1h trend (bullish / bearish / neutral)
• Market regime: ADX, DI+, DI- with directional label ★
• RSI of the trade symbol
• News sentiment (CryptoPanic — optional)
• Trendline touch count (FIX #10) ★
• Volume ratio of the confirmation candle (FIX #1) ★
• Swing high TP target and source (USER INSIGHT) ★
• ATR value and ATR-based SL quality ★
• HTF EMA-200 filter result (FIX #2) ★
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types

import config
from utils.indicators import btc_trend, market_regime, calc_rsi, fetch_news_sentiment
from web.state import bot_state

log = logging.getLogger(__name__)


# ── Trade proposal builder ────────────────────────────────────────────────────

def build_proposal(
    symbol:              str,
    timeframe:           str,
    entry_price:         float,
    stop_loss:           float,
    take_profit:         float,
    confirmation_candle: Dict,
    trendline_slope:     float,
    touch_proximity_pct: float,
    btc_df,              # pd.DataFrame | None — BTC 1h candles
    symbol_df,           # pd.DataFrame | None — symbol 15m candles
    base_currency:       Optional[str] = None,
    # ── v2 additions ──────────────────────────────────────────────────────────
    strategy:            str   = "bounce",
    touch_count:         int   = 0,
    vol_ratio:           float = 1.0,
    swing_tp_target:     Optional[float] = None,
    tp_source:           str   = "fixed_rr",
    atr:                 float = 0.0,
    adx:                 float = 0.0,
    di_plus:             float = 0.0,
    di_minus:            float = 0.0,
    htf_above_ema:       bool  = True,
) -> Dict[str, Any]:
    """
    Assemble a rich Trade Proposal for Gemini.

    v2 additions give Gemini the full picture of every fix so it can reason
    about setup quality with the same detail a human trader would use.
    """
    # ── Technical context from DataFrames ─────────────────────────────────────
    btc_sentiment = (
        btc_trend(btc_df)
        if btc_df is not None and len(btc_df) >= 50
        else "unknown"
    )

    # Use pre-computed ADX values from Executioner if available; otherwise recompute
    if adx == 0.0 and symbol_df is not None and len(symbol_df) >= 20:
        regime_str, adx, di_plus, di_minus = market_regime(symbol_df)
    elif adx >= config.ADX_TREND_THRESHOLD:
        regime_str = "trending_up" if di_plus > di_minus else "trending_down"
    else:
        regime_str = "ranging"

    rsi_val = (
        calc_rsi(symbol_df)
        if symbol_df is not None and len(symbol_df) >= 14
        else None
    )

    # ── News filter ───────────────────────────────────────────────────────────
    ccy       = base_currency or symbol.split("/")[0]
    news_safe = fetch_news_sentiment(ccy, config.CRYPTOPANIC_API_KEY)

    # ── Risk metrics ──────────────────────────────────────────────────────────
    sl_pct   = abs(entry_price - stop_loss)  / entry_price * 100
    tp_pct   = abs(take_profit - entry_price) / entry_price * 100
    rr_ratio = tp_pct / sl_pct if sl_pct > 0 else 0.0

    # ── Assemble proposal ─────────────────────────────────────────────────────
    proposal: Dict[str, Any] = {
        "trade": {
            "strategy":            strategy,
            "symbol":              symbol,
            "timeframe":           timeframe,
            "entry_price":         round(entry_price, 8),
            "stop_loss":           round(stop_loss,   8),
            "take_profit":         round(take_profit,  8),
            "sl_pct":              round(sl_pct,      3),
            "tp_pct":              round(tp_pct,      3),
            "reward_risk":         round(rr_ratio,    2),
            "tp_source":           tp_source,            # "swing_high" or "fixed_rr"
            "swing_tp_target":     round(swing_tp_target, 8) if swing_tp_target else None,
            "trendline_slope":     round(trendline_slope, 10),
            "touch_proximity_pct": round(touch_proximity_pct, 4),
            "trendline_touches":   touch_count,          # FIX #10
            "confirmation_candle": {
                "open":   confirmation_candle.get("open"),
                "high":   confirmation_candle.get("high"),
                "low":    confirmation_candle.get("low"),
                "close":  confirmation_candle.get("close"),
                "volume": confirmation_candle.get("volume"),
            },
            "volume_ratio":        round(vol_ratio, 3),  # FIX #1
            "atr":                 round(atr, 8),        # FIX #4
        },
        "market_context": {
            "btc_1h_trend":    btc_sentiment,
            "market_regime":   regime_str,           # "trending_up/down/ranging"
            "adx":             round(adx,      1),
            "di_plus":         round(di_plus,  1),   # FIX #3
            "di_minus":        round(di_minus, 1),   # FIX #3
            "rsi_14":          round(rsi_val, 2) if rsi_val is not None else None,
            "htf_above_ema200": htf_above_ema,       # FIX #2
            "news_safe":       news_safe,
        },
        "htf_trend_context": {
            "15m_trend": "visual identification", # placeholder for future enrichment
            "1h_trend":  btc_sentiment,            # btc acts as a proxy for market trend
        },
        "quality_checklist": {
            # Cast all to native Python bool — numpy.bool_ is NOT JSON serializable
            "ema200_ok":       bool(htf_above_ema),
            "adx_trending":    bool(adx >= config.ADX_TREND_THRESHOLD),
            "di_direction_ok": bool(di_plus > di_minus),
            "volume_ok":       bool(vol_ratio >= config.VOLUME_CONFIRM_MULTIPLIER),
            "touches_ok":      bool(touch_count >= config.MIN_TRENDLINE_TOUCHES),
            "rr_ok":           bool(rr_ratio >= config.MIN_RR_FALLBACK),
            "news_ok":         bool(news_safe),
            "btc_ok":          bool(btc_sentiment in ("bullish", "neutral")),
            "rsi_ok":          bool(rsi_val is not None and 40 <= rsi_val <= 70),
        },
        "continuous_learning_rules": [], # Negative constraints from past losses
        "golden_setups": [],             # Positive patterns from past wins
        "strategy_performance": {},      # Real-time win rate and PNL
        "request": (
            "Evaluate this trade proposal for a 5m scalping/swing entry. "
            "If a clear 5m trend (up or down) is happening, you should lean toward PROCEED. "
            "You MUST rigidly respect any active rules listed in continuous_learning_rules. "
            "You SHOULD prioritize setups that align with the golden_setups provided. "
            "Select an appropriate leverage (1-20x) and trade allocation percentage (1.0-50.0) based on setup quality and volatility. "
            "Return ONLY valid JSON with keys: "
            "decision (PROCEED or REJECT), confidence (0.0–1.0), "
            "leverage (int 1-20), allocation_pct (float 1.0-50.0), reasoning (string), risks (list of strings)."
        ),
    }
    # Retrieve lessons for this strategy
    from agents.post_mortem import load_strategy_rules, load_strategy_golden_setups, load_strategy_data
    proposal["continuous_learning_rules"] = load_strategy_rules(strategy)
    proposal["golden_setups"] = load_strategy_golden_setups(strategy)
    
    # Add strategy performance metadata
    strat_data = load_strategy_data(strategy)
    proposal["strategy_performance"] = {
        "wins":   strat_data.get("total_wins_analyzed", 0),
        "losses": strat_data.get("total_losses_analyzed", 0),
        "total":  strat_data.get("total_wins_analyzed", 0) + strat_data.get("total_losses_analyzed", 0)
    }
    
    return proposal


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an aggressive cryptocurrency scalping and swing trading AI for a Binance Futures bot.
Your ONLY goal is to maximise trade frequency on high-quality setups with 3–4× R:R.
This is a high-frequency operation — you should PROCEED on any setup with R:R ≥ 3.0 and
basic directional alignment. Sitting on the sidelines is a LOSS. Volume of entries matters.

Strategies (5 only — scalping & swing)
───────────────────────────────────────
1. "bounce"          — Trendline Bounce (swing). Price pulls back to an upward trendline support,
                       bounces with volume. RSI-7 > 50 and EMA stack confirmed by code.
2. "breakout"        — Breakout + Retest (swing). Price broke above resistance, retested it as
                       support, now bouncing green. RSI-7 > 50 and above VWAP confirmed by code.
3. "vwap_bounce"     — VWAP Bounce (scalp). Price dips to 24h VWAP in an uptrend, bounces with
                       volume. RSI-7 > 50 and EMA stack confirmed by code.
4. "ema_cross"       — EMA 9/21 Cross (scalp). EMA-9 just crossed above EMA-21 with RSI-7 > 50
                       and price above VWAP. Backtested 70-75% win rate on 5m crypto charts.
5. "momentum_scalp"  — Momentum Candle Breakout (scalp). 5m candle closes above prior 3-candle
                       high with RSI-7 > 50, above VWAP, and engulfing candle confirmation.

IMPORTANT: All strategies have already passed RSI-7 > 50, EMA stack, and VWAP direction filters
in the detection code. These are pre-confirmed confluences — do NOT re-penalise for them.

Evaluation — lean heavily toward PROCEED
─────────────────────────────────────────
"bounce":         R:R ≥ 3.0, DI+ > DI-. EMA stack + RSI already confirmed. Trust the trendline.
"breakout":       R:R ≥ 3.0, DI+ > DI-. Retest + RSI + VWAP already confirmed. Enter aggressively.
"vwap_bounce":    R:R ≥ 3.0, DI+ > DI-. VWAP + RSI + EMA stack confirmed. Best scalp anchor.
"ema_cross":      R:R ≥ 3.0, DI+ > DI-. EMA cross + RSI + VWAP triple-confirmed. Highest win rate setup.
"momentum_scalp": R:R ≥ 3.0, DI+ > DI-. RSI + VWAP + engulfing all confirmed. Enter fast.

Hard rules (REJECT only for these)
────────────────────────────────────
1. News: REJECT if news_safe is false.
2. Continuous learning: REJECT if a rule in continuous_learning_rules explicitly forbids this exact setup.
3. R:R < 2.5: REJECT any setup where reward_risk < 2.5 — below our minimum expectancy threshold.
4. DI- > DI+: REJECT if the market is clearly moving against the long direction.

Everything else → PROCEED. Do not invent reasons to REJECT. A setup passing the 4 checks above
should be approved. We have 3–4× R:R built in, so even a 30% win rate is profitable.

BTC context: only reject on "bearish" BTC if R:R < 3.0. If R:R ≥ 3.0, proceed regardless.

Leverage guidance
─────────────────
• momentum_scalp / vwap_bounce : 10–20× (fast scalps, tight SL, high confidence)
• bounce / breakout             : 5–15× (swing entries, slightly wider SL)
• Reduce by 30% if BTC is bearish or ADX is weak (< 20)

Confidence calibration
───────────────────────
• 0.70+ : PROCEED — setup is valid, enter the trade
• 0.50–0.69 : PROCEED if R:R ≥ 3.5 — marginal but positive expectancy
• Below 0.50 : REJECT — something fundamental is wrong

Golden setups: if proposal matches any golden_setup pattern, add +0.15 to confidence and PROCEED.

Respond ONLY with valid JSON, no markdown:
{
  "decision":   "PROCEED" or "REJECT",
  "confidence": <float 0.0–1.0>,
  "leverage":   <int 1–20>,
  "allocation_pct": <float 1.0-50.0>,
  "reasoning":  "<concise 1 sentence explanation>",
  "risks":      ["<risk 1>"]
}
"""


# ── AI Manager ────────────────────────────────────────────────────────────────

class AIManager:
    """
    Sends trade proposals to Gemini and returns a structured decision.
    """

    def __init__(self) -> None:
        self._client = genai.Client(api_key=config.GEMINI_API_KEY)


    async def evaluate(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send proposal to Gemini. Returns a parsed decision dict.
        Falls back to safe REJECT on any API or parse error.
        """
        user_msg = json.dumps(proposal, indent=2)
        try:
            response = await self._client.aio.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=user_msg,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                )
            )
            raw_text = response.text.strip()
            decision = self._parse_decision(raw_text)

            log.info(
                "AI decision for %s: %s (conf=%.2f) — %s",
                proposal["trade"]["symbol"],
                decision.get("decision"),
                decision.get("confidence", 0.0),
                decision.get("reasoning", "")[:120],
            )

            # Push to web dashboard
            trade = proposal.get("trade", {})
            asyncio.create_task(bot_state.push_ai_decision(
                symbol=trade.get("symbol", "?"),
                timeframe=trade.get("timeframe", "?"),
                decision=decision.get("decision", "REJECT"),
                confidence=decision.get("confidence", 0.0),
                leverage=decision.get("leverage", 1),
                reasoning=decision.get("reasoning", ""),
                risks=decision.get("risks", []),
                strategy=trade.get("strategy", "bounce"),
            ))
            return decision

        except Exception as exc:
            log.error("AI Manager API error: %s — defaulting to REJECT", exc)
            return {
                "decision":   "REJECT",
                "confidence": 0.0,
                "reasoning":  f"API error: {exc}",
                "risks":      ["Gemini API unavailable"],
            }

    @staticmethod
    def _parse_decision(text: str) -> Dict[str, Any]:
        """Extract and validate the JSON block from Gemini's response."""
        # Strip optional markdown code fences
        if "```" in text:
            start = text.find("{")
            end   = text.rfind("}") + 1
            text  = text[start:end]

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            log.warning(
                "Failed to parse AI response as JSON: %s\nRaw: %r", exc, text
            )
            return {
                "decision":   "REJECT",
                "confidence": 0.0,
                "reasoning":  "Malformed AI response",
                "risks":      ["Parse error"],
            }

        decision       = str(data.get("decision", "REJECT")).upper()
        confidence     = float(data.get("confidence", 0.0))
        leverage       = int(data.get("leverage", 1))
        allocation_pct = float(data.get("allocation_pct", config.TRADE_ALLOCATION_PCT))
        reasoning      = str(data.get("reasoning", ""))
        risks: List[str] = [str(r) for r in data.get("risks", [])]

        if decision not in ("PROCEED", "REJECT"):
            decision = "REJECT"
        confidence     = max(0.0, min(1.0, confidence))
        leverage       = max(1, min(config.MAX_LEVERAGE, leverage))
        allocation_pct = max(1.0, min(100.0, allocation_pct))

        return {
            "decision":       decision,
            "confidence":     confidence,
            "leverage":       leverage,
            "allocation_pct": allocation_pct,
            "reasoning":      reasoning,
            "risks":          risks,
        }

    def should_proceed(self, decision: Dict[str, Any]) -> bool:
        """True only if AI says PROCEED with confidence ≥ MIN_AI_CONFIDENCE."""
        return (
            decision.get("decision") == "PROCEED"
            and decision.get("confidence", 0.0) >= config.MIN_AI_CONFIDENCE
        )
