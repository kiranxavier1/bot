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
            "Select an appropriate leverage (1-20x) and trade allocation percentage (10.0-30.0) based on setup quality and volatility. "
            "Return ONLY valid JSON with keys: "
            "decision (PROCEED or REJECT), confidence (0.0–1.0), "
            "leverage (int 1-20), allocation_pct (float 10.0-30.0), reasoning (string), risks (list of strings)."
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


# ── Proactive proposal builder ────────────────────────────────────────────────

def build_proactive_proposal(
    symbol:          str,
    timeframe:       str,
    market_snapshot: Dict,
    btc_df,
    strategy_signals: List[Dict] = None,
) -> Dict[str, Any]:
    """
    Build a full market analysis proposal for proactive AI trading.
    The AI receives the full technical snapshot and decides whether/where to trade.
    """
    from agents.post_mortem import load_strategy_rules, load_strategy_golden_setups

    btc_sentiment = (
        btc_trend(btc_df)
        if btc_df is not None and len(btc_df) >= 50
        else "unknown"
    )
    ccy       = symbol.split("/")[0]
    news_safe = fetch_news_sentiment(ccy, config.CRYPTOPANIC_API_KEY)

    # Detected patterns from the Mathematician (hints for the AI, not requirements)
    detected_patterns = [
        {"strategy": s["strategy"], "direction": s["direction"]}
        for s in (strategy_signals or [])
    ]

    # Aggregate continuous-learning rules and golden setups across all strategies
    # Aggregate continuous-learning rules and golden setups from global file
    all_rules: List[str] = load_strategy_rules("global")
    all_setups: List[str] = load_strategy_golden_setups("global")

    return {
        "symbol":                    symbol,
        "timeframe":                 timeframe,
        "btc_trend":                 btc_sentiment,
        "news_safe":                 news_safe,
        "market_snapshot":           market_snapshot,
        "detected_patterns":         detected_patterns,
        "continuous_learning_rules": all_rules[:12],   # cap to avoid token bloat
        "golden_setups":             all_setups[:6],
        "observer_directive":        bot_state.observer_directive,
        "request": (
            "Analyze this market snapshot and decide whether to enter a trade RIGHT NOW. "
            "You are the strategy engine — pick the best opportunity. You MUST highly prioritize making a TRADE over PASS. "
            "We want at minimum 1 trade every 5 minutes. DO NOT passively 'wait for a better setup' if the current one has positive expectancy."
            "Set entry_price, stop_loss, take_profit, leverage, allocation_pct when decision is TRADE."
        ),
    }


# ── Proactive system prompt ────────────────────────────────────────────────────

_PROACTIVE_SYSTEM_PROMPT = """\
You are an expert cryptocurrency futures trader and the PRIMARY STRATEGY ENGINE for a Binance Futures bot.
Your job: analyze the market snapshot and DECIDE WHETHER TO TRADE RIGHT NOW — back to back, candle by candle.

You do NOT wait for a "perfect" setup. You trade the best available opportunity on every candle.
If R:R ≥ 2.0 and at least 2 technical confirmations align, you TRADE.

════════════════════════════════════════
 LONG setups to look for
════════════════════════════════════════
• EMA bullish stack (9>21>50) + RSI-7 > 50 + above VWAP → momentum long
• Price above EMA50, pulling back to VWAP or EMA21, bull engulfing candle → pullback long
• EMA 9/21 just crossed up + ADX rising + DI+ > DI- → trend entry long
• BB lower band + StochRSI K < 20 crossing above D + reversal candle → mean-reversion long
• 3 consecutive bull closes + volume above avg + above EMA50 → momentum continuation long

════════════════════════════════════════
 SHORT setups to look for
════════════════════════════════════════
• EMA bearish stack (9<21<50) + RSI-7 < 50 + below VWAP → momentum short
• Price below EMA50, bouncing to VWAP or EMA21, bear engulfing candle → pullback short
• EMA 9/21 just crossed down + ADX rising + DI- > DI+ → trend entry short
• BB upper band + StochRSI K > 80 crossing below D + rejection candle → mean-reversion short
• 3 consecutive bear closes + volume above avg + below EMA50 → momentum continuation short

════════════════════════════════════════
 SL / TP rules (YOU set these)
════════════════════════════════════════
• SL LONG:  below nearest support in market_snapshot.structure.nearest_support, or entry - 1×ATR (min)
• SL SHORT: above nearest resistance in market_snapshot.structure.nearest_resistance, or entry + 1×ATR (min)
• SL cap: must be within 0.8% of entry price (no wide stops — hard limit enforced by code)
• TP: target next resistance (long) or support (short). Minimum R:R = 2.0. If no clear level, use 2× SL distance.
• Leverage: scale by ATR%:
    ATR% < 0.3% → 15–20×
    ATR% 0.3–0.6% → 10–15×
    ATR% > 0.6% → 5–10×
  Reduce 30% if BTC bearish or ADX < 20.

════════════════════════════════════════
 Decision rules
════════════════════════════════════════
• TRADE: 1+ technical confirmations from the setups above AND R:R ≥ 2.0 AND news_safe is true. Bias heavily toward TRADE over PASS.
• PASS:  ONLY if ADX < 15 AND no momentum AND no pattern — truly directionless market. Do NOT PASS to "wait for better".
• NEVER PASS just because it is not a "textbook" setup. A 60% setup with 2:1 R:R is TRADE.
• You MUST respect any rule in continuous_learning_rules (these are hard constraints from past losses).
• You MUST respect the global `observer_directive` strictly. It overrides standard logic.
• Golden setups in golden_setups should boost confidence by +0.15.
• Detected patterns from the code-level analysis (detected_patterns) are strong hints — weight them heavily.

Respond ONLY with valid JSON (no markdown):
{
  "decision":       "TRADE" or "PASS",
  "direction":      "long" or "short",
  "entry_price":    <float — use current price from market_snapshot.price.current>,
  "stop_loss":      <float>,
  "take_profit":    <float>,
  "confidence":     <float 0.0–1.0>,
  "leverage":       <int 1–20>,
  "allocation_pct": <float 10.0–30.0>,
  "strategy_used":  "<brief name: e.g. momentum_long, pullback_short, mean_reversion_long>",
  "reasoning":      "<1 concise sentence>",
  "risks":          ["<risk 1>"]
}
"""


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
"bounce":         R:R ≥ 2.0, DI+ > DI-. EMA stack + RSI already confirmed. Trust the trendline.
"breakout":       R:R ≥ 2.0, DI+ > DI-. Retest + RSI + VWAP already confirmed. Enter aggressively.
"vwap_bounce":    R:R ≥ 2.0, DI+ > DI-. VWAP + RSI + EMA stack confirmed. Best scalp anchor.
"ema_cross":      R:R ≥ 2.0, DI+ > DI-. EMA cross + RSI + VWAP triple-confirmed. Highest win rate setup.
"momentum_scalp": R:R ≥ 2.0, DI+ > DI-. RSI + VWAP + engulfing all confirmed. Enter fast.

Hard rules (REJECT only for these)
────────────────────────────────────
1. News: REJECT if news_safe is false.
2. Continuous learning: REJECT if a rule in continuous_learning_rules explicitly forbids this exact setup.
3. R:R < 2.0: REJECT any setup where reward_risk < 2.0 — below our minimum expectancy threshold.
4. DI- > DI+: REJECT if the market is clearly moving against the long direction.

Everything else → PROCEED. Do not invent reasons to REJECT. A setup passing the 4 checks above
should be approved. We have 2–3× R:R built in, so even a 35% win rate is profitable.

BTC context: only reject on "bearish" BTC if R:R < 2.5. If R:R ≥ 2.5, proceed regardless.

Leverage guidance
─────────────────
• momentum_scalp / vwap_bounce : 10–20× (fast scalps, tight SL, high confidence)
• bounce / breakout             : 5–15× (swing entries, slightly wider SL)
• Reduce by 30% if BTC is bearish or ADX is weak (< 20)

Confidence calibration
───────────────────────
• 0.60+ : PROCEED — setup is valid, enter the trade
• 0.40–0.59 : PROCEED if R:R ≥ 2.5 — marginal but positive expectancy
• Below 0.40 : REJECT — something fundamental is wrong

Golden setups: if proposal matches any golden_setup pattern, add +0.15 to confidence and PROCEED.

Respond ONLY with valid JSON, no markdown:
{
  "decision":   "PROCEED" or "REJECT",
  "confidence": <float 0.0–1.0>,
  "leverage":   <int 1–20>,
  "allocation_pct": <float 10.0-30.0>,
  "reasoning":  "<concise 1 sentence explanation>",
  "risks":      ["<risk 1>"]
}
"""


# ── AI Manager ────────────────────────────────────────────────────────────────

class AIManager:
    """
    Sends trade proposals to Anthropic and returns a structured decision.
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
                    max_output_tokens=600,
                    response_mime_type="application/json",
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
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]

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
        allocation_pct = max(10.0, min(30.0, allocation_pct))

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

    # ── Proactive market analysis ──────────────────────────────────────────────

    async def analyze_market(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Proactive market analysis — AI acts as the strategy engine.
        Sends a full market snapshot and expects: direction, entry, SL, TP.
        Falls back to PASS on any error.
        """
        user_msg = json.dumps(proposal, indent=2)
        try:
            response = await self._client.aio.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=user_msg,
                config=types.GenerateContentConfig(
                    system_instruction=_PROACTIVE_SYSTEM_PROMPT,
                    max_output_tokens=1500,
                    response_mime_type="application/json",
                )
            )
            if not response.candidates:
                log.warning("analyze_market: Gemini returned no candidates for %s", proposal.get("symbol"))
                return {"decision": "PASS", "confidence": 0.0, "reasoning": "No candidates", "risks": []}
            raw_text = response.text.strip()
            decision = self._parse_proactive_decision(raw_text)

            log.info(
                "🤖 Proactive AI [%s]: %s dir=%s conf=%.2f — %s",
                proposal.get("symbol"),
                decision.get("decision"),
                decision.get("direction", "—"),
                decision.get("confidence", 0.0),
                decision.get("reasoning", "")[:120],
            )

            asyncio.create_task(bot_state.push_ai_decision(
                symbol=proposal.get("symbol", "?"),
                timeframe=proposal.get("timeframe", "?"),
                decision=decision.get("decision", "PASS"),
                confidence=decision.get("confidence", 0.0),
                leverage=decision.get("leverage", 1),
                reasoning=decision.get("reasoning", ""),
                risks=decision.get("risks", []),
                strategy=decision.get("strategy_used", "ai_proactive"),
            ))
            return decision

        except Exception as exc:
            log.error("analyze_market API error: %s — defaulting to PASS", exc)
            return {
                "decision":   "PASS",
                "confidence": 0.0,
                "reasoning":  f"API error: {exc}",
                "risks":      ["Gemini API unavailable"],
            }

    async def evaluate_observer(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Special API call for the top-level Observer Agent to audit bot performance.
        Expected to return a JSON dict overriding config and setting macro directives.
        """
        try:
            response = await self._client.aio.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=2000,
                    response_mime_type="application/json",
                )
            )
            # response.text raises if no candidates (blocked/empty response)
            if not response.candidates:
                log.warning("Observer: Gemini returned no candidates — skipping audit")
                return {}
            raw_text = response.text.strip()

            start = raw_text.find("{")
            end   = raw_text.rfind("}") + 1
            if start != -1 and end > start:
                raw_text = raw_text[start:end]

            if not raw_text.strip():
                return {}

            return json.loads(raw_text)
        except Exception as exc:
            log.error("Observer API error: %s", exc)
            return {}

    @staticmethod
    def _parse_proactive_decision(text: str) -> Dict[str, Any]:
        """Parse the proactive AI response (TRADE/PASS with full SL/TP)."""
        # Strip markdown code fences Gemini sometimes wraps responses in
        text = text.replace("```json", "").replace("```", "").strip()
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            log.warning("Failed to parse proactive AI response: %s\nRaw: %r", exc, text)
            return {
                "decision":   "PASS",
                "confidence": 0.0,
                "reasoning":  "Malformed AI response",
                "risks":      ["Parse error"],
            }

        decision   = str(data.get("decision", "PASS")).upper()
        direction  = str(data.get("direction", "long")).lower()
        confidence = float(data.get("confidence", 0.0))
        leverage   = int(data.get("leverage", 1))
        alloc_pct  = float(data.get("allocation_pct", config.TRADE_ALLOCATION_PCT))

        if decision not in ("TRADE", "PASS"):
            decision = "PASS"
        if direction not in ("long", "short"):
            direction = "long"
        confidence = max(0.0, min(1.0, confidence))
        leverage   = max(1, min(config.MAX_LEVERAGE, leverage))
        alloc_pct  = max(10.0, min(30.0, alloc_pct))

        return {
            "decision":       decision,
            "direction":      direction,
            "entry_price":    float(data.get("entry_price", 0.0)),
            "stop_loss":      float(data.get("stop_loss",   0.0)),
            "take_profit":    float(data.get("take_profit",  0.0)),
            "confidence":     confidence,
            "leverage":       leverage,
            "allocation_pct": alloc_pct,
            "strategy_used":  str(data.get("strategy_used", "ai_proactive")),
            "reasoning":      str(data.get("reasoning", "")),
            "risks":          [str(r) for r in data.get("risks", [])],
        }

    def should_trade_proactive(self, decision: Dict[str, Any]) -> bool:
        """
        True if AI says TRADE with confidence ≥ threshold and valid SL/TP geometry.
        """
        if decision.get("decision") != "TRADE":
            return False
        if decision.get("confidence", 0.0) < config.MIN_AI_CONFIDENCE:
            return False
        entry     = float(decision.get("entry_price", 0.0))
        stop_loss = float(decision.get("stop_loss",   0.0))
        take_profit = float(decision.get("take_profit", 0.0))
        if entry <= 0 or stop_loss <= 0 or take_profit <= 0:
            return False
        direction = decision.get("direction", "long")
        if direction == "long":
            return stop_loss < entry < take_profit
        else:
            return take_profit < entry < stop_loss
