"""
agents/ai_manager.py  (v2)
─────────────────────────────────────────────────────────────────────────────
The Strategic Manager — AI gating layer via Claude.

Before the Executioner places a limit buy, a rich Trade Proposal JSON is
sent to Claude.  Claude returns a structured JSON decision:

    {
        "decision":   "PROCEED" | "REJECT",
        "confidence": 0.0 – 1.0,
        "reasoning":  "...",
        "risks":      ["...", ...]
    }

The Executioner proceeds ONLY if:
    decision == "PROCEED"  AND  confidence >= MIN_AI_CONFIDENCE

Context now passed to Claude (v2 additions in ★)
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

import anthropic

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
    Assemble a rich Trade Proposal for Claude.

    v2 additions give Claude the full picture of every fix so it can reason
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
        "quality_checklist": {
            # Give Claude a pre-computed quality scorecard
            "ema200_ok":       htf_above_ema,
            "adx_trending":    adx >= config.ADX_TREND_THRESHOLD,
            "di_direction_ok": di_plus > di_minus,
            "volume_ok":       vol_ratio >= config.VOLUME_CONFIRM_MULTIPLIER,
            "touches_ok":      touch_count >= config.MIN_TRENDLINE_TOUCHES,
            "rr_ok":           rr_ratio >= config.MIN_RR_FALLBACK,
            "news_ok":         news_safe,
            "btc_ok":          btc_sentiment in ("bullish", "neutral"),
            "rsi_ok":          (rsi_val is not None and 40 <= rsi_val <= 70),
        },
        "request": (
            "Evaluate this trendline bounce trade proposal. "
            "The quality_checklist summarises each pre-trade filter result. "
            "Return ONLY valid JSON with keys: "
            "decision (PROCEED or REJECT), confidence (0.0–1.0), "
            "reasoning (string), risks (list of strings)."
        ),
    }
    return proposal


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a professional cryptocurrency risk manager reviewing trade proposals
for an automated Binance Spot trading bot.

Strategy: 3rd-Touch Trendline Bounce
The bot detects rising trendlines (minimum 3 confirmed touches), waits for
price to pull back to the trendline a 3rd+ time, and enters on a confirmed
bounce candle.  Stop-loss is ATR-based (1.5× ATR-14).  Take-profit targets
the previous swing high or a fixed 1.5:1 RR fallback.

Your evaluation criteria
────────────────────────
1. BTC macro: avoid longs when BTC 1h trend is "bearish"
2. Market regime: prefer "trending_up" (ADX ≥ 25, DI+ > DI-)
3. RSI: avoid if RSI > 70 (overbought) or < 30 (momentum breakdown)
   Prefer RSI 40–65 for bounce entries
4. News: REJECT if news_safe is false
5. R:R: minimum 1.5:1; prefer 2:1+
6. Trendline quality:
   - Positive (upward) slope is required for support bounces
   - More touches = higher confidence
7. Volume: confirmation candle volume_ratio ≥ 1.3 × average is meaningful
8. HTF EMA-200: coin should be above its 200-EMA on the 1h timeframe
9. The quality_checklist gives you a pre-computed pass/fail for each filter.

Decision logic
──────────────
• PROCEED only when the majority of quality checks pass and overall context
  is favourable.  A confidence above 0.8 is reserved for high-quality setups
  where almost all filters pass.
• REJECT if BTC is bearish, news is bad, RSI > 70, or R:R < 1.5.
• Be strict — false positives are expensive; false negatives merely miss one
  trade.

Respond ONLY with valid JSON, no markdown:
{
  "decision":   "PROCEED" or "REJECT",
  "confidence": <float 0.0–1.0>,
  "reasoning":  "<concise 1-2 sentence explanation>",
  "risks":      ["<risk 1>", "<risk 2>"]
}
"""


# ── AI Manager ────────────────────────────────────────────────────────────────

class AIManager:
    """
    Sends trade proposals to Claude and returns a structured decision.
    """

    def __init__(self) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        self._model  = config.CLAUDE_MODEL

    async def evaluate(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send proposal to Claude.  Returns a parsed decision dict.
        Falls back to safe REJECT on any API or parse error.
        """
        user_msg = json.dumps(proposal, indent=2)
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=512,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw_text = response.content[0].text.strip()
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
                reasoning=decision.get("reasoning", ""),
                risks=decision.get("risks", []),
            ))
            return decision

        except Exception as exc:
            log.error("AI Manager API error: %s — defaulting to REJECT", exc)
            return {
                "decision":   "REJECT",
                "confidence": 0.0,
                "reasoning":  f"API error: {exc}",
                "risks":      ["Claude API unavailable"],
            }

    @staticmethod
    def _parse_decision(text: str) -> Dict[str, Any]:
        """Extract and validate the JSON block from Claude's response."""
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

        decision   = str(data.get("decision", "REJECT")).upper()
        confidence = float(data.get("confidence", 0.0))
        reasoning  = str(data.get("reasoning", ""))
        risks: List[str] = [str(r) for r in data.get("risks", [])]

        if decision not in ("PROCEED", "REJECT"):
            decision = "REJECT"
        confidence = max(0.0, min(1.0, confidence))

        return {
            "decision":   decision,
            "confidence": confidence,
            "reasoning":  reasoning,
            "risks":      risks,
        }

    def should_proceed(self, decision: Dict[str, Any]) -> bool:
        """True only if AI says PROCEED with confidence ≥ MIN_AI_CONFIDENCE."""
        return (
            decision.get("decision") == "PROCEED"
            and decision.get("confidence", 0.0) >= config.MIN_AI_CONFIDENCE
        )
