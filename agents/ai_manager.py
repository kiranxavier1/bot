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
        "continuous_learning_rules": [], # Injected dynamically prior to evaluation
        "request": (
            "Evaluate this trade proposal. "
            "You MUST rigidly respect any active rules listed in continuous_learning_rules. "
            "Return ONLY valid JSON with keys: "
            "decision (PROCEED or REJECT), confidence (0.0–1.0), "
            "reasoning (string), risks (list of strings)."
        ),
    }
    # Retrieve lessons for this strategy
    from agents.post_mortem import load_strategy_rules
    lessons = load_strategy_rules(strategy)
    proposal["continuous_learning_rules"] = lessons
    
    return proposal


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a professional cryptocurrency risk manager reviewing trade proposals
for an automated Binance Spot trading bot targeting high-volume, volatile coins
(BTC, ETH, SOL, BNB etc.) for scalping and swing entries.

Strategies Supported
────────────────────
The bot executes 6 distinct strategies. The `trade.strategy` field tells you which triggered:
1. "bounce"         — 3rd-Touch Trendline Bounce. Upward sloping trendline, price pulls to support, volume confirmed bounce.
2. "mean_reversion" — Bollinger Band Fade. ONLY in ranging markets (ADX < 25). Price dips below lower BB and closes inside.
3. "fvg"            — Fair Value Gap (SMC). Buying into a bullish imbalance zone, optionally with a prior liquidity sweep.
4. "breakout"       — Breakout + Retest. Broke above resistance, pulled back to retest it as support, now bouncing green.
5. "vwap_bounce"    — VWAP Bounce (Scalping). Price pulls to the 24h VWAP in an uptrend and bounces with volume. High win-rate intraday setup.
6. "rsi_divergence" — RSI Divergence Swing. Bullish divergence (price lower low, RSI higher low) in oversold territory. Proven reversal signal.

Evaluation criteria by strategy
────────────────────────────────
"bounce":         Needs trending_up (ADX ≥ 25, DI+ > DI-), above HTF EMA-200, volume ≥ 1.3×, trendline touches ≥ 2, R:R ≥ 2.0.
"mean_reversion": MUST be ranging (ADX < 25), R:R ≥ 1.5. BTC trend less critical.
"fvg":            Needs trending_up, above HTF EMA-200. If "sweep":true in proposal → HIGHER conviction, lower confidence bar.
"breakout":       Direction confirmed (DI+ > DI-), R:R ≥ 2.0. Retest pattern is already confirmed by code — trust it.
"vwap_bounce":    Mild trend (DI+ > DI-), above HTF EMA-200. VWAP is the institutional fair-value anchor — strong reversal probability. R:R ≥ 1.5 acceptable.
"rsi_divergence": Price lower low + RSI higher low confirmed by code. RSI was oversold. R:R ≥ 1.5 acceptable. Very reliable reversal signal.

Universal rules
───────────────
1. BTC macro: avoid longs when BTC 1h is "bearish" UNLESS R:R ≥ 3.0 OR strategy is "mean_reversion"/"rsi_divergence".
2. News: REJECT immediately if news_safe is false.
3. CRITICAL: If any rule in continuous_learning_rules explicitly forbids the specific conditions in this proposal, REJECT.
4. For all strategies: minimum R:R = 1.5. Prefer 2.0+. Never enter negative-expectancy setups.
5. Accept MORE opportunities: if the quality_checklist majority passes and R:R ≥ 2.0, lean toward PROCEED even on borderline regime conditions — we want to capture scalp and swing moves, not sit on the sidelines.

Confidence calibration
───────────────────────
• 0.90+ : Exceptional — all checks pass, ideal macro, strong volume
• 0.80–0.89 : Strong — most checks pass, minor concerns
• 0.70–0.79 : Acceptable — core risk-reward is sound, some uncertainty
• Below 0.70 : Marginal — should REJECT unless extraordinary R:R

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
                strategy=trade.get("strategy", "bounce"),
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
