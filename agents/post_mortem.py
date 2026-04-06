"""
agents/post_mortem.py  (v3 — AI Evaluator)
─────────────────────────────────────────────────────────────────────────────
Self-improving Strategy Evolution Engine.

Consumes losing trades from the global queue, performs deep AI post-mortem
analysis, and REWRITES / APPENDS per-strategy rule files so the trading AI
follows updated strategies in real-time.

Architecture
────────────
1. RetrainingAgent listens to `bot_state.losing_trades_queue`.
2. On each loss, Claude generates a rich post-mortem report:
   - cause, market_conditions, what_went_wrong, solution, new_rule, severity
3. The report is saved as a timestamped JSON in `data/trade_analyses/`.
4. The strategy's rule file at `data/strategies/{strategy}.json` is
   appended/rewritten with the new rule.
5. AIManager.build_proposal() hot-loads the latest rules on every call.
─────────────────────────────────────────────────────────────────────────────
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types
import config

from web.state import bot_state

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
_STRATEGIES_DIR   = os.path.join(_BASE_DIR, '..', 'data', 'strategies')
_ANALYSES_DIR     = os.path.join(_BASE_DIR, '..', 'data', 'trade_analyses')
_LEGACY_PATH      = os.path.join(_BASE_DIR, '..', 'data', 'strategy_lessons.json')


def _ensure_dirs():
    os.makedirs(_STRATEGIES_DIR, exist_ok=True)
    os.makedirs(_ANALYSES_DIR, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trade_id(trade: Dict) -> str:
    """Generate a unique trade ID from timestamp + symbol + strategy."""
    ts = trade.get("closed_at", _now_iso()).replace(":", "-").replace("+", "p")
    sym = trade.get("symbol", "UNK").replace("/", "-")
    strat = trade.get("strategy", "unknown")
    return f"{ts}_{sym}_{strat}"


# ── Strategy file operations ─────────────────────────────────────────────────

def _strategy_path(strategy: str) -> str:
    return os.path.join(_STRATEGIES_DIR, f"{strategy}.json")


def _load_strategy_file(strategy: str) -> Dict[str, Any]:
    """Load a strategy's learning file, creating one if it doesn't exist."""
    path = _strategy_path(strategy)
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "strategy":              strategy,
        "rules":                 [],
        "total_losses_analyzed": 0,
        "last_updated":          None,
        "evolution_log":         [],
    }


def _save_strategy_file(strategy: str, data: Dict[str, Any]):
    _ensure_dirs()
    path = _strategy_path(strategy)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


# ── Analysis file operations ─────────────────────────────────────────────────

def _save_analysis(trade_id: str, analysis: Dict[str, Any]):
    _ensure_dirs()
    path = os.path.join(_ANALYSES_DIR, f"{trade_id}.json")
    with open(path, 'w') as f:
        json.dump(analysis, f, indent=2)


def load_analysis(trade_id: str) -> Optional[Dict[str, Any]]:
    """Load a specific trade analysis by ID."""
    path = os.path.join(_ANALYSES_DIR, f"{trade_id}.json")
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    return None


def list_analyses() -> List[Dict[str, Any]]:
    """List all trade analyses (metadata only)."""
    _ensure_dirs()
    analyses = []
    for fname in sorted(os.listdir(_ANALYSES_DIR), reverse=True):
        if fname.endswith('.json'):
            path = os.path.join(_ANALYSES_DIR, fname)
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                analyses.append(data)
            except Exception:
                continue
    return analyses


# ── Public API for AIManager ─────────────────────────────────────────────────

def load_strategy_rules(strategy: str) -> List[str]:
    """Return the list of active rules for a strategy (hot-loaded each call)."""
    data = _load_strategy_file(strategy)
    return [r["rule"] for r in data.get("rules", []) if isinstance(r, dict) and "rule" in r]


def load_strategy_data(strategy: str) -> Dict[str, Any]:
    """Return the full strategy learning file."""
    return _load_strategy_file(strategy)


def list_all_strategies() -> List[Dict[str, Any]]:
    """Return summary info for all strategies."""
    _ensure_dirs()
    all_strategies = ["bounce", "mean_reversion", "fvg", "breakout", "vwap_bounce", "rsi_divergence"]
    result = []
    for strat in all_strategies:
        data = _load_strategy_file(strat)
        result.append({
            "strategy":     strat,
            "rule_count":   len(data.get("rules", [])),
            "losses_analyzed": data.get("total_losses_analyzed", 0),
            "last_updated": data.get("last_updated"),
        })
    return result


# ── RetrainingAgent ──────────────────────────────────────────────────────────

class RetrainingAgent:
    """
    AI Evaluator Agent — consumes losing trades, produces deep analysis,
    and evolves strategy rule files in real time.
    """

    def __init__(self, exchange=None):
        self._client = genai.Client(api_key=config.GEMINI_API_KEY)
        self._system_instruction = (
            "You are a quantitative trading analyst. You analyze losing trades "
            "and produce structured JSON post-mortem reports. Be specific and "
            "actionable. Every rule you create must be concrete enough for an AI "
            "to evaluate on the next trade."
        )
        self._exchange = exchange
        self._running = False
        _ensure_dirs()

    def start(self) -> asyncio.Task | None:
        if not self._running:
            self._running = True
            task = asyncio.create_task(self._monitor_queue(), name="retrainer")
            log.info("🧠 AI Evaluator Agent started — listening for losing trades.")
            return task
        return None

    async def _monitor_queue(self):
        while self._running:
            try:
                trade = await bot_state.losing_trades_queue.get()
                log.info(
                    "🧠 AI Evaluator analyzing losing trade: %s (Strategy: %s)",
                    trade.get("symbol"), trade.get("strategy"),
                )
                await self._analyze_loss(trade)
                bot_state.losing_trades_queue.task_done()
            except asyncio.CancelledError:
                self._running = False
                break
            except Exception as e:
                log.error("AI Evaluator error processing trade: %s", e)

    async def _analyze_loss(self, trade: Dict[str, Any]):
        symbol = trade.get("symbol", "UNKNOWN")
        timeframe = trade.get("timeframe", "15m")
        strategy = trade.get("strategy", "unknown")
        trade_id = _trade_id(trade)

        # Load current strategy rules for context
        current_rules = load_strategy_rules(strategy)
        rules_context = "\n".join(f"  - {r}" for r in current_rules) if current_rules else "  (none yet)"

        # ── Build the analysis prompt ─────────────────────────────────────────
        prompt = f"""You are an elite quantitative trading AI evaluator performing a deep post-mortem analysis on a losing trade.

## Trade Details
- Symbol: {symbol}
- Strategy: {strategy}
- Timeframe: {timeframe}
- Entry Price: {trade.get("entry")}
- Exit Price (SL): {trade.get("exit")}
- Stop Loss: {trade.get("sl")}
- Take Profit: {trade.get("tp")}
- P&L %: {trade.get("pnl_pct")}
- Real Trade: {trade.get("real", False)}
- Opened At: {trade.get("opened_at")}
- Closed At: {trade.get("closed_at")}

## Current Strategy Rules
{rules_context}

## Your Task
Analyze this losing trade deeply. Consider:
1. What structural, momentum, or timing failure caused this loss?
2. What were the likely market conditions that trapped us?
3. What specific, actionable rule can prevent this exact type of loss in the future?
4. Should any existing rule be strengthened or modified?

Output ONLY valid JSON (no markdown):
{{
  "cause": "<1-2 sentence root cause of the loss>",
  "market_conditions": "<description of the market conditions that led to the loss>",
  "what_went_wrong": "<detailed explanation of what went wrong with this specific trade setup>",
  "solution": "<concrete solution to avoid this type of loss — be specific about conditions/thresholds>",
  "new_rule": "<a precise, machine-readable rule to add to the strategy's rule file — e.g. 'Do not enter {strategy} trades when RSI is above 70 on the signal timeframe'>",
  "severity": "<low|medium|high|critical>",
  "confidence": <0.0-1.0 how confident you are in this analysis>,
  "should_modify_existing_rule": false,
  "existing_rule_modification": null
}}"""

        try:
            response = await self._client.aio.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self._system_instruction,
                )
            )
            raw = response.text.strip()

            # Strip markdown if present
            if "```" in raw:
                start = raw.find("{")
                end = raw.rfind("}") + 1
                raw = raw[start:end]

            analysis = json.loads(raw)

            # ── Build full analysis record ────────────────────────────────────
            full_analysis = {
                "trade_id":     trade_id,
                "trade":        trade,
                "analysis":     analysis,
                "analyzed_at":  _now_iso(),
            }

            # ── Save analysis file ────────────────────────────────────────────
            _save_analysis(trade_id, full_analysis)

            # ── Update strategy file ──────────────────────────────────────────
            new_rule = analysis.get("new_rule")
            if new_rule:
                self._evolve_strategy(strategy, analysis, trade_id)

            # ── Push to dashboard via BotState ────────────────────────────────
            await bot_state.push_trade_analysis(full_analysis)

            log.info(
                "✅ AI Evaluator: strategy '%s' evolved — new rule: %s",
                strategy, new_rule,
            )

        except Exception as e:
            log.error("AI Evaluator analysis failed for %s (%s): %s", symbol, strategy, e)
            # Save a minimal analysis even on failure
            fallback = {
                "trade_id":    trade_id,
                "trade":       trade,
                "analysis":    {
                    "cause": f"Analysis failed: {e}",
                    "market_conditions": "Unable to determine",
                    "what_went_wrong": "AI analysis error",
                    "solution": "Retry analysis on next occurrence",
                    "new_rule": None,
                    "severity": "low",
                    "confidence": 0.0,
                },
                "analyzed_at": _now_iso(),
                "error":       str(e),
            }
            _save_analysis(trade_id, fallback)
            await bot_state.push_trade_analysis(fallback)

    def _evolve_strategy(self, strategy: str, analysis: Dict, trade_id: str):
        """Append a new rule to the strategy file and log the evolution."""
        data = _load_strategy_file(strategy)
        new_rule = analysis.get("new_rule")
        if not new_rule:
            return

        # ── Append new rule ───────────────────────────────────────────────────
        rule_entry = {
            "rule":       new_rule,
            "severity":   analysis.get("severity", "medium"),
            "confidence": analysis.get("confidence", 0.5),
            "cause":      analysis.get("cause", ""),
            "added_at":   _now_iso(),
            "trade_id":   trade_id,
        }
        data["rules"].append(rule_entry)

        # Keep only the last 20 rules per strategy to prevent context bloat
        if len(data["rules"]) > 20:
            data["rules"] = data["rules"][-20:]

        # ── Update metadata ───────────────────────────────────────────────────
        data["total_losses_analyzed"] = data.get("total_losses_analyzed", 0) + 1
        data["last_updated"] = _now_iso()

        # ── Evolution log ─────────────────────────────────────────────────────
        data["evolution_log"].append({
            "action":     "rule_added",
            "rule":       new_rule,
            "trade_id":   trade_id,
            "timestamp":  _now_iso(),
        })
        # Keep last 50 evolution entries
        if len(data["evolution_log"]) > 50:
            data["evolution_log"] = data["evolution_log"][-50:]

        _save_strategy_file(strategy, data)
        log.info("📝 Strategy '%s' file updated — %d active rules", strategy, len(data["rules"]))

    # ── Legacy compatibility (class-level access) ─────────────────────────────
    @classmethod
    def load_lessons(cls, strategy: str) -> List[str]:
        """Backward-compatible: returns rule strings for a strategy."""
        return load_strategy_rules(strategy)
