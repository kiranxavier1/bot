from __future__ import annotations
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

_STRATEGY_CACHE: Dict[str, Dict[str, Any]] = {}


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

def _strategy_path() -> str:
    return os.path.join(_STRATEGIES_DIR, "global_learning.json")


def _load_strategy_file(strategy_ignored: str = "") -> Dict[str, Any]:
    """Load the unified global learning file, creating one if it doesn't exist."""
    if "global" in _STRATEGY_CACHE:
        return _STRATEGY_CACHE["global"]
        
    path = _strategy_path()
    data = None
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except Exception:
            pass
            
    if not data:
        data = {
            "strategy":              "global",
            "rules":                 [],
            "golden_setups":         [],
            "total_losses_analyzed": 0,
            "total_wins_analyzed":   0,
            "last_updated":          None,
            "evolution_log":         [],
        }
        
    _STRATEGY_CACHE["global"] = data
    return data


def _save_strategy_file(strategy_ignored: str, data: Dict[str, Any]):
    _STRATEGY_CACHE["global"] = data
    _ensure_dirs()
    path = _strategy_path()
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
    """Return the list of active negative rules for a strategy."""
    data = _load_strategy_file(strategy)
    return [r["rule"] for r in data.get("rules", []) if isinstance(r, dict) and "rule" in r]


def load_strategy_golden_setups(strategy: str) -> List[str]:
    """Return the list of active positive golden setups for a strategy."""
    data = _load_strategy_file(strategy)
    return [s["pattern"] for s in data.get("golden_setups", []) if isinstance(s, dict) and "pattern" in s]


def load_strategy_data(strategy: str) -> Dict[str, Any]:
    """Return the full strategy learning file."""
    return _load_strategy_file(strategy)


def list_all_strategies() -> List[Dict[str, Any]]:
    """Return summary info for the unified global strategy."""
    _ensure_dirs()
    data = _load_strategy_file()
    return [{
        "strategy":     "global_learning",
        "rule_count":   len(data.get("rules", [])),
        "golden_count": len(data.get("golden_setups", [])),
        "losses_analyzed": data.get("total_losses_analyzed", 0),
        "wins_analyzed":   data.get("total_wins_analyzed", 0),
        "last_updated": data.get("last_updated"),
    }]


def clear_all_strategy_rules() -> None:
    """Clear all continuous learning rules and positive patterns from JSON files."""
    _ensure_dirs()
    global _STRATEGY_CACHE
    _STRATEGY_CACHE.clear()
    
    path = _strategy_path()
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception as e:
            log.error("Failed to delete strategy file %s: %s", path, e)


# ── RetrainingAgent ──────────────────────────────────────────────────────────

class RetrainingAgent:
    """
    Subscribes to bot_state.closed_trades_queue.
    Passes closed trades to Anthropic AI for asynchronous post-mortem analysis.
    Evolves rules inside the unified global learning file.
    """

    def __init__(self, exchange=None) -> None:
        self._client = genai.Client(api_key=config.GEMINI_API_KEY)
        self._system_instruction = (
            "You are a quantitative trading analyst. You analyze trades (wins and losses) "
            "and produce structured JSON reports. Be specific and actionable. "
            "For losses, identify preventable causes. For wins, identify 'Golden Setup' "
            "confluence factors that should be repeated."
        )
        self._exchange = exchange
        self._running = False
        _ensure_dirs()

    def start(self) -> Optional[asyncio.Task]:
        if not self._running:
            self._running = True
            task = asyncio.create_task(self._monitor_queue(), name="retrainer")
            log.info("🧠 AI Evaluator Agent started — listening for closed trades.")
            return task
        return None

    async def _monitor_queue(self):
        while self._running:
            try:
                trade = await bot_state.closed_trades_queue.get()
                pnl = trade.get("pnl_pct", 0)
                reason = trade.get("reason", "manual")
                
                is_loss = pnl < 0 or reason == "SL"
                
                log.info(
                    "🧠 AI Evaluator analyzing %s trade: %s (Strategy: %s)",
                    "LOSING" if is_loss else "WINNING",
                    trade.get("symbol"), trade.get("strategy"),
                )
                
                if is_loss:
                    await self._analyze_loss(trade)
                else:
                    await self._analyze_win(trade)
                    
                bot_state.closed_trades_queue.task_done()
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
- Exit Price: {trade.get("exit")}
- P&L %: {trade.get("pnl_pct")}
- Reason: {trade.get("reason")}
- Opened At: {trade.get("opened_at")}
- Closed At: {trade.get("closed_at")}

## Current Strategy Rules
{rules_context}

## Your Task
Analyze this losing trade deeply. Identify why it failed and create a concrete rule to avoid this in the future.

Output ONLY valid JSON:
{{
  "cause": "<root cause>",
  "market_conditions": "<market state>",
  "what_went_wrong": "<setup failure>",
  "solution": "<how to avoid>",
  "new_rule": "<precise rule for next trade>",
  "severity": "<low|medium|high|critical>",
  "confidence": <0.0-1.0>
}}"""

        try:
            response = await self._client.aio.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self._system_instruction,
                    max_output_tokens=1000,
                )
            )
            raw = response.text.strip()
            if "```" in raw:
                start = raw.find("{")
                end = raw.rfind("}") + 1
                raw = raw[start:end]

            analysis = json.loads(raw)
            full_analysis = {
                "trade_id":     trade_id,
                "trade":        trade,
                "type":         "loss",
                "analysis":     analysis,
                "analyzed_at":  _now_iso(),
            }
            _save_analysis(trade_id, full_analysis)
            
            new_rule = analysis.get("new_rule")
            if new_rule:
                self._evolve_strategy(strategy, analysis, trade_id, is_loss=True)

            await bot_state.push_trade_analysis(full_analysis)
            log.info("✅ AI Evaluator: strategy '%s' evolved — new rule added", strategy)

        except Exception as e:
            log.error("AI Evaluator analysis failed for %s (%s): %s", symbol, strategy, e)

    async def _analyze_win(self, trade: Dict[str, Any]):
        """Study a winning trade to extract the 'Golden Setup' patterns."""
        symbol = trade.get("symbol", "UNKNOWN")
        timeframe = trade.get("timeframe", "15m")
        strategy = trade.get("strategy", "unknown")
        trade_id = _trade_id(trade)

        # Load current golden setups for context
        current_goldens = load_strategy_golden_setups(strategy)
        goldens_context = "\n".join(f"  - {g}" for g in current_goldens) if current_goldens else "  (none yet)"

        prompt = f"""You are an elite quantitative trading AI evaluator analyzing a highly successful trade to extract its 'Golden Setup' characteristics.

## Trade Details
- Symbol: {symbol}
- Strategy: {strategy}
- Timeframe: {timeframe}
- Entry Price: {trade.get("entry")}
- Exit Price: {trade.get("exit")}
- P&L %: {trade.get("pnl_pct")}
- Reason: {trade.get("reason")} (TP hit)

## Existing Golden Patterns
{goldens_context}

## Your Task
Analyze this winning trade. What were the key confluence factors? 
Identify the 'Golden Pattern' — the specific combination of indicators, market regime, or price action that made this trade succeed. This will be used to reinforce high-conviction entries.

Output ONLY valid JSON:
{{
  "confluence_factors": ["list", "of", "key", "factors"],
  "market_regime_context": "<description of market state that favored this>",
  "why_it_worked": "<explanation of success>",
  "positive_pattern": "<a precise, machine-readable pattern to look for — e.g. 'Enter {strategy} when RSI < 40 and volume is 2x average on 5m bounce'>",
  "quality_score": <0.0-1.0 setup quality>,
  "confidence": <0.0-1.0 analysis confidence>
}}"""

        try:
            response = await self._client.aio.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self._system_instruction,
                    max_output_tokens=1000,
                )
            )
            raw = response.text.strip()
            if "```" in raw:
                start = raw.find("{")
                end = raw.rfind("}") + 1
                raw = raw[start:end]

            analysis = json.loads(raw)
            full_analysis = {
                "trade_id":     trade_id,
                "trade":        trade,
                "type":         "win",
                "analysis":     analysis,
                "analyzed_at":  _now_iso(),
            }
            _save_analysis(trade_id, full_analysis)
            
            pattern = analysis.get("positive_pattern")
            if pattern:
                self._evolve_strategy(strategy, analysis, trade_id, is_loss=False)

            await bot_state.push_trade_analysis(full_analysis)
            log.info("🌟 AI Evaluator: strategy '%s' reinforced — new golden pattern extracted", strategy)

        except Exception as e:
            log.error("AI Evaluator win analysis failed for %s (%s): %s", symbol, strategy, e)

    def _evolve_strategy(self, strategy: str, analysis: Dict, trade_id: str, is_loss: bool = True):
        """Append a new rule or golden pattern to the strategy file."""
        data = _load_strategy_file(strategy)
        
        if is_loss:
            new_rule = analysis.get("new_rule")
            if not new_rule: return
            
            rule_entry = {
                "rule":       new_rule,
                "severity":   analysis.get("severity", "medium"),
                "confidence": analysis.get("confidence", 0.5),
                "cause":      analysis.get("cause", ""),
                "added_at":   _now_iso(),
                "trade_id":   trade_id,
            }
            data["rules"].append(rule_entry)
            if len(data["rules"]) > 20: data["rules"] = data["rules"][-20:]
            data["total_losses_analyzed"] = data.get("total_losses_analyzed", 0) + 1
        else:
            pattern = analysis.get("positive_pattern")
            if not pattern: return
            
            pattern_entry = {
                "pattern":    pattern,
                "quality":    analysis.get("quality_score", 0.7),
                "factors":    analysis.get("confluence_factors", []),
                "added_at":   _now_iso(),
                "trade_id":   trade_id,
            }
            if "golden_setups" not in data: data["golden_setups"] = []
            data["golden_setups"].append(pattern_entry)
            if len(data["golden_setups"]) > 10: data["golden_setups"] = data["golden_setups"][-10:]
            data["total_wins_analyzed"] = data.get("total_wins_analyzed", 0) + 1
        data["last_updated"] = _now_iso()

        # ── Evolution log ─────────────────────────────────────────────────────
        log_entry = {
            "action":     "rule_added" if is_loss else "pattern_added",
            "content":    new_rule if is_loss else pattern,
            "trade_id":   trade_id,
            "timestamp":  _now_iso(),
        }
        data["evolution_log"].append(log_entry)
        if len(data["evolution_log"]) > 50:
            data["evolution_log"] = data["evolution_log"][-50:]

        _save_strategy_file(strategy, data)
        log.info("📝 Strategy '%s' file updated — %d active rules", strategy, len(data["rules"]))

    # ── Legacy compatibility (class-level access) ─────────────────────────────
    @classmethod
    def load_lessons(cls, strategy: str) -> List[str]:
        """Backward-compatible: returns rule strings for a strategy."""
        return load_strategy_rules(strategy)
