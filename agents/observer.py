"""
Observer Agent
Periodically evaluates bot performance and completely overrides memory variables in config
to maximize profit/10x capital based on real-time market regimes.
"""
import asyncio
import json
import logging
from typing import Optional

import config
from agents.ai_manager import AIManager
from web.state import bot_state

log = logging.getLogger(__name__)

_OBSERVER_SYSTEM_PROMPT = """You are the Supreme Overlord AI for a crypto futures high-frequency trading bot. 
Your singular goal is to 10x the user's capital as fast as mathematically possible while strictly avoiding liquidation or ruin.

You have permission to dynamically edit the bot's live Python configuration.
- P&L, Win Rate, and Market Trends dictate your behavior. 
- If losing or bleeding, tighten constraints (less concurrent pairs, wider stops, lower allocation, strict macro filters).
- If winning hard, compound aggressively (higher allocation up to 30%, more concurrent pairs, tight stops).

Respond ONLY with valid JSON matching this schema exactly:
{
    "new_config": {
        "MAX_SL_PCT": <float, e.g. 1.0 to 5.0>,
        "MIN_AI_CONFIDENCE": <float, e.g. 0.3 to 0.8>,
        "COOLDOWN_SECONDS": <int, e.g. 0 to 300>,
        "MAX_CONCURRENT_POSITIONS": <int, e.g. 2 to 10>
    },
    "global_directive": "<An overarching English instruction sent to the sub-AI that chooses entries right now. Be aggressive. e.g. 'Long everything on 5m pullbacks immediately.'>",
    "reasoning": "<Concise explanation of your macroeconomic analysis and risk adjustments>"
}
"""

class ObserverAgent:
    def __init__(self) -> None:
        self._running = False
        self._ai = AIManager()

    def start(self) -> Optional[asyncio.Task]:
        if not self._running:
            self._running = True
            # Audit after 30 seconds for immediate impact, then loop
            task = asyncio.create_task(self._loop(), name="observer_loop")
            log.info("👁️‍🗨️ Master Observer Agent started — taking control of bot settings.")
            return task
        return None

    async def _loop(self) -> None:
        try:
            await asyncio.sleep(10) # Quick initial audit
            while self._running:
                await self._audit()
                await asyncio.sleep(300) # Re-audit every 5 minutes
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error("Observer Agent loop error: %s", e)
            await asyncio.sleep(60)

    async def _audit(self) -> None:
        stats = bot_state.stats
        recent_trades = list(bot_state._trades)[:10]
        
        current_config = {
            "MAX_SL_PCT": config.MAX_SL_PCT,
            "MIN_AI_CONFIDENCE": config.MIN_AI_CONFIDENCE,
            "COOLDOWN_SECONDS": config.COOLDOWN_SECONDS,
            "MAX_CONCURRENT_POSITIONS": config.MAX_CONCURRENT_POSITIONS,
        }

        user_prompt = {
            "current_performance": stats,
            "recent_trades": recent_trades,
            "current_configuration": current_config
        }
        
        log.info("👁️‍🗨️ Master Observer auditing performance and risks...")
        response = await self._ai.evaluate_observer(
            system_prompt=_OBSERVER_SYSTEM_PROMPT,
            user_prompt=json.dumps(user_prompt, default=str)
        )
        
        if response and "new_config" in response:
            self._apply_overrides(response["new_config"])
            if "global_directive" in response:
                bot_state.observer_directive = response["global_directive"]
                log.info("👁️‍🗨️ Observer Directive: %s", bot_state.observer_directive)
            if "reasoning" in response:
                log.info("👁️‍🗨️ Observer Reasoning: %s", response["reasoning"])

    # Hard limits the Observer cannot exceed — prevents it from killing trade flow
    _OVERRIDE_LIMITS = {
        "MIN_AI_CONFIDENCE":       (0.55, 0.70),   # never above 0.70 — kills all signals
        "MAX_SL_PCT":              (0.5,  5.0),
        "COOLDOWN_SECONDS":        (60,   600),
        "MAX_CONCURRENT_POSITIONS":(2,    10),
    }

    def _apply_overrides(self, new_cfg: dict) -> None:
        for key, val in new_cfg.items():
            if not hasattr(config, key):
                continue
            # Clamp to allowed range if defined
            if key in self._OVERRIDE_LIMITS:
                lo, hi = self._OVERRIDE_LIMITS[key]
                val = max(lo, min(hi, val))
            setattr(config, key, val)
            log.info("👁️‍🗨️ Observer forced config.%s = %s", key, val)
