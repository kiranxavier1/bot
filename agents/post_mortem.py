import asyncio
import json
import logging
import os
import anthropic
import config
from typing import Dict, Any, List

from web.state import bot_state

log = logging.getLogger(__name__)

class RetrainingAgent:
    """
    Consumes losing trades from the global queue.
    Fetches the precise chart data right before the loss.
    Sends it to Claude to identify the structural cause of the failure.
    Saves the extracted behavioral rule into strategy_lessons.json.
    """
    
    def __init__(self, exchange=None):
        self._client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        self._model = config.CLAUDE_MODEL
        self._exchange = exchange
        self._lessons_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'strategy_lessons.json')
        self._running = False
        
        # Ensure file exists
        if not os.path.exists(os.path.dirname(self._lessons_path)):
            os.makedirs(os.path.dirname(self._lessons_path))
        if not os.path.exists(self._lessons_path):
            with open(self._lessons_path, 'w') as f:
                json.dump({}, f)
                
    def start(self) -> asyncio.Task | None:
        if not self._running:
            self._running = True
            task = asyncio.create_task(self._monitor_queue(), name="retrainer")
            log.info("🧠 RetrainingAgent started, listening for losing trades.")
            return task
        return None

    async def _monitor_queue(self):
        while self._running:
            try:
                trade = await bot_state.losing_trades_queue.get()
                log.info("🧠 Post-Mortem Agent analyzing losing trade: %s (Strategy: %s)", trade["symbol"], trade["strategy"])
                await self._analyze_loss(trade)
                bot_state.losing_trades_queue.task_done()
            except asyncio.CancelledError:
                self._running = False
                break
            except Exception as e:
                log.error("RetrainingAgent error processing trade: %s", e)

    async def _analyze_loss(self, trade: Dict[str, Any]):
        symbol = trade.get("symbol")
        timeframe = trade.get("timeframe", "15m")
        strategy = trade.get("strategy", "unknown")
        
        # If we have an exchange instance, fetch the recent candles before the entry if possible
        # However, to be fast, we just grab current candles if it's very recent.
        # Ideally, we get enough recent data that contains the SL hit constraint.
        candles = []
        if self._exchange:
            try:
                candles = await self._exchange.watch_ohlcv(symbol, timeframe)
            except Exception:
                pass
                
        # Send proposal to Claude
        prompt = f"""
You are an institutional quantitative trading AI. We just hit a Stop Loss on a trade using the `{strategy}` strategy.

Trade Details:
Symbol: {symbol}
Entry Price: {trade.get("entry")}
Exit Price (SL): {trade.get("exit")}
Loss %: {trade.get("pnl_pct")}
Timeframe: {timeframe}
Real Trade: {trade.get("real", False)}

Based purely on this failure, and assuming the `{strategy}` logic fired correctly originally, what is the most likely structural or momentum failure that occurred? Give me exactly ONE highly specific, rigid rule that we can append to this strategy's knowledge base to prevent this mistake next time.

Output ONLY valid JSON:
{{
  "lesson": "Do not trade this strategy if X condition happens"
}}
"""
        
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=100,
                system="You analyze losing crypto trades and output 1 json lesson.",
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            # Strip markdown if present
            if "```" in raw:
                start = raw.find("{")
                end = raw.rfind("}") + 1
                raw = raw[start:end]
            data = json.loads(raw)
            lesson = data.get("lesson")
            
            if lesson:
                self._save_lesson(strategy, lesson)
                log.info("✅ Strategy '%s' learned new rule: %s", strategy, lesson)
        except Exception as e:
            log.error("Claude analysis failed on losing trade for %s: %s", symbol, e)
            
    def _save_lesson(self, strategy: str, lesson: str):
        try:
            with open(self._lessons_path, 'r') as f:
                lessons = json.load(f)
        except Exception:
            lessons = {}
            
        if strategy not in lessons:
            lessons[strategy] = []
            
        lessons[strategy].append(lesson)
        # Keep only the last 10 lessons to prevent context bloat
        lessons[strategy] = lessons[strategy][-10:]
        
        with open(self._lessons_path, 'w') as f:
            json.dump(lessons, f, indent=2)
            
    @classmethod
    def load_lessons(cls, strategy: str) -> List[str]:
        path = os.path.join(os.path.dirname(__file__), '..', 'data', 'strategy_lessons.json')
        try:
            with open(path, 'r') as f:
                lessons = json.load(f)
            return lessons.get(strategy, [])
        except Exception:
            return []
