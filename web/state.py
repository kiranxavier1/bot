"""
web/state.py
─────────────────────────────────────────────────────────────────────────────
Shared, thread-safe in-memory state store that bridges the trading bot
agents with the FastAPI web server.

The bot agents call the write methods (push_*) to record events.
The FastAPI server reads from this state and broadcasts it over WebSocket.

All timestamps are stored as ISO-8601 strings for easy JSON serialisation.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional

import config

# ── Max history sizes ─────────────────────────────────────────────────────────
MAX_TRADES     = 200
MAX_AI_LOG     = 100
MAX_ALERTS     = 50


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Singleton state object ────────────────────────────────────────────────────

class BotState:
    """
    Central mutable state for the dashboard.

    Thread-safety: all writes and reads are protected by asyncio.Lock so they
    can be safely called from different asyncio tasks (the bot and the web
    server share the same event loop in main.py).
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

        # Bot meta
        self.started_at: str = _now_iso()
        self.is_running:  bool = True
        self.symbols_tracked: int = 0

        # Open positions  {symbol: position_dict}
        self.positions: Dict[str, Dict] = {}

        # Closed trade history  (deque, newest first)
        self._trades: Deque[Dict] = deque(maxlen=MAX_TRADES)

        # AI decision log  (deque, newest first)
        self._ai_log: Deque[Dict] = deque(maxlen=MAX_AI_LOG)

        # Watcher alerts log  (deque, newest first)
        self._alerts: Deque[Dict] = deque(maxlen=MAX_ALERTS)

        # Analyzed trades log (deque, newest first)
        self._trade_analyses: Deque[Dict] = deque(maxlen=MAX_TRADES)

        # Scan log (newest first)
        self._scan_log: Deque[Dict] = deque(maxlen=60)
        self._last_scan_broadcast: float = 0.0

        # ── Live Trading ──────────────────────────────────────────────────────
        self.live_balance_usdt: float = 0.0
        self.live_pnl_usdt: float = 0.0
        self.trade_allocation_pct: float = config.TRADE_ALLOCATION_PCT  # % of balance per trade (set by dashboard slider)

        # Aggregate stats (recomputed on each trade close)
        self.stats: Dict[str, Any] = {
            "total_trades":   0,
            "wins":           0,
            "losses":         0,
            "win_rate":       0.0,
            "total_pnl_pct":  0.0,
            "ai_approved":    0,
            "ai_rejected":    0,
        }

        # Queue for Strategy Retraining Agent
        self.closed_trades_queue: asyncio.Queue = asyncio.Queue()

        # Subscribers — asyncio Queues that receive state snapshots on change
        self._subscribers: List[asyncio.Queue] = []

        # Load persisted state
        self._load_from_disk()

    # ── Subscription (WebSocket push) ────────────────────────────────────────
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=32)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    async def _broadcast(self) -> None:
        """Push a full snapshot to every subscribed WebSocket queue."""
        snapshot = await self.snapshot()
        for q in list(self._subscribers):
            try:
                q.put_nowait(snapshot)
            except asyncio.QueueFull:
                pass  # slow consumer — drop frame

    # ── Position events ───────────────────────────────────────────────────────
    async def push_position_opened(
        self,
        symbol:      str,
        entry:       float,
        stop_loss:   float,
        take_profit: float,
        quantity:    float,
        strategy:    str = "bounce",
        **kwargs,
    ) -> None:
        async with self._lock:
            self.positions[symbol] = {
                "symbol":       symbol,
                "strategy":     strategy,
                "entry":        entry,
                "stop_loss":    stop_loss,
                "take_profit":  take_profit,
                "quantity":     quantity,
                "leverage":      kwargs.get("leverage", 1),
                "is_futures":    kwargs.get("is_futures", config.USE_FUTURES),
                "current_price": entry,
                "pnl_pct":      0.0,
                "opened_at":    _now_iso(),
                "be_activated": False,
                "sl_order_id":  kwargs.get("sl_order_id"),
                "tp_order_id":  kwargs.get("tp_order_id"),
            }
        await self._broadcast()

    async def push_price_update(self, symbol: str, current_price: float) -> None:
        """Called periodically to keep live P&L accurate (optional)."""
        async with self._lock:
            pos = self.positions.get(symbol)
            if pos:
                pos["current_price"] = current_price
                if pos["entry"] > 0:
                    pos["pnl_pct"] = (current_price - pos["entry"]) / pos["entry"] * 100
        await self._broadcast()
        
    async def update_live_balance(self, balance_usdt: float) -> None:
        async with self._lock:
            self.live_balance_usdt = balance_usdt
        await self._broadcast()

    async def push_breakeven_activated(self, symbol: str, new_sl: float, sl_order_id: str = None) -> None:
        async with self._lock:
            pos = self.positions.get(symbol)
            if pos:
                pos["stop_loss"]    = new_sl
                pos["be_activated"] = True
                if sl_order_id:
                    pos["sl_order_id"] = sl_order_id
        await self._broadcast()

    async def push_position_closed(
        self,
        symbol:      str,
        exit_price:  float,
        reason:      str,   # "SL" | "TP" | "MANUAL"
        strategy:    str = "bounce",
    ) -> None:
        async with self._lock:
            pos = self.positions.pop(symbol, None)
            if pos is None:
                return
            entry = pos.get("entry", 0.0)
            pnl_pct = (exit_price - entry) / entry * 100 if entry > 0 else 0.0
            trade = {
                "symbol":     symbol,
                "entry":      entry,
                "exit":       exit_price,
                "pnl_pct":    round(pnl_pct, 3),
                "reason":     reason,
                "sl":         pos.get("stop_loss"),
                "tp":         pos.get("take_profit"),
                "quantity":   pos.get("quantity"),
                "strategy":   strategy,
                "opened_at":  pos.get("opened_at"),
                "closed_at":  _now_iso(),
                "real":       True,
            }
            self._trades.appendleft(trade)
            
            # Tally USDT P&L if quantity is available
            qty = pos.get("quantity")
            if qty and exit_price and entry:
                # Realised P&L calculation: (exit - entry) * base_qty
                # (For shorts, if any, it would be opposite. Assume long for now.)
                realised = (exit_price - entry) * qty
                self.live_pnl_usdt += realised

            self._recompute_stats_locked()
            self._save_to_disk()
            
            # Phase 3: Pipe all closed trades to Continuous Learning agent
            self.closed_trades_queue.put_nowait(trade)
                
        await self._broadcast()

    # ── AI decision events ────────────────────────────────────────────────────
    async def push_ai_decision(
        self,
        symbol:     str,
        timeframe:  str,
        decision:   str,    # "PROCEED" | "REJECT"
        confidence: float,
        leverage:   int = 1,
        reasoning:  str = "",
        risks:      List[str] = [],
        strategy:   str = "bounce",
    ) -> None:
        async with self._lock:
            self._ai_log.appendleft({
                "symbol":     symbol,
                "timeframe":  timeframe,
                "strategy":   strategy,
                "decision":   decision,
                "confidence": round(confidence, 3),
                "leverage":   leverage,
                "reasoning":  reasoning,
                "risks":      risks,
                "timestamp":  _now_iso(),
            })
            if decision == "PROCEED":
                self.stats["ai_approved"] += 1
            else:
                self.stats["ai_rejected"] += 1
        await self._broadcast()

    # ── Watcher alert events ──────────────────────────────────────────────────
    async def push_watcher_alert(
        self,
        symbol:          str,
        timeframe:       str,
        price:           float,
        strategy:        str   = "bounce",
        trendline_price: float = 0.0,
        proximity_pct:   float = 0.0,
        entry_price:     float = 0.0,
        stop_loss:       float = 0.0,
        take_profit:     float = 0.0,
        rr:              float = 0.0,
    ) -> None:
        async with self._lock:
            self._alerts.appendleft({
                "symbol":          symbol,
                "timeframe":       timeframe,
                "price":           price,
                "strategy":        strategy,
                "trendline_price": trendline_price,
                "proximity_pct":   round(proximity_pct, 3),
                "entry_price":     round(entry_price, 8) if entry_price else 0.0,
                "stop_loss":       round(stop_loss, 8)   if stop_loss   else 0.0,
                "take_profit":     round(take_profit, 8) if take_profit else 0.0,
                "rr":              round(rr, 2)           if rr          else 0.0,
                "timestamp":       _now_iso(),
            })
        await self._broadcast()

    # ── Trade analysis events ──────────────────────────────────────────────────
    async def push_trade_analysis(self, analysis: Dict[str, Any]) -> None:
        async with self._lock:
            self._trade_analyses.appendleft(analysis)
        await self._broadcast()

    # ── Scan log events ───────────────────────────────────────────────────────
    async def push_scan_event(self, symbol: str, timeframe: str, price: float) -> None:
        async with self._lock:
            self._scan_log.appendleft({
                "symbol": symbol,
                "timeframe": timeframe,
                "price": price,
                "timestamp": _now_iso(),
            })
        now = time.time()
        if now - self._last_scan_broadcast > 1.0:
            self._last_scan_broadcast = now
            await self._broadcast()


    # ── Bot meta ──────────────────────────────────────────────────────────────
    async def set_symbols_tracked(self, count: int) -> None:
        async with self._lock:
            self.symbols_tracked = count
        await self._broadcast()

    async def set_stopped(self) -> None:
        async with self._lock:
            self.is_running = False
        await self._broadcast()

    # ── Stats recompute (must be called inside lock) ──────────────────────────
    def _recompute_stats_locked(self) -> None:
        trades = list(self._trades)
        n      = len(trades)
        wins   = sum(1 for t in trades if t["pnl_pct"] >= 0)
        total_pnl = sum(t["pnl_pct"] for t in trades)
        self.stats["total_trades"]  = n
        self.stats["wins"]          = wins
        self.stats["losses"]        = n - wins
        self.stats["win_rate"]      = round(wins / n * 100, 1) if n > 0 else 0.0
        self.stats["total_pnl_pct"] = round(total_pnl, 3)

    # ── Snapshot for REST / WebSocket ─────────────────────────────────────────
    async def snapshot(self) -> Dict[str, Any]:
        async with self._lock:
            return {
                "meta": {
                    "started_at":       self.started_at,
                    "is_running":       self.is_running,
                    "symbols_tracked":  self.symbols_tracked,
                    "uptime_seconds":   int(
                        (datetime.now(timezone.utc) -
                         datetime.fromisoformat(self.started_at)).total_seconds()
                    ),
                },
                "positions": list(self.positions.values()),
                "trades":    list(self._trades),
                "ai_log":    list(self._ai_log),
                "alerts":    list(self._alerts),
                "trade_analyses": list(self._trade_analyses),
                "scan_log":  list(self._scan_log),
                "stats":     dict(self.stats),
                "sim": {
                    "balance_usdt":    round(self.live_balance_usdt, 2),
                    "pnl_usdt":        round(self.live_pnl_usdt, 2),
                    "trade_allocation_pct": self.trade_allocation_pct,
                    "open":           list(self.positions.values()),
                    "trades":         list(self._trades),
                },
            }

    # ── Persistence ───────────────────────────────────────────────────────────
    def _state_path(self) -> str:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_dir, "data", "state.json")

    def _save_to_disk(self) -> None:
        """Persist current sim and trade state to disk."""
        try:
            path = self._state_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            data = {
                "live_balance_usdt":     self.live_balance_usdt,
                "live_pnl_usdt":         self.live_pnl_usdt,
                "trade_allocation_pct":  self.trade_allocation_pct,
                "trades":                list(self._trades),
                "stats":                 self.stats,
            }
            
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            # We fail silently to avoid crashing the bot on disk errors
            print(f"Error saving state: {e}")

    def _load_from_disk(self) -> None:
        """Load sim and trade state from disk if available."""
        try:
            path = self._state_path()
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                
                self.live_balance_usdt     = data.get("live_balance_usdt", 0.0)
                self.live_pnl_usdt         = data.get("live_pnl_usdt", 0.0)
                self.trade_allocation_pct  = data.get("trade_allocation_pct", config.TRADE_ALLOCATION_PCT)
                
                # Reconstruct deques
                self._trades      = deque(data.get("trades", []),      maxlen=MAX_TRADES)
                self.stats        = data.get("stats", self.stats)
        except Exception as e:
            print(f"Error loading state: {e}")


# ── Global singleton ──────────────────────────────────────────────────────────
bot_state = BotState()
