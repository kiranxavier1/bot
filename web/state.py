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
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional

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

        # Scan log (newest first)
        self._scan_log: Deque[Dict] = deque(maxlen=60)
        self._last_scan_broadcast: float = 0.0

        # ── Simulation (paper trading) ────────────────────────────────────────
        self.sim_start_balance_gbp: float = 1000.0
        self.sim_balance_gbp: float = 1000.0   # running virtual GBP balance
        self.sim_pnl_gbp: float = 0.0          # cumulative P&L in GBP
        self._sim_open: Dict[str, Dict] = {}   # open sim positions keyed by symbol
        self._sim_trades: Deque[Dict] = deque(maxlen=100)  # closed sim trades

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

        # Subscribers — asyncio Queues that receive state snapshots on change
        self._subscribers: List[asyncio.Queue] = []

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
    ) -> None:
        async with self._lock:
            self.positions[symbol] = {
                "symbol":       symbol,
                "entry":        entry,
                "stop_loss":    stop_loss,
                "take_profit":  take_profit,
                "quantity":     quantity,
                "current_price": entry,
                "pnl_pct":      0.0,
                "opened_at":    _now_iso(),
                "be_activated": False,
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

    async def push_breakeven_activated(self, symbol: str, new_sl: float) -> None:
        async with self._lock:
            pos = self.positions.get(symbol)
            if pos:
                pos["stop_loss"]    = new_sl
                pos["be_activated"] = True
        await self._broadcast()

    async def push_position_closed(
        self,
        symbol:      str,
        exit_price:  float,
        reason:      str,   # "SL" | "TP" | "MANUAL"
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
                "opened_at":  pos.get("opened_at"),
                "closed_at":  _now_iso(),
            }
            self._trades.appendleft(trade)
            self._recompute_stats_locked()
        await self._broadcast()

    # ── AI decision events ────────────────────────────────────────────────────
    async def push_ai_decision(
        self,
        symbol:     str,
        timeframe:  str,
        decision:   str,    # "PROCEED" | "REJECT"
        confidence: float,
        reasoning:  str,
        risks:      List[str],
    ) -> None:
        async with self._lock:
            self._ai_log.appendleft({
                "symbol":     symbol,
                "timeframe":  timeframe,
                "decision":   decision,
                "confidence": round(confidence, 3),
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
        symbol:         str,
        timeframe:      str,
        price:          float,
        trendline_price: float,
        proximity_pct:  float,
    ) -> None:
        async with self._lock:
            self._alerts.appendleft({
                "symbol":          symbol,
                "timeframe":       timeframe,
                "price":           price,
                "trendline_price": trendline_price,
                "proximity_pct":   round(proximity_pct, 3),
                "timestamp":       _now_iso(),
            })
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

    # ── Simulation (paper trading) ────────────────────────────────────────────
    async def push_sim_entry(
        self,
        symbol:      str,
        timeframe:   str,
        entry_price: float,
        stop_loss:   float,
        take_profit: float,
    ) -> None:
        """Open a paper-trade position; stake = £50 or 5% of remaining balance."""
        async with self._lock:
            if symbol in self._sim_open:
                return  # already tracking this symbol
            stake = min(50.0, self.sim_balance_gbp * 0.05)
            if stake < 1.0:
                return  # no balance left
            self._sim_open[symbol] = {
                "symbol":       symbol,
                "timeframe":    timeframe,
                "entry":        entry_price,
                "stop_loss":    stop_loss,
                "take_profit":  take_profit,
                "stake_gbp":    round(stake, 2),
                "current":      entry_price,
                "pnl_pct":      0.0,
                "opened_at":    _now_iso(),
            }
        await self._broadcast()

    async def push_sim_price_update(self, symbol: str, price: float) -> None:
        """Update live price for an open sim position; auto-close if SL/TP hit."""
        closed_trade: Optional[Dict] = None
        async with self._lock:
            pos = self._sim_open.get(symbol)
            if pos is None:
                return
            entry = pos["entry"]
            pnl_pct = (price - entry) / entry * 100 if entry > 0 else 0.0
            pos["current"] = price
            pos["pnl_pct"] = round(pnl_pct, 3)

            hit_sl = price <= pos["stop_loss"]
            hit_tp = price >= pos["take_profit"]

            if hit_sl or hit_tp:
                reason = "TP" if hit_tp else "SL"
                stake  = pos["stake_gbp"]
                gbp_pnl = stake * pnl_pct / 100
                self.sim_pnl_gbp = round(self.sim_pnl_gbp + gbp_pnl, 2)
                self.sim_balance_gbp = round(self.sim_balance_gbp + gbp_pnl, 2)
                closed_trade = {
                    "symbol":     symbol,
                    "timeframe":  pos["timeframe"],
                    "entry":      entry,
                    "exit":       price,
                    "pnl_pct":    round(pnl_pct, 3),
                    "pnl_gbp":    round(gbp_pnl, 2),
                    "stake_gbp":  stake,
                    "reason":     reason,
                    "opened_at":  pos["opened_at"],
                    "closed_at":  _now_iso(),
                }
                self._sim_trades.appendleft(closed_trade)
                del self._sim_open[symbol]

        if closed_trade is not None:
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
                "scan_log":  list(self._scan_log),
                "stats":     dict(self.stats),
                "sim": {
                    "balance_gbp":    round(self.sim_balance_gbp, 2),
                    "pnl_gbp":        round(self.sim_pnl_gbp, 2),
                    "start_gbp":      self.sim_start_balance_gbp,
                    "open":           list(self._sim_open.values()),
                    "trades":         list(self._sim_trades),
                },
            }


# ── Global singleton ──────────────────────────────────────────────────────────
bot_state = BotState()
