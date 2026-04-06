"""
agents/warden.py  (v2)
─────────────────────────────────────────────────────────────────────────────
The Warden — Risk & Trade Management.

Fixes applied here
──────────────────
FIX #4  — ATR-based stop loss: SL = entry − ATR × ATR_MULTIPLIER
FIX #6  — Portfolio daily loss circuit breaker: halt all trading at −3 % P&L
FIX #8  — Structure-based break-even: trail SL below the most recent
           post-entry pivot low (instead of a fixed profit-% trigger)

Architecture
────────────
• Position      — dataclass tracking entry/SL/TP/qty/status
• DailyLossTracker — stateful daily P&L accumulator + circuit-breaker flag
• calculate_stop_loss_atr   — FIX #4 ATR-based SL
• calculate_stop_loss       — legacy fixed-% SL (kept as fallback)
• calculate_take_profit     — SL-distance × RR-ratio
• calculate_position_size   — fixed-fractional sizing
• WardenAgent               — open/close/monitor positions
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

import config
from agents.mathematician import find_pivot_lows
from utils.indicators import calc_atr
from web.state import bot_state

log = logging.getLogger(__name__)


# ── Position dataclass ────────────────────────────────────────────────────────

@dataclass
class Position:
    symbol:       str
    entry_price:  float
    stop_loss:    float
    take_profit:  float
    quantity:     float          # base asset quantity
    strategy:     str = "bounce"
    leverage:     int = 1
    is_futures:   bool = False
    entry_ts:     float = field(default_factory=time.time)  # seconds epoch
    be_activated: bool  = False  # True once SL has passed entry price
    # Tracks the highest close seen since entry — used for ATR trailing SL
    highest_close_since_entry: float = 0.0

    @property
    def sl_pct(self) -> float:
        return abs(self.entry_price - self.stop_loss) / self.entry_price * 100

    @property
    def tp_pct(self) -> float:
        return abs(self.take_profit - self.entry_price) / self.entry_price * 100


# ── FIX #6 — Daily loss circuit breaker ──────────────────────────────────────

class DailyLossTracker:
    """
    Accumulates realised USDT P&L for the current UTC day.
    If the day's loss exceeds MAX_DAILY_LOSS_PCT of the starting balance,
    `is_circuit_breaker_hit()` returns True and no new trades are opened.
    Resets automatically at UTC midnight.
    """

    def __init__(self) -> None:
        self._date          = self._today()
        self._start_balance = 0.0
        self._realized_pnl  = 0.0   # positive = profit, negative = loss

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _roll_if_new_day(self) -> None:
        today = self._today()
        if today != self._date:
            self._date         = today
            self._realized_pnl = 0.0
            # start_balance carries over — Executioner calls set_start_balance
            # on each trade attempt, so it will be refreshed automatically

    def set_start_balance(self, balance: float) -> None:
        """Call at bot start-up and before each trade to seed the baseline."""
        self._roll_if_new_day()
        if self._start_balance <= 0:
            self._start_balance = balance
            log.info("Daily loss tracker seeded: balance=%.2f USDT", balance)

    def record_trade_pnl(self, pnl_usdt: float) -> None:
        """Add realised trade P&L (positive or negative) to the daily total."""
        self._roll_if_new_day()
        self._realized_pnl += pnl_usdt
        log.info(
            "📊 Daily P&L updated: %+.2f USDT today | total=%+.2f USDT "
            "(%.2f%% of starting balance)",
            pnl_usdt,
            self._realized_pnl,
            self.daily_pnl_pct * 100,
        )

    @property
    def daily_pnl_pct(self) -> float:
        if self._start_balance <= 0:
            return 0.0
        return self._realized_pnl / self._start_balance

    def is_circuit_breaker_hit(self) -> bool:
        """True when today's loss ≥ MAX_DAILY_LOSS_PCT of starting balance."""
        return self.daily_pnl_pct <= -config.MAX_DAILY_LOSS_PCT

    @property
    def summary(self) -> str:
        return (
            f"Date={self._date} | "
            f"P&L={self._realized_pnl:+.2f} USDT "
            f"({self.daily_pnl_pct*100:+.2f}%) | "
            f"CB={'🚨 HIT' if self.is_circuit_breaker_hit() else '✅ OK'}"
        )


# ── Risk calculations ─────────────────────────────────────────────────────────

def calculate_stop_loss_atr(
    entry_price: float,
    atr:         float,
    multiplier:  float = None,
) -> float:
    """
    FIX #4 — ATR-based stop loss.
    SL = entry_price − (ATR × ATR_MULTIPLIER)

    Tighter than a fixed % when volatility is low; wider when volatile.
    This keeps the SL proportional to current market noise.
    """
    multiplier = multiplier if multiplier is not None else config.ATR_MULTIPLIER
    return entry_price - atr * multiplier


def calculate_stop_loss(
    entry_price:             float,
    confirmation_candle_low: float,
    max_sl_pct:              float = 0.10,
) -> float:
    """
    Legacy fixed-% SL — used as fallback when ATR unavailable.
    Returns the TIGHTER (closest to entry) of:
      • confirmation candle low
      • entry × (1 − max_sl_pct)
    """
    hard_floor = entry_price * (1.0 - max_sl_pct)
    return max(confirmation_candle_low, hard_floor)


def calculate_take_profit(
    entry_price: float,
    stop_loss:   float,
    rr_ratio:    float = None,
) -> float:
    """
    Fixed RR take-profit: entry + SL_distance × rr_ratio.
    Used as fallback when no swing-high TP target is available.
    Default RR = config.MIN_RR_FALLBACK (1.5).
    """
    rr_ratio = rr_ratio if rr_ratio is not None else config.MIN_RR_FALLBACK
    sl_dist  = abs(entry_price - stop_loss)
    return entry_price + sl_dist * rr_ratio


def calculate_position_size(
    balance_usdt: float,
    entry_price:  float,
    stop_loss:    float,
    risk_pct:     float = None,
    fixed_stake:  float = None,
) -> float:
    """
    Capital-based position sizing dynamically driven by the UI dashboard state, 
    so the bot trades a percentage of the total capital.
    """
    if entry_price <= 0:
        log.warning("Entry price is zero — cannot size position")
        return 0.0

    if fixed_stake is not None:
        capital_to_deploy = fixed_stake
    else:
        alloc_pct = bot_state.trade_allocation_pct / 100.0
        capital_to_deploy = balance_usdt * alloc_pct
    
    return capital_to_deploy / entry_price


# ── Warden Agent ──────────────────────────────────────────────────────────────

class WardenAgent:
    """
    Maintains open positions and enforces SL / TP / trailing-SL rules.

    New in v2
    ─────────
    • FIX #4  — SL set via ATR; Executioner passes pre-computed ATR-based SL.
    • FIX #6  — Circuit breaker: `is_circuit_breaker_hit()` checked by Executioner
                before every trade attempt.
    • FIX #8  — Structure BE: `check_position()` now accepts `candles` and
                finds post-entry pivot lows to trail the SL under structure.

    Workflow
    ────────
    Executioner calls `open_position()` after a confirmed limit-order fill.
    On each candle close it calls `check_position()`.
    Warden returns "SL", "TP", "BE", or None.
    """

    def __init__(self, notifier=None) -> None:
        self._positions: Dict[str, Position] = {}
        self._cooldowns: Dict[str, float]    = {}
        self._notifier                       = notifier
        self.daily_loss                      = DailyLossTracker()

    # ── Circuit breaker ───────────────────────────────────────────────────────
    def is_circuit_breaker_hit(self) -> bool:
        """FIX #6 — Returns True when daily loss limit is exceeded."""
        if self.daily_loss.is_circuit_breaker_hit():
            log.warning("🚨 Circuit breaker ACTIVE — %s", self.daily_loss.summary)
            return True
        return False

    # ── Cooldown ──────────────────────────────────────────────────────────────
    def is_on_cooldown(self, symbol: str) -> bool:
        return time.time() < self._cooldowns.get(symbol, 0)

    def set_cooldown(self, symbol: str) -> None:
        self._cooldowns[symbol] = time.time() + config.COOLDOWN_SECONDS
        log.info(
            "🛑 Cooldown set: %s — no re-trading for %.1f hours",
            symbol, config.COOLDOWN_SECONDS / 3600,
        )

    # ── Open position ─────────────────────────────────────────────────────────
    def open_position(
        self,
        symbol:      str,
        entry_price: float,
        stop_loss:   float,
        take_profit: float,
        quantity:    float,
        strategy:    str = "bounce",
        leverage:    int = 1,
        is_futures:  bool = False,
    ) -> Position:
        pos = Position(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=quantity,
            strategy=strategy,
            leverage=leverage,
            is_futures=is_futures,
            highest_close_since_entry=entry_price,
        )
        self._positions[symbol] = pos
        log.info(
            "📂 Position opened: %s | entry=%.6g | SL=%.6g (%.2f%%) | "
            "TP=%.6g (%.2f%%) | qty=%.6g | leverage=%dx",
            symbol, entry_price,
            stop_loss, pos.sl_pct,
            take_profit, pos.tp_pct,
            quantity, leverage,
        )
        asyncio.create_task(bot_state.push_position_opened(
            symbol=symbol,
            entry=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=quantity,
            strategy=strategy,
            leverage=leverage,
            is_futures=is_futures,
        ))
        return pos

    def close_position(self, symbol: str) -> Optional[Position]:
        return self._positions.pop(symbol, None)

    def get_position(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    # ── Structure-based SL trailing (FIX #8) ─────────────────────────────────
    def _find_structure_sl(
        self,
        pos:     Position,
        candles: List[Dict],
    ) -> Optional[float]:
        """
        FIX #8 — Find the most recent pivot low formed AFTER the entry
        timestamp that is also ABOVE the current stop loss.

        If found, return `pivot_low × (1 − 0.1%)` as the new, tighter SL.
        This trails the stop under real market structure rather than using a
        fixed profit-percentage trigger.

        Returns None if no qualifying post-entry pivot low exists yet.
        """
        entry_ts_ms = int(pos.entry_ts * 1000)  # convert seconds → ms
        post_entry  = [c for c in candles if c["timestamp"] > entry_ts_ms]

        # Need enough candles to form pivot lows on both sides
        n = max(2, min(config.PIVOT_N, len(post_entry) // 3))
        if len(post_entry) < 2 * n + 1:
            return None

        pivots = find_pivot_lows(post_entry, n=n)
        if not pivots:
            return None

        # Keep only pivots that are strictly above the current SL
        # (a pivot below current SL offers no improvement)
        candidates = [p for p in pivots if p.low > pos.stop_loss]
        if not candidates:
            return None

        # Take the most recent valid pivot low
        best = max(candidates, key=lambda p: p.index)

        # Apply a 0.1% cushion below the structure level
        return best.low * (1.0 - 0.001)

    # ── Position check on each candle close ──────────────────────────────────
    async def check_position(
        self,
        symbol:       str,
        latest_close: float,
        latest_high:  float,
        latest_low:   float,
        candles:      Optional[List] = None,   # FIX #8: for structure BE
    ) -> Optional[str]:
        """
        Evaluate the open position for `symbol` against the latest closed candle.

        Returns
        -------
        "SL"  — stop-loss hit  (caller must execute sell)
        "TP"  — take-profit hit (caller must execute sell)
        "BE"  — SL trailed under structure (no close, Warden updated internally)
        None  — no action
        """
        pos = self._positions.get(symbol)
        if pos is None:
            return None

        # ── Stop Loss (FIX #4: ATR-based SL was set at entry) ────────────────
        if latest_low <= pos.stop_loss:
            # Realised P&L (negative for a loss)
            pnl_usdt = (pos.stop_loss - pos.entry_price) * pos.quantity
            self.daily_loss.record_trade_pnl(pnl_usdt)

            log.warning(
                "❌ SL HIT: %s low=%.6g ≤ sl=%.6g | pnl=%+.2f USDT | %s",
                symbol, latest_low, pos.stop_loss, pnl_usdt,
                self.daily_loss.summary,
            )

            asyncio.create_task(bot_state.push_position_closed(
                symbol=symbol, exit_price=pos.stop_loss, reason="SL", strategy=pos.strategy
            ))
            self.close_position(symbol)
            self.set_cooldown(symbol)

            # FIX #6 — log circuit breaker state after every loss
            if self.daily_loss.is_circuit_breaker_hit():
                log.warning(
                    "🚨 CIRCUIT BREAKER HIT — halting all new trades today. %s",
                    self.daily_loss.summary,
                )

            return "SL"

        # ── Take Profit ──────────────────────────────────────────────────────
        if latest_high >= pos.take_profit:
            pnl_usdt = (pos.take_profit - pos.entry_price) * pos.quantity
            self.daily_loss.record_trade_pnl(pnl_usdt)

            log.info(
                "✅ TP HIT: %s high=%.6g ≥ tp=%.6g | pnl=%+.2f USDT | %s",
                symbol, latest_high, pos.take_profit, pnl_usdt,
                self.daily_loss.summary,
            )

            asyncio.create_task(bot_state.push_position_closed(
                symbol=symbol, exit_price=pos.take_profit, reason="TP", strategy=pos.strategy
            ))
            self.close_position(symbol)
            return "TP"

        # ── Trailing Stop Loss: ATR-based + Structure (unified) ──────────────
        # Runs on every candle — never stops, even after break-even is reached.
        # Candidate 1: ATR trailing  →  highest_close - ATR × multiplier
        # Candidate 2: Structure     →  most recent post-entry pivot low
        # Final SL = max(both candidates, current SL)  — only ever moves up.
        if candles:
            best_trail = pos.stop_loss  # never allow SL to go backward

            # ── ATR trail ────────────────────────────────────────────────────
            if len(candles) >= config.ATR_PERIOD + 2:
                try:
                    df_trail = pd.DataFrame(
                        candles[-50:],
                        columns=["timestamp", "open", "high", "low", "close", "volume"],
                    )
                    atr = calc_atr(df_trail)
                    if atr > 0:
                        # Keep a high-water mark of the close price since entry
                        pos.highest_close_since_entry = max(
                            pos.highest_close_since_entry, latest_close
                        )
                        atr_trail = pos.highest_close_since_entry - atr * config.ATR_MULTIPLIER
                        best_trail = max(best_trail, atr_trail)
                except Exception as _exc:
                    log.debug("ATR trail calc failed for %s: %s", symbol, _exc)

            # ── Structure trail ───────────────────────────────────────────────
            struct_sl = self._find_structure_sl(pos, candles)
            if struct_sl:
                best_trail = max(best_trail, struct_sl)

            # ── Apply if improved ─────────────────────────────────────────────
            if best_trail > pos.stop_loss:
                old_sl        = pos.stop_loss
                pos.stop_loss = best_trail

                if best_trail >= pos.entry_price:
                    pos.be_activated = True
                    label = "Break-Even ✅"
                else:
                    label = "TSL trailed"

                log.info(
                    "🔁 %s: %s | SL %.6g → %.6g (high=%.6g)",
                    label, symbol, old_sl, best_trail,
                    pos.highest_close_since_entry,
                )
                asyncio.create_task(bot_state.push_breakeven_activated(
                    symbol=symbol, new_sl=best_trail
                ))
                return "TSL"

        return None

    # ── Summary ───────────────────────────────────────────────────────────────
    def active_positions(self) -> List[Position]:
        return list(self._positions.values())

    def summary(self) -> Dict:
        return {
            "open_positions": len(self._positions),
            "symbols":        list(self._positions.keys()),
            "cooldowns": {
                sym: f"{(exp - time.time()) / 60:.1f}min remaining"
                for sym, exp in self._cooldowns.items()
                if time.time() < exp
            },
            "daily_loss": self.daily_loss.summary,
        }
