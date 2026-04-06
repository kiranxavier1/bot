"""
agents/mathematician.py  (v2)
─────────────────────────────────────────────────────────────────────────────
The Mathematician Agent — full rewrite incorporating all strategy fixes.

Fixes applied here
──────────────────
FIX #1  — Volume confirmation: bounce candle must exceed avg volume × 1.3
FIX #5  — Trendline slope validation: rejects near-flat and too-steep lines
FIX #10 — 3-touch trendline validation: trendline is only valid after 3+
           confirmed touches; the trade is taken on touch 4+
USER INSIGHT — Swing high targeting: detect previous pivot HIGH as TP target

Unchanged geometry
──────────────────
• Pivot low detection (N candles each side)
• Linear trendline through two highest-quality higher lows
• Watcher proximity arming at 0.5%
• 3rd-touch confirmation: low ≤ line AND close > line
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from utils.indicators import volume_ratio, calc_bollinger_bands, detect_fvg, market_regime

log = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class PivotLow:
    index:     int
    timestamp: int    # ms epoch
    low:       float


@dataclass
class PivotHigh:
    index:     int
    timestamp: int    # ms epoch
    high:      float


@dataclass
class Trendline:
    """y = m·x + b  (x in seconds, y in price)."""
    slope:     float
    intercept: float
    p1: PivotLow
    p2: PivotLow
    touch_count: int = 0    # validated touches (≥ 3 required before trading)

    def price_at(self, ts_ms: int) -> float:
        return self.slope * (ts_ms / 1000.0) + self.intercept

    @property
    def slope_per_candle_15m(self) -> float:
        """Slope expressed as price-fraction per 15-minute candle."""
        candle_seconds = 15 * 60
        base = self.p1.low if self.p1.low > 0 else 1.0
        return abs(self.slope * candle_seconds / base)


@dataclass
class WatcherState:
    symbol:    str
    timeframe: str
    trendline: Optional[Trendline] = None
    armed:     bool = False


# ── Pivot detection ───────────────────────────────────────────────────────────

def find_pivot_lows(candles: List[Dict], n: int = None) -> List[PivotLow]:
    n = n or config.PIVOT_N
    pivots: List[PivotLow] = []
    total = len(candles)
    for i in range(n, total - n):
        low_i = candles[i]["low"]
        left  = [candles[j]["low"] for j in range(i - n, i)]
        right = [candles[j]["low"] for j in range(i + 1, i + n + 1)]
        if all(low_i < lw for lw in left) and all(low_i < lw for lw in right):
            pivots.append(PivotLow(index=i, timestamp=candles[i]["timestamp"], low=low_i))
    return pivots


def find_pivot_highs(candles: List[Dict], n: int = None) -> List[PivotHigh]:
    """Mirror of find_pivot_lows — used to identify swing-high TP targets."""
    n = n or config.PIVOT_N
    pivots: List[PivotHigh] = []
    total = len(candles)
    for i in range(n, total - n):
        high_i = candles[i]["high"]
        left   = [candles[j]["high"] for j in range(i - n, i)]
        right  = [candles[j]["high"] for j in range(i + 1, i + n + 1)]
        if all(high_i > h for h in left) and all(high_i > h for h in right):
            pivots.append(PivotHigh(index=i, timestamp=candles[i]["timestamp"], high=high_i))
    return pivots


def find_higher_lows(pivots: List[PivotLow]) -> Optional[Tuple[PivotLow, PivotLow]]:
    if len(pivots) < 2:
        return None
    for j in range(len(pivots) - 1, 0, -1):
        for i in range(j - 1, -1, -1):
            if pivots[j].low > pivots[i].low:
                return pivots[i], pivots[j]
    return None


# ── Trendline ─────────────────────────────────────────────────────────────────

def compute_trendline(p1: PivotLow, p2: PivotLow) -> Trendline:
    x1, y1 = p1.timestamp / 1000.0, p1.low
    x2, y2 = p2.timestamp / 1000.0, p2.low
    if x2 == x1:
        raise ValueError("Pivots have identical timestamps")
    slope     = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return Trendline(slope=slope, intercept=intercept, p1=p1, p2=p2)


def validate_slope(trendline: Trendline) -> bool:
    """
    FIX #5 — Reject near-flat (noise) and too-steep (unsustainable) trendlines.
    Slope is expressed as fractional price change per 15m candle.
    """
    s = trendline.slope_per_candle_15m
    valid = config.TRENDLINE_MIN_SLOPE <= s <= config.TRENDLINE_MAX_SLOPE
    if not valid:
        log.debug(
            "Slope rejected: %.6f (bounds [%.6f, %.6f])",
            s, config.TRENDLINE_MIN_SLOPE, config.TRENDLINE_MAX_SLOPE,
        )
    return valid


def count_trendline_touches(
    candles: List[Dict],
    trendline: Trendline,
    tolerance: float = 0.005,
) -> int:
    """
    FIX #10 — Count how many candles touched the trendline within `tolerance`.
    A touch means: low ≤ line_price × (1 + tolerance)
    """
    count = 0
    for c in candles:
        line_price = trendline.price_at(c["timestamp"])
        if c["low"] <= line_price * (1 + tolerance):
            count += 1
    return count


# ── Proximity & signal ────────────────────────────────────────────────────────

def is_near_trendline(
    trendline: Trendline,
    current_price: float,
    current_ts: int,
    proximity: float = None,
) -> Tuple[bool, float]:
    proximity  = proximity if proximity is not None else config.WATCHER_PROXIMITY
    projected  = trendline.price_at(current_ts)
    if projected <= 0:
        return False, 0.0
    diff_pct = abs(current_price - projected) / projected
    return diff_pct <= proximity, diff_pct * 100.0


def check_third_touch(
    candles: List[Dict],
    trendline: Trendline,
    df: pd.DataFrame,
    lookback: int = 3,
) -> Optional[Dict]:
    """
    FIX #1 — Now also checks volume confirmation.
    A valid touch requires:
      • candle low ≤ trendline price  (touched/pierced)
      • candle close > trendline price  (bounced back)
      • candle volume ≥ avg volume × VOLUME_CONFIRM_MULTIPLIER
    """
    check_slice = candles[-lookback:]
    for candle in check_slice:
        ts         = candle["timestamp"]
        line_price = trendline.price_at(ts)
        low        = candle["low"]
        close      = candle["close"]

        if low <= line_price and close > line_price:
            # Volume check
            vol_ratio = volume_ratio(candle, df)
            if vol_ratio < config.VOLUME_CONFIRM_MULTIPLIER:
                log.debug(
                    "Touch found but volume too low: vol_ratio=%.2f < %.2f required",
                    vol_ratio, config.VOLUME_CONFIRM_MULTIPLIER,
                )
                continue   # FIX #1: skip low-volume touches
            log.info(
                "✅ Touch confirmed: low=%.6g ≤ line=%.6g < close=%.6g | vol_ratio=%.2f",
                low, line_price, close, vol_ratio,
            )
            return candle
    return None


# ── Swing high TP target (USER INSIGHT) ──────────────────────────────────────

def find_swing_tp_target(
    candles: List[Dict],
    entry_price: float,
    stop_loss: float,
) -> Optional[float]:
    """
    USER INSIGHT: use the previous swing high as TP target.

    Logic:
      1. Find all pivot highs in the candle history.
      2. Find the most recent pivot high that is ABOVE entry price
         (there's room for price to run to it).
      3. Target 0.5% below that high (SWING_TP_BUFFER) to sell into supply
         before the crowd does.
      4. Validate that the resulting R:R ≥ MIN_RR_FALLBACK (1.5).
         If not, return None and let the caller use a fixed-RR fallback.

    Why this works:
      Traders who bought at the previous high are now trapped (their cost
      basis is above current price during the trendline pullback).  When
      price approaches that level again they sell to "get even", creating
      a natural supply wall.  Targeting just below it maximises hit rate.
    """
    highs = find_pivot_highs(candles)
    if not highs:
        return None

    # Filter to highs above entry (need space to reach them)
    candidates = [h for h in highs if h.high > entry_price * 1.005]
    if not candidates:
        return None

    # Take the nearest (lowest) candidate — most conservative and highest probability
    nearest = min(candidates, key=lambda h: h.high)
    target  = nearest.high * (1 - config.SWING_TP_BUFFER)

    # Validate R:R
    sl_dist = entry_price - stop_loss
    tp_dist = target - entry_price
    if sl_dist <= 0 or tp_dist <= 0:
        return None

    rr = tp_dist / sl_dist
    if rr < config.MIN_RR_FALLBACK:
        log.debug(
            "Swing TP rejected: R:R=%.2f < %.2f minimum (target=%.6g)",
            rr, config.MIN_RR_FALLBACK, target,
        )
        return None

    log.info(
        "🎯 Swing TP target: %.6g (R:R=%.2f, previous high=%.6g)",
        target, rr, nearest.high,
    )
    return target


# ── Stateful Mathematician Agent ──────────────────────────────────────────────

class MathematicianAgent:
    """
    Stateful per-(symbol, timeframe) agent.  Called on every candle close.
    """

    def __init__(self) -> None:
        self._states: Dict[Tuple[str, str], WatcherState] = {}

    def _get_state(self, symbol: str, timeframe: str) -> WatcherState:
        key = (symbol, timeframe)
        if key not in self._states:
            self._states[key] = WatcherState(symbol=symbol, timeframe=timeframe)
        return self._states[key]

    async def process(
        self,
        symbol:    str,
        timeframe: str,
        candles:   List[Dict],
        df:        pd.DataFrame,
    ) -> Dict:
        """
        Main entry point — called after each candle close.

        Returns
        -------
        {
            "symbol", "timeframe",
            "trendline"           : Trendline | None,
            "trendline_price"     : float | None,
            "touch_count"         : int,
            "armed"               : bool,    # watcher proximity
            "proximity_pct"       : float,
            "signal"              : bool,    # confirmation candle found
            "confirmation_candle" : dict | None,
            "swing_tp_target"     : float | None,  # previous high TP target
            "vol_ratio"           : float,
        }
        """
        result: Dict = {
            "symbol":              symbol,
            "timeframe":           timeframe,
            "trendline":           None,
            "trendline_price":     None,
            "touch_count":         0,
            "armed":               False,
            "proximity_pct":       0.0,
            "signal":              False,
            "confirmation_candle": None,
            "swing_tp_target":     None,
            "vol_ratio":           1.0,
            "strategy_signals":    [],
        }

        if len(candles) < 2 * config.PIVOT_N + 5:
            return result

        state = self._get_state(symbol, timeframe)

        # ── Pivot lows ────────────────────────────────────────────────────────
        pivots = find_pivot_lows(candles)
        if len(pivots) < 2:
            return result

        pair = find_higher_lows(pivots)
        if pair is None:
            return result

        p1, p2 = pair

        # ── Trendline ─────────────────────────────────────────────────────────
        try:
            trendline = compute_trendline(p1, p2)
        except ValueError:
            return result

        # FIX #5 — slope validation
        if not validate_slope(trendline):
            return result

        # FIX #10 — count touches to validate the trendline
        touch_count = count_trendline_touches(candles[:-1], trendline)
        trendline.touch_count = touch_count
        result["touch_count"] = touch_count

        if touch_count < config.MIN_TRENDLINE_TOUCHES:
            log.debug(
                "%s [%s]: trendline only has %d touches (need %d)",
                symbol, timeframe, touch_count, config.MIN_TRENDLINE_TOUCHES,
            )
            return result   # Not yet validated — keep watching

        state.trendline = trendline
        result["trendline"] = trendline

        # ── Watcher proximity ─────────────────────────────────────────────────
        live_candle  = candles[-1]
        live_ts      = live_candle["timestamp"]
        live_close   = live_candle["close"]
        tl_price_now = trendline.price_at(live_ts)
        result["trendline_price"] = tl_price_now

        near, prox_pct = is_near_trendline(trendline, live_close, live_ts)
        result["proximity_pct"] = prox_pct

        if near and not state.armed:
            state.armed     = True
            result["armed"] = True
            log.info(
                "👁️  Watcher ARMED: %s [%s] price=%.6g ≈ line=%.6g (%.3f%%) touches=%d",
                symbol, timeframe, live_close, tl_price_now, prox_pct, touch_count,
            )

        if not near:
            state.armed = False

        # ── 3rd-touch confirmation (on closed candles, FIX #1 volume inside) ─
        if state.armed:
            closed = candles[:-1]
            conf   = check_third_touch(closed, trendline, df)
            if conf is not None:
                result["signal"]              = True
                result["confirmation_candle"] = conf
                result["vol_ratio"]           = volume_ratio(conf, df)
                state.armed = False

                # USER INSIGHT — swing high TP target
                rough_sl = tl_price_now * 0.985
                swing_tp = find_swing_tp_target(candles, conf["close"], rough_sl)
                result["swing_tp_target"] = swing_tp

                result["strategy_signals"].append({
                    "strategy": "bounce",
                    "entry_price": float(conf["close"]),
                    "stop_loss": float(tl_price_now * 0.998), # tight SL 0.2% below trendline break
                    "swing_tp_target": swing_tp,
                    "confirmation_candle": conf,
                })

                log.info(
                    "🎯 BOUNCE SIGNAL: %s [%s] close=%.6g | swing_tp=%s",
                    symbol, timeframe, conf["close"],
                    f"{swing_tp:.6g}" if swing_tp else "None (fallback RR)",
                )

        # ── Strategy 2: Mean Reversion (Bollinger Band Fade) ──────────────────
        # Logic: If market is ranging (ADX < 25), and candle closes back inside lower BB
        regime, adx, _, _ = market_regime(df)
        if adx < config.ADX_TREND_THRESHOLD:
            upper, mid, lower = calc_bollinger_bands(df)
            last_closed = candles[-2] if len(candles) > 1 else live_candle
            if last_closed["low"] < lower and last_closed["close"] > lower:
                # Strong rejection from the lower band in a ranging market
                result["strategy_signals"].append({
                    "strategy": "mean_reversion",
                    "entry_price": float(last_closed["close"]),
                    "stop_loss": float(last_closed["low"] * 0.998), # tight SL just below the wick
                    "take_profit": float(mid), # TP at the SMA (middle band)
                    "confirmation_candle": last_closed,
                    "adx": adx,
                })
                log.info("🎯 MEAN REVERSION SIGNAL: %s [%s] bounced off lower BB in ranging market (ADX=%.1f)", symbol, timeframe, adx)

        # ── Strategy 3: Fair Value Gap (SMC) ──────────────────────────────────
        # Logic: Bullish FVG exists, and price just dipped into it
        fvgs = detect_fvg(df)
        for fvg in fvgs:
            # If the current price is inside the FVG zone, that's a signal to buy the imbalance
            if fvg["bottom"] <= live_close <= fvg["top"]:
                result["strategy_signals"].append({
                    "strategy": "fvg",
                    "entry_price": float(live_close),
                    "stop_loss": float(fvg["bottom"] * 0.995), # SL just below the FVG
                    "take_profit": float(fvg["top"] + (fvg["top"] - fvg["bottom"]) * 2), # 1:2 RR baseline minimum
                    "confirmation_candle": live_candle,
                    "fvg": fvg,
                })
                log.info("🎯 FVG SIGNAL: %s [%s] dipped into active FVG zone. bottom=%.6g, top=%.6g", symbol, timeframe, fvg["bottom"], fvg["top"])
                break  # only trigger on the first matching FVG

        # ── Strategy 1: Breakout Momentum ─────────────────────────────────────
        # Logic: Current candle closes ABOVE recent major resistance with 2x+ volume
        recent_highs = find_pivot_highs(candles, n=15)
        if recent_highs:
            highest_resistance = max(h.high for h in recent_highs)
            last_closed = candles[-2] if len(candles) > 1 else live_candle
            vol_rat = volume_ratio(last_closed, df)
            
            if last_closed["close"] > highest_resistance and vol_rat >= 2.0:
                result["strategy_signals"].append({
                    "strategy": "breakout",
                    "entry_price": float(last_closed["close"]),
                    "stop_loss": float(highest_resistance * 0.99), # SL just below breakout line
                    "take_profit": float(last_closed["close"] + (last_closed["close"] - highest_resistance) * 2), # 1:2 RR minimum
                    "confirmation_candle": last_closed,
                })
                log.info("🎯 BREAKOUT SIGNAL: %s [%s] closed above resistance %.6g with vol_ratio=%.2f", symbol, timeframe, highest_resistance, vol_rat)

        return result
