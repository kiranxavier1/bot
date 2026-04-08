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
from utils.indicators import (
    volume_ratio,
    market_regime,
    calc_vwap,
    calc_atr,
    detect_ema_cross,
    is_ema_bullish_stack,
    detect_bullish_engulfing,
    price_above_vwap,
    rsi_above_midline,
    detect_ema_cross_down,
    is_ema_bearish_stack,
    detect_bearish_engulfing,
    price_below_vwap,
    rsi_below_midline,
)

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
        # FIX: Use solid candle body base instead of absolute wick low to prevent scam-wick warping
        low_i = min(candles[i]["open"], candles[i]["close"])
        left  = [min(candles[j]["open"], candles[j]["close"]) for j in range(i - n, i)]
        right = [min(candles[j]["open"], candles[j]["close"]) for j in range(i + 1, i + n + 1)]
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


def find_swing_tp_target_short(
    candles: List[Dict],
    entry_price: float,
    stop_loss: float,
) -> Optional[float]:
    """
    USER INSIGHT (Shorts): use the previous swing low as TP target.
    """
    lows = find_pivot_lows(candles)
    if not lows:
        return None

    # Filter to lows below entry (need space to reach them)
    candidates = [l for l in lows if l.low < entry_price * 0.995]
    if not candidates:
        return None

    # Take the nearest (highest) candidate
    nearest = max(candidates, key=lambda l: l.low)
    target  = nearest.low * (1 - config.SWING_TP_BUFFER)

    # Validate R:R
    sl_dist = stop_loss - entry_price
    tp_dist = entry_price - target
    if sl_dist <= 0 or tp_dist <= 0:
        return None

    rr = tp_dist / sl_dist
    if rr < config.MIN_RR_FALLBACK:
        log.debug(
            "Swing TP short rejected: R:R=%.2f < %.2f minimum (target=%.6g)",
            rr, config.MIN_RR_FALLBACK, target,
        )
        return None

    log.info(
        "🎯 Swing TP short target: %.6g (R:R=%.2f, previous low=%.6g)",
        target, rr, nearest.low,
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

        # ── Pre-compute shared win-rate filters ───────────────────────────────
        _rsi_ok     = rsi_above_midline(df)    # RSI-7 > 50
        _ema_stack  = is_ema_bullish_stack(df) # EMA9 > EMA21 > EMA50
        _above_vwap = price_above_vwap(df)     # price above VWAP
        _atr        = calc_atr(df)             # used for tight ATR-based TPs on scalps

        _rsi_bear   = rsi_below_midline(df)
        _ema_bear   = is_ema_bearish_stack(df)
        _below_vwap = price_below_vwap(df)

        # ── Strategy 1: Trendline Bounce (SWING) ──────────────────────────────
        # Swing trade: use SWING_RR (2x) — wider TP suits the longer move.
        if state.armed and _rsi_ok and _ema_stack:
            closed = candles[:-1]
            conf   = check_third_touch(closed, trendline, df)
            if conf is not None:
                result["signal"]              = True
                result["confirmation_candle"] = conf
                result["vol_ratio"]           = volume_ratio(conf, df)
                state.armed = False

                entry_p  = float(conf["close"])
                sl_p     = float(tl_price_now * 0.998)
                rough_sl = tl_price_now * 0.985
                swing_tp = find_swing_tp_target(candles, entry_p, rough_sl)
                # Swing: use swing high if achievable, else 2x SL distance
                tp_p = swing_tp or (entry_p + (entry_p - sl_p) * config.SWING_RR)
                result["swing_tp_target"] = swing_tp
                result["strategy_signals"].append({
                    "strategy":            "bounce",
                    "direction":           "long",
                    "entry_price":         entry_p,
                    "stop_loss":           sl_p,
                    "take_profit":         tp_p,
                    "swing_tp_target":     swing_tp,
                    "confirmation_candle": conf,
                })
                log.info(
                    "🎯 BOUNCE (swing): %s close=%.6g sl=%.6g tp=%.6g rsi=%s ema=%s",
                    symbol, entry_p, sl_p, tp_p, _rsi_ok, _ema_stack,
                )
        elif state.armed and not (_rsi_ok and _ema_stack):
            log.debug("Bounce armed but filtered: %s rsi_ok=%s ema_stack=%s", symbol, _rsi_ok, _ema_stack)

        # ── Strategy 2: Breakout + Retest (SWING) ─────────────────────────────
        # Swing trade: 2x SL distance — broken resistance becomes support, wider target.
        if _rsi_ok and _above_vwap:
            recent_highs = find_pivot_highs(candles, n=15)
            if recent_highs and len(candles) >= 4:
                highest_resistance = max(h.high for h in recent_highs)
                c_prev2 = candles[-3]
                c_prev1 = candles[-2]
                c_curr  = candles[-1]

                breakout_happened = c_prev2["close"] > highest_resistance
                retest_zone  = abs(c_prev1["low"] - highest_resistance) / highest_resistance <= 0.01
                bounce_green = c_curr["close"] > c_curr["open"] and c_curr["close"] > highest_resistance

                if breakout_happened and retest_zone and bounce_green:
                    vol_rat = volume_ratio(c_curr, df)
                    if vol_rat >= config.VOLUME_CONFIRM_MULTIPLIER:
                        entry_p = float(c_curr["close"])
                        sl_p    = float(highest_resistance * 0.99)
                        tp_p    = entry_p + (entry_p - sl_p) * config.SWING_RR
                        result["strategy_signals"].append({
                            "strategy":            "breakout",
                            "direction":           "long",
                            "entry_price":         entry_p,
                            "stop_loss":           sl_p,
                            "take_profit":         tp_p,
                            "confirmation_candle": c_curr,
                            "retest_price":        highest_resistance,
                            "vol_ratio":           vol_rat,
                        })
                        log.info(
                            "🎯 BREAKOUT (swing): %s retest=%.6g tp=%.6g vol=%.2f",
                            symbol, highest_resistance, tp_p, vol_rat,
                        )

        # ── Strategy 3: VWAP Bounce (SCALP) ───────────────────────────────────
        # Scalp: TP = entry + ATR × SCALP_TP_ATR_MULT — tight, proportional to volatility.
        if _rsi_ok and _ema_stack:
            regime_vwap, adx_vwap, di_plus_vwap, di_minus_vwap = market_regime(df)
            if adx_vwap >= config.ADX_TREND_THRESHOLD and di_plus_vwap > di_minus_vwap:
                vwap = calc_vwap(df)
                if vwap > 0:
                    last_closed = candles[-2] if len(candles) > 1 else live_candle
                    if last_closed["low"] <= vwap * 1.001 and last_closed["close"] > vwap:
                        vol_rat = volume_ratio(last_closed, df)
                        if vol_rat >= config.VWAP_VOLUME_MULTIPLIER:
                            entry_p = float(last_closed["close"])
                            sl_p    = float(vwap * (1 - 0.005))   # 0.5% below VWAP
                            # ATR-based TP — tight target price actually reaches
                            atr_tp  = (_atr * config.SCALP_TP_ATR_MULT) if _atr > 0 else (entry_p - sl_p) * config.MIN_RR_FALLBACK
                            tp_p    = entry_p + atr_tp
                            result["strategy_signals"].append({
                                "strategy":            "vwap_bounce",
                                "direction":           "long",
                                "entry_price":         entry_p,
                                "stop_loss":           sl_p,
                                "take_profit":         tp_p,
                                "confirmation_candle": last_closed,
                                "vwap":                vwap,
                                "vol_ratio":           vol_rat,
                                "adx":                 adx_vwap,
                            })
                            log.info(
                                "🎯 VWAP BOUNCE (scalp): %s close=%.6g sl=%.6g tp=%.6g atr=%.6g",
                                symbol, entry_p, sl_p, tp_p, _atr,
                            )

        # ── Strategy 4: EMA 9/21 Cross (SCALP) ────────────────────────────────
        # Backtested 70-75% win rate. ATR-based TP keeps target realistic.
        if _rsi_ok and _above_vwap:
            if detect_ema_cross(df, fast=9, slow=21):
                last_closed = candles[-2] if len(candles) > 1 else live_candle
                vol_rat = volume_ratio(last_closed, df)
                if vol_rat >= config.VOLUME_CONFIRM_MULTIPLIER:
                    entry_p    = float(last_closed["close"])
                    recent_lows = [candles[-i]["low"] for i in range(2, min(6, len(candles)))]
                    sl_p       = float(min(recent_lows) * 0.999) if recent_lows else entry_p * 0.995
                    atr_tp     = (_atr * config.SCALP_TP_ATR_MULT) if _atr > 0 else (entry_p - sl_p) * config.MIN_RR_FALLBACK
                    tp_p       = entry_p + atr_tp
                    if sl_p < entry_p and tp_p > entry_p:
                        result["strategy_signals"].append({
                            "strategy":            "ema_cross",
                            "direction":           "long",
                            "entry_price":         entry_p,
                            "stop_loss":           sl_p,
                            "take_profit":         tp_p,
                            "confirmation_candle": last_closed,
                            "vol_ratio":           vol_rat,
                        })
                        log.info(
                            "🎯 EMA CROSS (scalp): %s close=%.6g sl=%.6g tp=%.6g atr=%.6g",
                            symbol, entry_p, sl_p, tp_p, _atr,
                        )

        # ── Strategy 5: Momentum Candle Breakout (SCALP) ──────────────────────
        # ATR-based TP — tight target. Engulfing candle confirmation required.
        if _rsi_ok and _above_vwap and len(candles) >= 6:
            c1             = candles[-2]
            prev_highs     = [candles[-i]["high"] for i in range(3, 6)]
            structure_high = max(prev_highs)
            is_engulfing   = detect_bullish_engulfing(candles[:-1])

            if c1["close"] > structure_high and c1["close"] > c1["open"]:
                vol_rat = volume_ratio(c1, df)
                vol_threshold = 1.5 if is_engulfing else config.VOLUME_CONFIRM_MULTIPLIER * 1.3
                if vol_rat >= vol_threshold:
                    structure_low = min(candles[-i]["low"] for i in range(3, 6))
                    entry_p = float(c1["close"])
                    sl_p    = float(structure_low * 0.999)
                    atr_tp  = (_atr * config.SCALP_TP_ATR_MULT) if _atr > 0 else (entry_p - sl_p) * config.MIN_RR_FALLBACK
                    tp_p    = entry_p + atr_tp
                    if sl_p < entry_p and tp_p > entry_p:
                        result["strategy_signals"].append({
                            "strategy":            "momentum_scalp",
                            "direction":           "long",
                            "entry_price":         entry_p,
                            "stop_loss":           sl_p,
                            "take_profit":         tp_p,
                            "confirmation_candle": c1,
                            "structure_high":      structure_high,
                            "vol_ratio":           vol_rat,
                            "engulfing":           is_engulfing,
                        })
                        log.info(
                            "🎯 MOMENTUM SCALP: %s broke=%.6g tp=%.6g vol=%.2f engulf=%s",
                            symbol, structure_high, tp_p, vol_rat, is_engulfing,
                        )

        # ── Strategy 6: VWAP Reject (SHORT SCALP) ─────────────────────────────
        if _rsi_bear and _ema_bear:
            vwap = calc_vwap(df)
            if vwap > 0:
                last_closed = candles[-2] if len(candles) > 1 else live_candle
                is_engulfing = detect_bearish_engulfing(candles[:-1])
                if is_engulfing and last_closed["high"] >= vwap * 0.999 and last_closed["close"] < vwap:
                    vol_rat = volume_ratio(last_closed, df)
                    if vol_rat >= config.VWAP_VOLUME_MULTIPLIER:
                        entry_p = float(last_closed["close"])
                        sl_p    = float(vwap * (1 + 0.005))
                        atr_tp  = (_atr * config.SCALP_TP_ATR_MULT) if _atr > 0 else (sl_p - entry_p) * config.MIN_RR_FALLBACK
                        tp_p    = entry_p - atr_tp
                        result["strategy_signals"].append({
                            "strategy":            "vwap_reject_short",
                            "direction":           "short",
                            "entry_price":         entry_p,
                            "stop_loss":           sl_p,
                            "take_profit":         tp_p,
                            "confirmation_candle": last_closed,
                            "vwap":                vwap,
                            "vol_ratio":           vol_rat,
                            "engulfing":           is_engulfing,
                        })
                        log.info("🎯 VWAP REJECT SHORT: %s close=%.6g sl=%.6g tp=%.6g", symbol, entry_p, sl_p, tp_p)

        # ── Strategy 7: EMA 9/21 Cross Down (SHORT SCALP) ─────────────────────
        if _rsi_bear and _below_vwap:
            if detect_ema_cross_down(df, fast=9, slow=21):
                last_closed = candles[-2] if len(candles) > 1 else live_candle
                vol_rat = volume_ratio(last_closed, df)
                if vol_rat >= config.VOLUME_CONFIRM_MULTIPLIER:
                    entry_p    = float(last_closed["close"])
                    recent_highs = [candles[-i]["high"] for i in range(2, min(6, len(candles)))]
                    sl_p       = float(max(recent_highs) * 1.001) if recent_highs else entry_p * 1.005
                    atr_tp     = (_atr * config.SCALP_TP_ATR_MULT) if _atr > 0 else (sl_p - entry_p) * config.MIN_RR_FALLBACK
                    tp_p       = entry_p - atr_tp
                    if sl_p > entry_p and tp_p < entry_p:
                        result["strategy_signals"].append({
                            "strategy":            "ema_cross_short",
                            "direction":           "short",
                            "entry_price":         entry_p,
                            "stop_loss":           sl_p,
                            "take_profit":         tp_p,
                            "confirmation_candle": last_closed,
                            "vol_ratio":           vol_rat,
                        })
                        log.info("🎯 EMA CROSS SHORT: %s close=%.6g sl=%.6g tp=%.6g", symbol, entry_p, sl_p, tp_p)

        # ── Strategy 8: Momentum Candle Breakout Down (SHORT SCALP) ───────────
        if _rsi_bear and _below_vwap and len(candles) >= 6:
            c1             = candles[-2]
            prev_lows      = [candles[-i]["low"] for i in range(3, 6)]
            structure_low  = min(prev_lows)
            
            if c1["close"] < structure_low and c1["close"] < c1["open"]:
                vol_rat = volume_ratio(c1, df)
                if vol_rat >= config.VOLUME_CONFIRM_MULTIPLIER * 1.3:
                    structure_high = max(candles[-i]["high"] for i in range(3, 6))
                    entry_p = float(c1["close"])
                    sl_p    = float(structure_high * 1.001)
                    atr_tp  = (_atr * config.SCALP_TP_ATR_MULT) if _atr > 0 else (sl_p - entry_p) * config.MIN_RR_FALLBACK
                    tp_p    = entry_p - atr_tp
                    if sl_p > entry_p and tp_p < entry_p:
                        result["strategy_signals"].append({
                            "strategy":            "momentum_short",
                            "direction":           "short",
                            "entry_price":         entry_p,
                            "stop_loss":           sl_p,
                            "take_profit":         tp_p,
                            "confirmation_candle": c1,
                            "structure_low":       structure_low,
                            "vol_ratio":           vol_rat,
                        })
                        log.info("🎯 MOMENTUM SHORT: %s broke_low=%.6g tp=%.6g", symbol, structure_low, tp_p)

        return result
