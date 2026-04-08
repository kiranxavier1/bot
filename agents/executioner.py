"""
agents/executioner.py  (v2)
─────────────────────────────────────────────────────────────────────────────
The Disciplined Executioner — trade entry and position monitoring.

Fixes applied here
──────────────────
FIX #1  — Volume check now threaded through mathematician (handled there)
FIX #2  — 200-EMA higher-timeframe filter: coin must be above EMA-200 on 1h
FIX #3  — DI+ > DI- directional check: upward momentum must be confirmed
FIX #4  — ATR-based SL passed to Warden (computed via calc_atr)
FIX #6  — Circuit breaker checked before every new trade attempt
FIX #9  — Limit orders instead of market orders (placed at current ask price)
           Order tracked for fill; cancelled and skipped if not filled in time.
USER    — swing_tp_target from Mathematician used as primary TP; fixed-RR
           fallback when no valid swing high is available.

Flow (per candle close)
───────────────────────
1. Circuit-breaker guard (FIX #6)
2. Position monitoring if already in a trade (with candles for structure BE)
3. Cooldown guard
4. Retrieve candles + DataFrames
5. Mathematician analysis (now receives df — FIX #1 volume inside math)
6. Proximity watcher alert
7. No signal → return
8. HTF EMA-200 filter (FIX #2) — reject if coin below EMA-200 on 1h
9. ADX directional filter (FIX #3) — reject if DI- > DI+
10. Compute ATR-based SL (FIX #4)
11. Determine TP: swing high target (USER) or fixed-RR fallback
12. Build rich AI proposal
13. Claude gating
14. Execute limit buy (FIX #9)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import ccxt.pro as ccxtpro  # type: ignore
import pandas as pd

import config
from agents.mathematician import MathematicianAgent
from agents.ai_manager    import AIManager, build_proposal
from utils.indicators     import (
    calc_atr,
    market_regime,
    price_above_ema200,
    price_below_ema200,
    is_prime_session,
    is_weekday,
    btc_is_dropping,
    btc_is_rising,
    candle_close_strength,
    candle_close_weakness,
)
from agents.warden        import (
    WardenAgent,
    calculate_stop_loss_atr,
    calculate_stop_loss,
    calculate_take_profit,
    calculate_position_size,
    find_structural_sl_from_df,
    find_structural_sl_short_from_df,
)
from utils.notifications  import Notifier
from web.state            import bot_state

log = logging.getLogger(__name__)

# How long (seconds) to wait for a limit order to fill before cancelling
_ENTRY_ORDER_TIMEOUT = 90


class ExecutionerAgent:
    """
    Orchestrates the trade lifecycle for each (symbol, timeframe) pair.

    Parameters
    ----------
    exchange      : shared ccxt.pro exchange instance (REST + WS, authenticated)
    registry      : shared BufferRegistry
    mathematician : shared MathematicianAgent
    ai_manager    : shared AIManager
    warden        : shared WardenAgent
    notifier      : shared Notifier
    """

    def __init__(
        self,
        exchange:      ccxtpro.Exchange,
        registry:      BufferRegistry,
        mathematician: MathematicianAgent,
        ai_manager:    AIManager,
        warden:        WardenAgent,
        notifier:      Notifier,
    ) -> None:
        self._exchange = exchange
        self._registry = registry
        self._math     = mathematician
        self._ai       = ai_manager
        self._warden   = warden
        self._notifier = notifier

    # ── Main entry point ──────────────────────────────────────────────────────
    async def on_candle_close(
        self,
        symbol:        str,
        timeframe:     str,
        closed_candle: dict,
    ) -> None:
        """
        Dispatched per (symbol, timeframe) on each confirmed candle close.
        Only processes SIGNAL_TIMEFRAME candles for new entries; all timeframes
        are used for position monitoring.
        """
        # ── FIX #6 — Global circuit breaker guard ────────────────────────────
        if self._warden.is_circuit_breaker_hit():
            if self._warden.has_position(symbol):
                # Still monitor existing positions even when CB is active
                candles = await self._get_candles(symbol, timeframe)
                if candles:
                    await self._monitor_position(symbol, closed_candle, candles)
            return

        # ── Existing position monitoring ──────────────────────────────────────
        if self._warden.has_position(symbol):
            candles = await self._get_candles(symbol, timeframe)
            if candles:
                await self._monitor_position(symbol, closed_candle, candles)
            return

        # ── Only look for new entries on the signal timeframe (5m) ─────────────
        if timeframe != config.SIGNAL_TIMEFRAME:
            return

        # ── Session timing filter — active trading hours 06:00-20:00 UTC ───
        if not is_prime_session():
            log.debug("Outside prime session — skipping new entries for %s", symbol)
            return

        # ── Weekend filter — Sat/Sun have low volume & high false-positive rate ─
        if not is_weekday():
            log.debug("Weekend — skipping new entries for %s", symbol)
            return

        # ── Cooldown guard ────────────────────────────────────────────────────
        if self._warden.is_on_cooldown(symbol):
            log.debug("Skipping %s — on cooldown", symbol)
            return

        # ── Retrieve candles + DataFrames ─────────────────────────────────────
        candles = await self._get_candles(symbol, timeframe)
        if not candles or len(candles) < 2 * config.PIVOT_N + 5:
            return

        symbol_df = await self._get_df(symbol, timeframe)
        if symbol_df is None:
            return

        # ── Mathematician analysis (FIX #1: df passed for volume check) ───────
        math_result = await self._math.process(
            symbol, timeframe, candles, symbol_df
        )

        # Proximity watcher alert (trendline bounce only)
        if math_result.get("armed"):
            live_price = candles[-1]["close"]
            tl_price   = math_result.get("trendline_price", 0.0)
            prox_pct   = math_result.get("proximity_pct", 0.0)
            await self._notifier.send(
                Notifier.watcher_alert_msg(
                    symbol=symbol,
                    timeframe=timeframe,
                    current_price=live_price,
                    trendline_price=tl_price,
                    proximity_pct=prox_pct,
                )
            )
            asyncio.create_task(bot_state.push_watcher_alert(
                symbol=symbol,
                timeframe=timeframe,
                price=live_price,
                strategy="bounce",
                trendline_price=tl_price,
                proximity_pct=prox_pct,
            ))

        strategy_signals = math_result.get("strategy_signals", [])
        if not strategy_signals:
            return

        # Push every detected signal to the Live Opportunities panel immediately
        # so the user can see all strategies firing in real time, before AI filtering.
        live_price = candles[-1]["close"]
        for _sig in strategy_signals:
            _entry = _sig.get("entry_price", live_price)
            _sl    = _sig.get("stop_loss",   0.0)
            _tp    = _sig.get("take_profit",  0.0)
            _rr    = ((_tp - _entry) / (_entry - _sl)) if _sl and _sl < _entry and _tp > _entry else 0.0
            asyncio.create_task(bot_state.push_watcher_alert(
                symbol=symbol,
                timeframe=timeframe,
                price=live_price,
                strategy=_sig.get("strategy", "unknown"),
                entry_price=_entry,
                stop_loss=_sl,
                take_profit=_tp,
                rr=_rr,
            ))

        # ── Pre-compute common filters (15m HTF only — no 1h needed for scalping) ─
        htf_15m_df = await self._get_htf_df(symbol, "15m")

        # EMA-50 on 15m: coin must be above it to take long entries, below for shorts
        htf_above_ema = price_above_ema200(htf_15m_df) if htf_15m_df is not None else True
        htf_below_ema = price_below_ema200(htf_15m_df) if htf_15m_df is not None else True

        regime, adx, di_plus, di_minus = market_regime(symbol_df)
        atr = calc_atr(symbol_df)
        btc_df = await self._get_df(config.BTC_SYMBOL, config.BTC_TIMEFRAME)

        # ── BTC sharp-drop / rise macro filter ───────────────────────────────
        btc_dropping = btc_is_dropping(btc_df)
        btc_rising   = btc_is_rising(btc_df)

        valid_signals = []
        for sig in strategy_signals:
            strategy = sig["strategy"]
            direction = sig.get("direction", "long")
            log.info("🎯 Signal detected: %s [%s] strategy=%s dir=%s — running pre-trade filters...", symbol, timeframe, strategy, direction)

            if direction == "long":
                if btc_dropping:
                    log.info("❌ BTC drop filter rejected %s — paused long entries", symbol)
                    continue
                if not htf_above_ema:
                    log.info("❌ HTF filter rejected %s (%s) — price BELOW 15m EMA-50", symbol, strategy)
                    continue
                if config.ADX_REQUIRE_DIRECTION and di_minus >= di_plus:
                    log.info("❌ Direction rejected %s (%s) — DI- > DI+", symbol, strategy)
                    continue
            else:
                if btc_rising:
                    log.info("❌ BTC rise filter rejected %s — paused short entries", symbol)
                    continue
                if not htf_below_ema:
                    log.info("❌ HTF filter rejected %s (%s) — price ABOVE 15m EMA-50", symbol, strategy)
                    continue
                if config.ADX_REQUIRE_DIRECTION and di_plus >= di_minus:
                    log.info("❌ Direction rejected %s (%s) — DI+ > DI-", symbol, strategy)
                    continue

            # Trend strategies also need minimum ADX momentum
            if strategy in ("bounce", "breakout", "vwap_bounce", "momentum_scalp", "vwap_reject_short", "ema_cross_short", "momentum_short"):
                if adx < config.ADX_TREND_THRESHOLD:
                    log.info("❌ ADX filter rejected %s (%s) — ADX=%.1f too low", symbol, strategy, adx)
                    continue

            # Candle close strength (long) vs weakness (short)
            conf_candle = sig.get("confirmation_candle", {})
            if conf_candle:
                if direction == "long" and not candle_close_strength(conf_candle):
                    log.info("❌ Close-strength rejected %s (%s)", symbol, strategy)
                    continue
                if direction == "short" and not candle_close_weakness(conf_candle):
                    log.info("❌ Close-weakness rejected %s (%s)", symbol, strategy)
                    continue

            log.info("✅ Pre-trade filters passed for %s (strategy: %s, dir: %s)", symbol, strategy, direction)
            valid_signals.append(sig)

        if not valid_signals:
            return

        async def _eval_signal(sig):
            strategy = sig["strategy"]
            direction = sig.get("direction", "long")
            conf_candle = sig["confirmation_candle"]
            entry_est = sig["entry_price"]
            swing_tp_math = sig.get("swing_tp_target")
            trendline = math_result.get("trendline")
            trendline_slope = trendline.slope if trendline else 0.0
            touch_count = math_result.get("touch_count", 0)
            vol_ratio = math_result.get("vol_ratio", 1.0)
            prox_pct = math_result.get("proximity_pct", 0.0)

            raw_sl = sig.get("stop_loss")
            if not raw_sl:
                if atr > 0 and config.USE_STRUCTURAL_SL and htf_15m_df is not None:
                    if direction == "long":
                        raw_sl = find_structural_sl_from_df(htf_15m_df, entry_est, atr)
                    else:
                        raw_sl = find_structural_sl_short_from_df(htf_15m_df, entry_est, atr)
                    log.info("📐 Structural SL for %s: %.6g (15m pivot-based)", symbol, raw_sl)
                elif atr > 0:
                    if direction == "long":
                        raw_sl = calculate_stop_loss_atr(entry_est, atr)
                    else:
                        raw_sl = entry_est + atr * config.ATR_MULTIPLIER
                else:
                    if direction == "long":
                        raw_sl = calculate_stop_loss(entry_est, conf_candle["low"])
                    else:
                        raw_sl = entry_est * (1.0 + config.MAX_SL_PCT / 2)

            # Max SL% cap — skip trade if SL is too wide (volatile spike protection)
            if raw_sl and entry_est > 0:
                sl_pct = abs(entry_est - raw_sl) / entry_est * 100
                if sl_pct > config.MAX_SL_PCT:
                    log.info(
                        "❌ SL too wide for %s (%s) — SL=%.2f%% > MAX %.2f%%",
                        symbol, strategy, sl_pct, config.MAX_SL_PCT,
                    )
                    return

            raw_tp = sig.get("take_profit")
            tp_source = "strategy_default"
            if not raw_tp:
                if direction == "long":
                    if swing_tp_math and swing_tp_math > entry_est:
                        raw_tp = swing_tp_math
                        tp_source = "swing_high"
                    else:
                        raw_tp = calculate_take_profit(entry_est, raw_sl)
                        tp_source = "fixed_rr"
                else:
                    if swing_tp_math and swing_tp_math < entry_est:
                        raw_tp = swing_tp_math
                        tp_source = "swing_low"
                    else:
                        raw_tp = entry_est - abs(entry_est - raw_sl) * config.MIN_RR_FALLBACK
                        tp_source = "fixed_rr"

            proposal = build_proposal(
                strategy=strategy,
                symbol=symbol,
                timeframe=timeframe,
                entry_price=entry_est,
                stop_loss=raw_sl,
                take_profit=raw_tp,
                confirmation_candle=conf_candle,
                trendline_slope=trendline_slope,
                touch_proximity_pct=prox_pct,
                btc_df=btc_df,
                symbol_df=symbol_df,
                touch_count=touch_count,
                vol_ratio=vol_ratio,
                swing_tp_target=swing_tp_math,
                tp_source=tp_source,
                atr=atr,
                adx=adx,
                di_plus=di_plus,
                di_minus=di_minus,
                htf_above_ema=htf_above_ema,
            )

            decision = await self._ai.evaluate(proposal)
            return sig, decision, raw_sl, raw_tp, swing_tp_math, conf_candle

        eval_tasks = [_eval_signal(sig) for sig in valid_signals]
        results = await asyncio.gather(*eval_tasks, return_exceptions=True)

        approved = []
        for result in results:
            if isinstance(result, Exception):
                log.error("AI Evaluation failed with exception: %s", result)
                continue
            
            sig, decision, raw_sl, raw_tp, swing_tp_math, conf_candle = result
            strategy = sig["strategy"]
            if not self._ai.should_proceed(decision):
                log.info("🤖 AI rejected %s (%s) conf=%.2f — %s", symbol, strategy, decision.get("confidence", 0.0), decision.get("reasoning", "")[:100])
                # We optionally fire off a reject message here, but async so it doesn't block.
                asyncio.create_task(self._notifier.send(
                    Notifier.ai_rejected_msg(
                        symbol=symbol,
                        reason=decision.get("reasoning", "No reason given"),
                        confidence=decision.get("confidence", 0.0),
                    )
                ))
                continue
            
            approved.append((sig, decision, raw_sl, raw_tp, swing_tp_math, conf_candle))

        if not approved:
            return

        # Sort by confidence descending, pick the absolute best one
        approved.sort(key=lambda x: x[1].get("confidence", 0.0), reverse=True)
        best_sig, best_decision, best_sl, best_tp, best_swing_tp, best_conf_candle = approved[0]
        strategy = best_sig["strategy"]

        log.info("✅ AI approved best setup (%s) for %s — executing limit buy (conf: %.2f)", strategy, symbol, best_decision.get("confidence", 0.0))

        await self._execute_limit_buy(
            symbol=symbol,
            conf_candle=best_conf_candle,
            atr=atr,
            swing_tp=best_swing_tp,
            target_sl=best_sl,
            target_tp=best_tp,
            strategy=strategy,
            direction=best_sig.get("direction", "long"),
            leverage=best_decision.get("leverage", 1),
            confidence=best_decision.get("confidence", 0.0),
        )

    # ── FIX #9 — Limit buy with fill tracking ────────────────────────────────
    async def _execute_limit_buy(
        self,
        symbol:    str,
        conf_candle: dict,
        atr:       float,
        swing_tp:  Optional[float] = None,
        target_sl: Optional[float] = None,
        target_tp: Optional[float] = None,
        strategy:  str = "bounce",
        direction: str = "long",
        leverage:  int = 1,
        confidence: float = 0.0,
    ) -> None:
        """
        Place a limit buy at the current ask price (fills like a market order
        but with price protection against adverse fills).

        After placing, we poll for fill every ~5 s for up to
        _ENTRY_ORDER_TIMEOUT seconds.  If not filled in time, cancel and skip.
        """
        try:
            # ── Concurrent position guard ─────────────────────────────────────
            open_count = len(self._warden.active_positions())
            if open_count >= config.MAX_CONCURRENT_POSITIONS:
                log.info(
                    "⛔ Max concurrent positions (%d/%d) reached — skipping %s",
                    open_count, config.MAX_CONCURRENT_POSITIONS, symbol,
                )
                return

            # ── Fetch balance (FIX #6 seeds daily tracker) ───────────────────
            balance   = await self._exchange.fetch_balance()
            if config.USE_FUTURES:
                # Binance Futures balance is under assets list
                usdt_free = 0.0
                for asset in balance.get("info", {}).get("assets", []):
                    if asset.get("asset") == "USDT":
                        usdt_free = float(asset.get("availableBalance", 0.0))
                        break
                # Fallback to standard path if info not available
                if usdt_free == 0.0:
                    usdt_free = float(balance.get("USDT", {}).get("free", 0.0))
            else:
                usdt_free = float(balance.get("USDT", {}).get("free", 0.0))
            self._warden.daily_loss.set_start_balance(usdt_free)
            asyncio.create_task(bot_state.update_live_balance(usdt_free))

            if usdt_free < config.MIN_TRADE_BALANCE:
                log.warning(
                    "Insufficient USDT balance (%.2f < %.2f MIN_TRADE_BALANCE) — skipping buy for %s",
                    usdt_free, config.MIN_TRADE_BALANCE, symbol,
                )
                return

            # ── Get current market price ──────────────────────────────
            ticker      = await self._exchange.fetch_ticker(symbol)
            if direction == "long":
                market_price = float(ticker.get("ask") or ticker.get("last") or 0.0)
            else:
                market_price = float(ticker.get("bid") or ticker.get("last") or 0.0)
            
            if market_price <= 0:
                log.error("Invalid market price for %s — aborting", symbol)
                return

            # ── SL / TP Calculation ──────────────────────────────────────────
            if target_sl is not None:
                stop_loss = target_sl
            elif atr > 0:
                stop_loss = calculate_stop_loss_atr(market_price, atr) if direction == "long" else market_price + atr * config.ATR_MULTIPLIER
            else:
                stop_loss = calculate_stop_loss(market_price, conf_candle["low"]) if direction == "long" else market_price * (1.0 + config.MAX_SL_PCT / 2)

            if target_tp is not None:
                take_profit = target_tp
            elif direction == "long":
                if swing_tp and swing_tp > market_price:
                    take_profit = swing_tp
                else:
                    take_profit = calculate_take_profit(market_price, stop_loss)
            else:
                if swing_tp and swing_tp < market_price:
                    take_profit = swing_tp
                else:
                    take_profit = market_price - abs(market_price - stop_loss) * config.MIN_RR_FALLBACK
            
            # ── Futures Setup ────────────────────────────────────────────────
            if config.USE_FUTURES:
                try:
                    await self._exchange.set_leverage(leverage, symbol)
                    log.info("🎯 Leverage set to %dx for %s", leverage, symbol)
                except Exception as lev_exc:
                    log.warning("Failed to set leverage for %s: %s", symbol, lev_exc)

            # For Futures: leverage scales the notional exposure of our margin.
            # e.g. 5% of $200 balance = $10 margin × 5x leverage = $50 position.
            effective_leverage = leverage if config.USE_FUTURES else 1
            quantity = calculate_position_size(
                balance_usdt=usdt_free * effective_leverage,
                entry_price=market_price,
                stop_loss=stop_loss,
            )
            if quantity <= 0:
                log.warning("Zero quantity for %s — skipping", symbol)
                return

            # Round to exchange precision
            quantity = float(
                self._exchange.amount_to_precision(symbol, quantity)
            )
            # Add a 0.2% adaptive slippage buffer.
            if direction == "long":
                adaptive_limit = market_price * 1.002
            else:
                adaptive_limit = market_price * 0.998
                
            limit_price = float(
                self._exchange.price_to_precision(symbol, adaptive_limit)
            )

            log.info(
                "🛒 Placing Limit Buy: %s qty=%.6g @ %.6g | SL=%.6g | TP=%.6g",
                symbol, quantity, limit_price, stop_loss, take_profit,
            )

            # ── Place order ───────────────────────────────────────────────────
            entry_side = "buy" if direction == "long" else "sell"
            if config.USE_FUTURES:
                order = await self._exchange.create_order(
                    symbol, "limit", entry_side, quantity, limit_price,
                    params={"timeInForce": "GTC", "positionSide": "BOTH"},
                )
            else:
                if entry_side == "buy":
                    order = await self._exchange.create_limit_buy_order(symbol, quantity, limit_price)
                else:
                    order = await self._exchange.create_limit_sell_order(symbol, quantity, limit_price)
                    
            order_id = order["id"]
            log.info("Order placed: %s id=%s — waiting for fill...", symbol, order_id)

            # ── Poll for fill (FIX #9) ────────────────────────────────────────
            deadline = time.time() + _ENTRY_ORDER_TIMEOUT
            while time.time() < deadline:
                await asyncio.sleep(5)
                try:
                    order = await self._exchange.fetch_order(order_id, symbol)
                except Exception as fetch_exc:
                    log.warning(
                        "fetch_order failed for %s id=%s: %s",
                        symbol, order_id, fetch_exc,
                    )
                    continue

                status = order.get("status", "open")
                if status == "closed":
                    break
                if status == "canceled":
                    log.warning("Order %s was externally cancelled — aborting", order_id)
                    return

            if order.get("status") != "closed":
                # Timed out — cancel the open order and skip this signal
                try:
                    await self._exchange.cancel_order(order_id, symbol)
                    log.warning(
                        "Limit order timed out (%ds) and was cancelled: %s id=%s",
                        _ENTRY_ORDER_TIMEOUT, symbol, order_id,
                    )
                except Exception as cancel_exc:
                    log.error(
                        "Failed to cancel order %s for %s: %s",
                        order_id, symbol, cancel_exc,
                    )
                await self._notifier.send(
                    f"⏱️ <b>Limit order expired</b> for <code>{symbol}</code> — "
                    f"signal skipped (no fill within {_ENTRY_ORDER_TIMEOUT}s)."
                )
                return

            # ── Order filled ──────────────────────────────────────────────────
            filled_price = float(order.get("average") or order.get("price") or limit_price)
            filled_qty   = float(order.get("filled")  or quantity)

            log.info(
                "✅ Fill confirmed: %s | price=%.6g | qty=%.6g | orderId=%s",
                symbol, filled_price, filled_qty, order_id,
            )

            # Recalculate SL / TP on actual fill price
            if target_sl is not None:
                stop_loss = target_sl
            elif atr > 0:
                stop_loss = calculate_stop_loss_atr(filled_price, atr) if direction == "long" else filled_price + atr * config.ATR_MULTIPLIER
            else:
                stop_loss = calculate_stop_loss(filled_price, conf_candle["low"]) if direction == "long" else filled_price * (1.0 + config.MAX_SL_PCT / 2)

            if target_tp is not None:
                take_profit = target_tp
            elif direction == "long":
                if swing_tp and swing_tp > filled_price:
                    take_profit = swing_tp
                else:
                    take_profit = calculate_take_profit(filled_price, stop_loss)
            else:
                if swing_tp and swing_tp < filled_price:
                    take_profit = swing_tp
                else:
                    take_profit = filled_price - abs(filled_price - stop_loss) * config.MIN_RR_FALLBACK

            # ── Register with Warden ──────────────────────────────────────────
            pos = self._warden.open_position(
                symbol=symbol,
                entry_price=filled_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                quantity=float(order.get("filled", quantity)),
                strategy=strategy,
                leverage=leverage if config.USE_FUTURES else 1,
                is_futures=config.USE_FUTURES,
                direction=direction,
            )

            # ── FIX: Synchronize Live SL/TP orders on Binance ────────────────
            # This ensures exits happen even if the 5m candle hasn't closed yet.
            if config.USE_FUTURES:
                try:
                    sl_price = float(self._exchange.price_to_precision(symbol, stop_loss))
                    tp_price = float(self._exchange.price_to_precision(symbol, take_profit))
                    
                    close_side = "sell" if direction == "long" else "buy"

                    # 1. Stop Loss Order
                    log.info("🎯 Placing live STOP_MARKET on Binance at %s", sl_price)
                    sl_order = await self._exchange.create_order(
                        symbol, "STOP_MARKET", close_side, filled_qty,
                        params={
                            "stopPrice": sl_price,
                            "reduceOnly": True,
                            "positionSide": "BOTH"
                        }
                    )
                    pos.sl_order_id = sl_order.get("id")

                    # 2. Take Profit Order
                    log.info("🎯 Placing live TAKE_PROFIT_MARKET on Binance at %s", tp_price)
                    tp_order = await self._exchange.create_order(
                        symbol, "TAKE_PROFIT_MARKET", close_side, filled_qty,
                        params={
                            "stopPrice": tp_price,
                            "reduceOnly": True,
                            "positionSide": "BOTH"
                        }
                    )
                    pos.tp_order_id = tp_order.get("id")

                except Exception as ex_sync:
                    log.error("Failed to place exchange SL/TP for %s: %s", symbol, ex_sync)

            # ── Telegram notification ─────────────────────────────────────────
            await self._notifier.send(
                Notifier.trade_opened_msg(
                    symbol=symbol,
                    side="buy",
                    entry=filled_price,
                    sl=stop_loss,
                    tp=take_profit,
                    size=filled_qty,
                )
            )

        except Exception as exc:
            log.error("Limit buy failed for %s: %s", symbol, exc, exc_info=True)
            await self._notifier.send(
                Notifier.error_msg(f"Limit buy {symbol}", exc)
            )

    # ── Monitor existing position ─────────────────────────────────────────────
    async def _monitor_position(
        self,
        symbol:  str,
        candle:  dict,
        candles: List[Dict],   # FIX #8: passed to Warden for structure BE
    ) -> None:
        """Check SL / TP / structure-BE on each candle close for open position."""

        # Send live price to dashboard so P&L updates continuously
        await bot_state.push_price_update(symbol, float(candle["close"]))

        # ── CRITICAL: capture position data BEFORE Warden potentially closes it ──
        pos = self._warden.get_position(symbol)
        if pos is None:
            return
        saved_qty          = pos.quantity
        saved_entry        = pos.entry_price
        saved_sl           = pos.stop_loss
        saved_tp           = pos.take_profit
        saved_strat        = pos.strategy
        saved_sl_order_id  = pos.sl_order_id
        saved_tp_order_id  = pos.tp_order_id

        result = await self._warden.check_position(
            symbol=symbol,
            latest_close=candle["close"],
            latest_high=candle["high"],
            latest_low=candle["low"],
            candles=candles,    # FIX #8
        )

        if result in ("SL", "TP"):
            exit_price = saved_sl if result == "SL" else saved_tp

            # ── Dual-exit guard ───────────────────────────────────────────────
            # Binance STOP_MARKET / TAKE_PROFIT_MARKET may have already fired
            # intra-candle.  Check actual exchange position size before placing
            # a second sell — otherwise we'd create an unintended short.
            if config.USE_FUTURES:
                try:
                    exchange_positions = await self._exchange.fetch_positions([symbol])
                    already_flat = all(
                        abs(float(p.get("contracts") or p.get("positionAmt") or 0)) < 0.0001
                        for p in exchange_positions
                        if p.get("symbol") in (symbol, symbol.replace("/", ""))
                    )
                    if already_flat:
                        log.info(
                            "✅ %s already flat on exchange (native SL/TP fired) — "
                            "skipping manual sell, syncing dashboard",
                            symbol,
                        )
                        pnl_pct = (exit_price - saved_entry) / saved_entry * 100
                        await self._notifier.send(
                            Notifier.trade_closed_msg(
                                symbol=symbol, reason=f"{result}_NATIVE",
                                entry=saved_entry, exit_price=exit_price, pnl_pct=pnl_pct,
                            )
                        )
                        return
                except Exception as guard_exc:
                    log.warning(
                        "Dual-exit guard check failed for %s: %s — proceeding with sell",
                        symbol, guard_exc,
                    )

            # ── Cancel remaining exchange SL/TP orders before selling ─────────
            # Prevents a race where our limit-sell AND the native order both fill.
            if config.USE_FUTURES:
                for oid in filter(None, [saved_sl_order_id, saved_tp_order_id]):
                    try:
                        await self._exchange.cancel_order(oid, symbol)
                        log.info("🧹 Cancelled exchange order %s before manual sell", oid)
                    except Exception:
                        pass  # already filled or cancelled — safe to ignore

            await self._execute_sell(symbol, exit_price, quantity=saved_qty)

            pnl_pct = (exit_price - saved_entry) / saved_entry * 100
            await self._notifier.send(
                Notifier.trade_closed_msg(
                    symbol=symbol,
                    reason=result,
                    entry=saved_entry,
                    exit_price=exit_price,
                    pnl_pct=pnl_pct,
                )
            )

        elif result in ("TSL", "BE"):
            pos = self._warden.get_position(symbol)
            if pos:
                # ── Sync TSL move to Binance ─────────────────────────────────
                if config.USE_FUTURES and pos.sl_order_id:
                    try:
                        log.info("🔁 Syncing TSL move to Binance for %s", symbol)
                        # Cancel old SL
                        try:
                            await self._exchange.cancel_order(pos.sl_order_id, symbol)
                        except Exception as cancel_exc:
                            log.warning("Could not cancel old SL %s: %s", pos.sl_order_id, cancel_exc)

                        # Place new SL
                        sl_price = float(self._exchange.price_to_precision(symbol, pos.stop_loss))
                        new_sl_order = await self._exchange.create_order(
                            symbol, "STOP_MARKET", "sell", pos.quantity,
                            params={
                                "stopPrice": sl_price,
                                "reduceOnly": True,
                                "positionSide": "BOTH"
                            }
                        )
                        pos.sl_order_id = new_sl_order.get("id")
                        log.info("✅ Binance SL updated to %s", sl_price)

                    except Exception as sync_exc:
                        log.error("Failed to sync TSL to Binance for %s: %s", symbol, sync_exc)

                await self._notifier.send(
                    f"🔁 <b>SL trailed</b> for <code>{symbol}</code> — "
                    f"new SL: <code>{pos.stop_loss:.6g}</code> "
                    f"{'(Break-Even ✅)' if pos.be_activated else '(trailing 📈)'}"
                )
                asyncio.create_task(bot_state.push_breakeven_activated(
                    symbol=symbol, new_sl=pos.stop_loss, sl_order_id=pos.sl_order_id,
                    be_activated=pos.be_activated,
                ))

    # ── Execute a limit sell ──────────────────────────────────────────────────
    async def _execute_sell(self, symbol: str, exit_price: float, quantity: float = 0.0) -> None:
        """
        Place a limit sell at the current bid price.
        Falls back to market sell if limit not filled within timeout.
        """
        if quantity <= 0:
            # Fallback: try reading from warden (may still exist for manual sells)
            pos = self._warden.get_position(symbol)
            quantity = pos.quantity if pos else 0.0
            
        pos = self._warden.get_position(symbol)
        direction = pos.direction if pos else "long"

        if quantity <= 0:
            log.warning("No quantity to close for %s", symbol)
            return

        try:
            ticker    = await self._exchange.fetch_ticker(symbol)
            if direction == "long":
                close_price = float(ticker.get("bid") or ticker.get("last") or exit_price)
                close_side = "sell"
            else:
                close_price = float(ticker.get("ask") or ticker.get("last") or exit_price)
                close_side = "buy"

            qty_str   = float(self._exchange.amount_to_precision(symbol, quantity))
            price_str = float(self._exchange.price_to_precision(symbol, close_price))

            if config.USE_FUTURES:
                order = await self._exchange.create_order(
                    symbol, "limit", close_side, qty_str, price_str,
                    params={"timeInForce": "GTC", "positionSide": "BOTH"},
                )
            else:
                if close_side == "sell":
                    order = await self._exchange.create_limit_sell_order(symbol, qty_str, price_str)
                else:
                    order = await self._exchange.create_limit_buy_order(symbol, qty_str, price_str)
                    
            order_id = order["id"]
            log.info(
                "📤 Limit Close placed: %s qty=%.6g @ %.6g | id=%s",
                symbol, qty_str, price_str, order_id,
            )

            # Brief wait for fill (sells at bid usually fill fast)
            deadline = time.time() + _ENTRY_ORDER_TIMEOUT
            while time.time() < deadline:
                await asyncio.sleep(5)
                try:
                    order = await self._exchange.fetch_order(order_id, symbol)
                except Exception:
                    continue
                if order.get("status") == "closed":
                    break

            if order.get("status") != "closed":
                # Fall back to market sell/buy
                log.warning(
                    "Limit close timed out for %s — falling back to market close",
                    symbol,
                )
                await self._exchange.cancel_order(order_id, symbol)
                if config.USE_FUTURES:
                    order = await self._exchange.create_order(
                        symbol, "market", close_side, qty_str,
                        params={"positionSide": "BOTH"},
                    )
                else:
                    if close_side == "sell":
                        order = await self._exchange.create_market_sell_order(symbol, qty_str)
                    else:
                        order = await self._exchange.create_market_buy_order(symbol, qty_str)

            filled = float(order.get("average") or order.get("price") or close_price)
            log.info(
                "💰 Close confirmed: %s qty=%.6g @ %.6g | id=%s",
                symbol, qty_str, filled, order.get("id"),
            )

            # ── Cleanup exchange SL/TP orders (FIX #12) ──────────────────────
            # If the trade is closed, we must kill any remaining SL/TP orders on Binance.
            pos = self._warden.get_position(symbol)
            if pos:
                for oid in [pos.sl_order_id, pos.tp_order_id]:
                    if oid:
                        try:
                            log.info("🧹 Cleaning up ghost order on Binance: %s", oid)
                            await self._exchange.cancel_order(oid, symbol)
                        except Exception:
                            # Usually means the order already filled or was already cancelled
                            pass

        except Exception as exc:
            log.error(
                "Sell failed for %s: %s — position may still be open!",
                symbol, exc, exc_info=True,
            )
            await self._notifier.send(
                Notifier.error_msg(f"Sell {symbol}", exc)
            )

    # ── Helpers ───────────────────────────────────────────────────────────────
    async def _get_candles(
        self, symbol: str, timeframe: str
    ) -> Optional[List[Dict]]:
        buf = self._registry.get(symbol, timeframe)
        if buf is None:
            return None
        try:
            return await buf.snapshot()
        except Exception:
            return None

    async def _get_df(
        self, symbol: str, timeframe: str
    ) -> Optional[pd.DataFrame]:
        buf = self._registry.get(symbol, timeframe)
        if buf is None:
            return None
        try:
            return await buf.to_dataframe()
        except Exception:
            return None

    async def _get_htf_df(self, symbol: str, tf: str = None) -> Optional[pd.DataFrame]:
        """
        Fetch historical candles for HTF filtering.
        """
        tf = tf or config.HTF_FILTER_TF
        # Check registry
        buf = self._registry.get(symbol, tf)
        if buf is not None:
            try:
                df = await buf.to_dataframe()
                if df is not None and len(df) >= 50:
                    return df
            except Exception:
                pass

        # REST fallback — OHLCV is a public endpoint, no auth needed
        try:
            raw = await self._exchange.fetch_ohlcv(
                symbol,
                config.HTF_FILTER_TF,
                limit=config.HTF_CANDLE_LIMIT,
            )
            if not raw:
                return None
            df = pd.DataFrame(
                raw,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            return df
        except Exception as exc:
            log.warning(
                "HTF OHLCV fetch failed for %s %s: %s — HTF filter will be skipped",
                symbol, config.HTF_FILTER_TF, exc,
            )
            return None
