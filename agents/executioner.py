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
from agents.warden        import (
    WardenAgent,
    calculate_stop_loss_atr,
    calculate_stop_loss,
    calculate_take_profit,
    calculate_position_size,
)
from data.candle_buffer   import BufferRegistry
from utils.indicators     import (
    calc_atr,
    market_regime,
    price_above_ema200,
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

        # ── Pre-compute common filters (15m + 1h) ────────────────────────────
        htf_15m_df = await self._get_htf_df(symbol, "15m")
        htf_1h_df  = await self._get_htf_df(symbol, "1h")
        
        # EMA filter on 1h (macro)
        htf_above_ema = price_above_ema200(htf_1h_df) if htf_1h_df is not None else True
        
        regime, adx, di_plus, di_minus = market_regime(symbol_df)
        atr = calc_atr(symbol_df)
        btc_df = await self._get_df(config.BTC_SYMBOL, config.BTC_TIMEFRAME)

        valid_signals = []
        for sig in strategy_signals:
            strategy = sig["strategy"]
            log.info("🎯 Signal detected: %s [%s] strategy=%s — running pre-trade filters...", symbol, timeframe, strategy)
            if strategy in ("bounce", "fvg"):
                if not htf_above_ema:
                    log.info("❌ HTF filter rejected %s (%s) — price is BELOW 200-EMA", symbol, strategy)
                    continue
                if config.ADX_REQUIRE_DIRECTION and di_minus >= di_plus:
                    log.info("❌ ADX direction rejected %s (%s)", symbol, strategy)
                    continue
                if adx < config.ADX_TREND_THRESHOLD:
                    log.info("❌ ADX filter rejected %s (%s) — ADX=%.1f (ranging)", symbol, strategy, adx)
                    continue
            elif strategy in ("vwap_bounce", "rsi_divergence"):
                if not htf_above_ema:
                    log.info("❌ HTF filter rejected %s (%s) — price is BELOW 200-EMA", symbol, strategy)
                    continue
                if di_minus >= di_plus:
                    log.info("❌ Direction rejected %s (%s) — DI- > DI+", symbol, strategy)
                    continue
            elif strategy == "breakout":
                if di_minus >= di_plus:
                    log.info("❌ Direction rejected %s (%s) — DI- > DI+", symbol, strategy)
                    continue
            elif strategy == "mean_reversion":
                if adx >= config.ADX_TREND_THRESHOLD:
                    log.info("❌ Mean Reversion rejected %s — ADX=%.1f (trending)", symbol, adx)
                    continue

            log.info("✅ Pre-trade filters passed for %s (strategy: %s)", symbol, strategy)
            valid_signals.append(sig)

        if not valid_signals:
            return

        async def _eval_signal(sig):
            strategy = sig["strategy"]
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
                if atr > 0: raw_sl = calculate_stop_loss_atr(entry_est, atr)
                else: raw_sl = calculate_stop_loss(entry_est, conf_candle["low"])
                    
            raw_tp = sig.get("take_profit")
            tp_source = "strategy_default"
            if not raw_tp:
                if swing_tp_math and swing_tp_math > entry_est:
                    raw_tp = swing_tp_math
                    tp_source = "swing_high"
                else:
                    raw_tp = calculate_take_profit(entry_est, raw_sl)
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
        
        asyncio.create_task(bot_state.push_sim_entry(
            symbol=symbol,
            timeframe=timeframe,
            entry_price=best_sig["entry_price"],
            stop_loss=best_sl,
            take_profit=best_tp,
            strategy=strategy,
        ))

        await self._execute_limit_buy(
            symbol=symbol,
            conf_candle=best_conf_candle,
            atr=atr,
            swing_tp=best_swing_tp,
            target_sl=best_sl,
            target_tp=best_tp,
            strategy=strategy,
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
            # ── Fetch balance (FIX #6 seeds daily tracker) ───────────────────
            balance   = await self._exchange.fetch_balance()
            usdt_free = float(balance.get("USDT", {}).get("free", 0.0))
            self._warden.daily_loss.set_start_balance(usdt_free)

            if usdt_free < 10:
                log.warning(
                    "Insufficient USDT balance (%.2f) — skipping buy for %s",
                    usdt_free, symbol,
                )
                return

            # ── Get current ask for limit price ──────────────────────────────
            ticker      = await self._exchange.fetch_ticker(symbol)
            ask_price   = float(ticker.get("ask") or ticker.get("last") or 0.0)
            if ask_price <= 0:
                log.error("Invalid ask price for %s — aborting buy", symbol)
                return

            # ── SL / TP Calculation ──────────────────────────────────────────
            if target_sl is not None:
                stop_loss = target_sl
            elif atr > 0:
                stop_loss = calculate_stop_loss_atr(ask_price, atr)
            else:
                stop_loss = calculate_stop_loss(ask_price, conf_candle["low"])

            if target_tp is not None:
                take_profit = target_tp
            elif swing_tp and swing_tp > ask_price:
                take_profit = swing_tp
            else:
                take_profit = calculate_take_profit(ask_price, stop_loss)
            
            # User request: £100 for confident, £50 for regular
            stake_gbp = config.CAPITAL_CONFIDENT if confidence >= config.CONFIDENCE_LEVEL else config.CAPITAL_REGULAR

            # ── Futures Setup ────────────────────────────────────────────────
            if config.USE_FUTURES:
                try:
                    await self._exchange.set_leverage(leverage, symbol)
                    log.info("🎯 Leverage set to %dx for %s", leverage, symbol)
                except Exception as lev_exc:
                    log.warning("Failed to set leverage for %s: %s", symbol, lev_exc)

            quantity = calculate_position_size(
                balance_usdt=usdt_free, 
                entry_price=ask_price, 
                stop_loss=stop_loss,
                fixed_stake=stake_gbp
            )
            if quantity <= 0:
                log.warning("Zero quantity for %s — skipping", symbol)
                return

            # Round to exchange precision
            quantity = float(
                self._exchange.amount_to_precision(symbol, quantity)
            )
            # Add a 0.2% adaptive slippage buffer. This forces the limit order to heavily cross the 
            # order book spread, acting essentially as a protected Market Buy to guarantee our bounce entry.
            adaptive_limit = ask_price * 1.002
            limit_price = float(
                self._exchange.price_to_precision(symbol, adaptive_limit)
            )

            log.info(
                "🛒 Placing Limit Buy: %s qty=%.6g @ %.6g | SL=%.6g | TP=%.6g",
                symbol, quantity, limit_price, stop_loss, take_profit,
            )

            # ── Place order ───────────────────────────────────────────────────
            order = await self._exchange.create_limit_buy_order(
                symbol, quantity, limit_price
            )
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
                stop_loss = calculate_stop_loss_atr(filled_price, atr)
            else:
                stop_loss = calculate_stop_loss(filled_price, conf_candle["low"])

            if target_tp is not None:
                take_profit = target_tp
            elif swing_tp and swing_tp > filled_price:
                take_profit = swing_tp
            else:
                take_profit = calculate_take_profit(filled_price, stop_loss)

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
            )

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
        result = await self._warden.check_position(
            symbol=symbol,
            latest_close=candle["close"],
            latest_high=candle["high"],
            latest_low=candle["low"],
            candles=candles,    # FIX #8
        )

        if result in ("SL", "TP"):
            pos = self._warden.get_position(symbol)   # already removed if closed
            exit_price  = candle["low"]  if result == "SL" else candle["high"]
            # pos is None here because Warden already popped it; use candle prices
            entry_price = 0.0            # we log approximate values

            await self._execute_sell(symbol, exit_price)

            await self._notifier.send(
                Notifier.trade_closed_msg(
                    symbol=symbol,
                    reason=result,
                    entry=entry_price,
                    exit_price=exit_price,
                    pnl_pct=0.0,    # Warden already logged exact P&L
                )
            )

        elif result in ("TSL", "BE"):
            pos = self._warden.get_position(symbol)
            if pos:
                await self._notifier.send(
                    f"🔁 <b>SL trailed</b> for <code>{symbol}</code> — "
                    f"new SL: <code>{pos.stop_loss:.6g}</code> "
                    f"{'(Break-Even ✅)' if pos.be_activated else '(trailing 📈)'}"
                )

    # ── Execute a limit sell ──────────────────────────────────────────────────
    async def _execute_sell(self, symbol: str, exit_price: float) -> None:
        """
        Place a limit sell at the current bid price.
        Falls back to market sell if limit not filled within timeout.
        """
        pos      = self._warden.get_position(symbol)
        quantity = pos.quantity if pos else 0.0

        if quantity <= 0:
            log.warning("No quantity to sell for %s", symbol)
            return

        try:
            ticker    = await self._exchange.fetch_ticker(symbol)
            bid_price = float(ticker.get("bid") or ticker.get("last") or exit_price)
            qty_str   = float(self._exchange.amount_to_precision(symbol, quantity))
            bid_str   = float(self._exchange.price_to_precision(symbol, bid_price))

            order = await self._exchange.create_limit_sell_order(
                symbol, qty_str, bid_str
            )
            order_id = order["id"]
            log.info(
                "📤 Limit Sell placed: %s qty=%.6g @ %.6g | id=%s",
                symbol, qty_str, bid_str, order_id,
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
                # Fall back to market sell
                log.warning(
                    "Limit sell timed out for %s — falling back to market sell",
                    symbol,
                )
                await self._exchange.cancel_order(order_id, symbol)
                order = await self._exchange.create_market_sell_order(
                    symbol, qty_str
                )

            filled = float(order.get("average") or order.get("price") or bid_price)
            log.info(
                "💰 Sell confirmed: %s qty=%.6g @ %.6g | id=%s",
                symbol, qty_str, filled, order.get("id"),
            )

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
