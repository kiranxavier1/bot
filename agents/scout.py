"""
agents/scout.py
─────────────────────────────────────────────────────────────────────────────
The Scout Agent — two responsibilities:

1. SYMBOL DISCOVERY  (runs every 60 min)
   Fetches the Top-N USDT pairs on Binance by 24h quote volume via REST,
   returns the ranked list, and tells main.py which symbols to add / remove
   from the active WebSocket pool.

2. WEBSOCKET STREAM MANAGER
   Opens one watch_ohlcv() WebSocket per (symbol, timeframe) using ccxt.pro,
   pipes each tick into the matching CandleBuffer, and calls the provided
   `on_candle_close` callback whenever a *closed* candle is detected.

   "Closed" means the new tick's timestamp is strictly greater than the
   previous tick's timestamp for that symbol-TF pair — i.e. the old candle
   has been sealed and a fresh one has opened.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Dict, List, Optional, Set

import ccxt.pro as ccxtpro  # type: ignore

import config
from utils.candle_buffer import BufferRegistry
from web.state import bot_state

log = logging.getLogger(__name__)

# Callback signature: async def on_candle_close(symbol, timeframe, candle_dict)
OnCandleClose = Callable[[str, str, dict], Awaitable[None]]


class ScoutAgent:
    """
    Manages the Binance WebSocket streams and symbol discovery.

    Parameters
    ----------
    registry    : shared BufferRegistry
    on_close_cb : async callback fired when a candle closes
    """

    def __init__(
        self,
        registry: BufferRegistry,
        on_close_cb: Optional[OnCandleClose] = None,
    ):
        self._registry    = registry
        self._on_close_cb = on_close_cb
        self._exchange: Optional[ccxtpro.binance] = None

        # Tracking last seen timestamp per (symbol, tf) to detect close events
        self._last_ts: Dict[tuple, int] = {}

        # Currently active streaming symbols
        self._active_symbols: Set[str] = set()

        # Tasks keyed by (symbol, tf)
        self._stream_tasks: Dict[tuple, asyncio.Task] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    async def start(self) -> None:
        """Initialise the exchange connection."""
        # Public exchange — no API key needed for market data (tickers, OHLCV)
        self._public_exchange = ccxtpro.binance(
            {"options": {"defaultType": "spot"}, "enableRateLimit": True}
        )
        log.info("ScoutAgent started — exchange: binance (spot)")

    async def stop(self) -> None:
        """Cancel all stream tasks and close the exchange connection."""
        for task in self._stream_tasks.values():
            task.cancel()
        self._stream_tasks.clear()
        if self._exchange:
            await self._exchange.close()
        if hasattr(self, '_public_exchange') and self._public_exchange:
            await self._public_exchange.close()
        log.info("ScoutAgent stopped — exchange connection closed.")

    # ── Symbol Discovery ──────────────────────────────────────────────────────
    async def discover_top_symbols(self) -> List[str]:
        """
        Fetch the Top-N USDT pairs by 24h quote volume via REST.
        Returns a list of symbols like ["BTC/USDT", "ETH/USDT", ...].
        """
        try:
            tickers = await self._public_exchange.fetch_tickers()
            usdt_pairs = []
            for sym, t in tickers.items():
                if not sym.endswith("/USDT"):
                    continue
                if any(x in sym for x in [":USDT", "UP/", "DOWN/", "BULL/", "BEAR/"]):
                    continue
                quote_vol = t.get("quoteVolume") or 0.0
                # Volatility filter: require minimum 24h high-low range as % of price.
                # This ensures we only track coins with enough intraday movement for
                # scalping and swing trades. Stablecoins and dead coins are excluded.
                high_24h = t.get("high") or 0.0
                low_24h  = t.get("low")  or 0.0
                close    = t.get("close") or t.get("last") or 0.0
                if close > 0 and low_24h > 0:
                    daily_range_pct = (high_24h - low_24h) / low_24h * 100
                else:
                    daily_range_pct = 0.0
                if daily_range_pct >= config.MIN_DAILY_RANGE_PCT:
                    usdt_pairs.append((sym, quote_vol))
            usdt_pairs.sort(key=lambda x: x[1], reverse=True)
            top = [sym for sym, _ in usdt_pairs[: config.TOP_N_PAIRS]]
            # Always include BTC for macro sentiment
            if config.BTC_SYMBOL not in top:
                top.insert(0, config.BTC_SYMBOL)
            log.info("Symbol discovery: found %d top USDT pairs", len(top))
            return top
        except Exception as exc:
            log.error("Symbol discovery failed: %s", exc)
            return list(self._active_symbols) or [config.BTC_SYMBOL]

    async def update_symbol_pool(self, new_symbols: List[str]) -> None:
        """
        Diff the current symbol pool vs `new_symbols`.
        - Start streams for newly added symbols.
        - Cancel streams for removed symbols.
        - Seed buffers for new symbols via REST before opening WebSocket.
        """
        new_set = set(new_symbols)
        to_add    = new_set - self._active_symbols
        to_remove = self._active_symbols - new_set

        # Stop removed streams
        for sym in to_remove:
            for tf in config.TIMEFRAMES:
                key = (sym, tf)
                if key in self._stream_tasks:
                    self._stream_tasks[key].cancel()
                    del self._stream_tasks[key]
            self._registry.remove_symbol(sym)
            self._active_symbols.discard(sym)
            log.info("Removed symbol from pool: %s", sym)

        # Start added streams
        for sym in to_add:
            await self._bootstrap_symbol(sym)
            self._active_symbols.add(sym)
            log.info("Added symbol to pool: %s", sym)

    # ── Bootstrap via REST ────────────────────────────────────────────────────
    async def _bootstrap_symbol(self, symbol: str) -> None:
        """Fetch historical candles via REST and seed the buffer before WS."""
        for tf in config.TIMEFRAMES:
            buf = self._registry.get_or_create(symbol, tf)
            try:
                raw = await self._public_exchange.fetch_ohlcv(
                    symbol, tf, limit=config.CANDLE_BUFFER_SIZE
                )
                await buf.seed(raw)
                log.debug("Seeded %s %s: %d candles", symbol, tf, len(raw))
            except Exception as exc:
                log.warning("REST seed failed for %s %s: %s", symbol, tf, exc)

            # Launch WebSocket stream task
            key = (symbol, tf)
            if key not in self._stream_tasks or self._stream_tasks[key].done():
                task = asyncio.create_task(
                    self._stream_loop(symbol, tf),
                    name=f"ws-{symbol}-{tf}",
                )
                self._stream_tasks[key] = task

    # ── WebSocket Stream Loop ─────────────────────────────────────────────────
    async def _stream_loop(self, symbol: str, timeframe: str) -> None:
        """
        Infinite loop consuming watch_ohlcv for one (symbol, timeframe).
        Reconnects automatically on error with exponential back-off.
        """
        key     = (symbol, timeframe)
        buf     = self._registry.get_or_create(symbol, timeframe)
        backoff = 1.0

        while True:
            try:
                # CCXT Pro: watch_ohlcv uses public WebSocket — no auth needed for market data
                candles = await self._public_exchange.watch_ohlcv(symbol, timeframe)
                if not candles:
                    continue
                await buf.update(candles)

                # Push scan event to state
                asyncio.create_task(
                    bot_state.push_scan_event(symbol, timeframe, float(candles[-1][4]))
                )

                # Detect candle close event
                latest_ts = int(candles[-1][0])
                prev_ts   = self._last_ts.get(key, 0)

                if prev_ts != 0 and latest_ts > prev_ts:
                    # The old candle has now closed — fire the callback
                    if self._on_close_cb:
                        try:
                            closed = await buf.last_closed(1)
                            if closed:
                                asyncio.create_task(
                                    self._on_close_cb(
                                        symbol, timeframe, closed[-1]
                                    )
                                )
                        except Exception as cb_exc:
                            log.error(
                                "on_candle_close callback error (%s %s): %s",
                                symbol, timeframe, cb_exc,
                            )

                self._last_ts[key] = latest_ts
                backoff = 1.0   # reset back-off on successful tick

            except asyncio.CancelledError:
                log.debug("Stream cancelled: %s %s", symbol, timeframe)
                return
            except Exception as exc:
                log.warning(
                    "WS stream error %s %s — reconnecting in %.0fs: %s",
                    symbol, timeframe, backoff, exc,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    # ── Discovery loop (called from main.py) ──────────────────────────────────
    async def run_discovery_loop(self) -> None:
        """
        Runs forever: discover top symbols every DISCOVERY_INTERVAL_SECS.
        Designed to be launched as an asyncio.Task.
        """
        while True:
            try:
                symbols = await self.discover_top_symbols()
                await self.update_symbol_pool(symbols)
                await bot_state.set_symbols_tracked(len(self._active_symbols))
            except asyncio.CancelledError:
                return
            except Exception as exc:
                log.error("Discovery loop error: %s", exc)
            await asyncio.sleep(config.DISCOVERY_INTERVAL_SECS)
