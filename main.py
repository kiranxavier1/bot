"""
main.py
─────────────────────────────────────────────────────────────────────────────
Entry point — wires all agents together and runs the asyncio event loop.

Architecture at a glance:
                                                        ┌──────────────┐
  ┌─────────┐   candle close   ┌──────────────────┐    │  AI Manager  │
  │  Scout  │ ────────────────▶│   Executioner    │───▶│  (Claude)    │
  │ (WS/REST│                  │                  │    └──────────────┘
  │  agent) │                  │  Mathematician   │
  └─────────┘                  │  Warden          │
       │                       └──────────────────┘
       │ discovery loop
       ▼
  BufferRegistry
  (shared rolling OHLCV buffers)

Start-up sequence:
  1. Load config, start logging.
  2. Create shared objects (registry, agents, notifier).
  3. Scout fetches the initial Top-50 symbol list and seeds REST candles.
  4. Scout opens WebSocket streams for all symbols × timeframes.
  5. Scout registers `executioner.on_candle_close` as the close callback.
  6. BTC 1h buffer is seeded and streamed separately (for AI context).
  7. Discovery loop runs every 60 min to refresh the symbol pool.

Graceful shutdown via Ctrl-C:
  - All stream tasks are cancelled.
  - Exchange connection is closed.
  - Telegram sends a "Bot stopped" notice.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import os

import ccxt.pro as ccxtpro  # type: ignore

import config
from utils.logger        import setup_logging
from utils.notifications import Notifier
from data.candle_buffer  import BufferRegistry
from agents.scout        import ScoutAgent
from agents.mathematician import MathematicianAgent
from agents.ai_manager   import AIManager
from agents.warden       import WardenAgent
from agents.executioner  import ExecutionerAgent
from web.server          import run_server
from web.state           import bot_state

log = logging.getLogger("main")


async def seed_btc_1h(registry: BufferRegistry) -> None:
    """
    Pre-seed the BTC/USDT 1h buffer using a public (no-auth) exchange instance.
    """
    buf = registry.get_or_create(config.BTC_SYMBOL, config.BTC_TIMEFRAME)
    public = ccxtpro.binance({"options": {"defaultType": "spot"}, "enableRateLimit": True})
    try:
        raw = await public.fetch_ohlcv(
            config.BTC_SYMBOL, config.BTC_TIMEFRAME, limit=config.CANDLE_BUFFER_SIZE
        )
        await buf.seed(raw)
        log.info("BTC 1h buffer seeded: %d candles", len(raw))
    except Exception as exc:
        log.warning("BTC 1h seed failed: %s", exc)
    finally:
        await public.close()


async def btc_stream_loop(
    scout: "ScoutAgent",
    registry: BufferRegistry,
) -> None:
    """
    Maintain a live WebSocket stream for BTC/USDT 1h candles.
    Runs as a long-lived background task.
    """
    buf = registry.get_or_create(config.BTC_SYMBOL, config.BTC_TIMEFRAME)
    backoff = 1.0
    while True:
        try:
            # CCXT Pro: watch_ohlcv is a coroutine called in a loop, not an async generator
            # BTC 1h uses the public exchange — market data doesn't need auth
            candles = await scout._public_exchange.watch_ohlcv(
                config.BTC_SYMBOL, config.BTC_TIMEFRAME
            )
            if candles:
                await buf.update(candles)
                backoff = 1.0
        except asyncio.CancelledError:
            return
        except Exception as exc:
            log.warning(
                "BTC 1h stream error — reconnecting in %.0fs: %s", backoff, exc
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)


async def print_status_loop(warden: WardenAgent) -> None:
    """Log a periodic status summary every 15 minutes."""
    while True:
        await asyncio.sleep(900)
        summary = warden.summary()
        log.info(
            "📊 Status: %d open position(s) — %s | cooldowns: %s",
            summary["open_positions"],
            summary["symbols"] or "none",
            summary["cooldowns"] or "none",
        )


async def main() -> None:
    # ── 1. Logging ────────────────────────────────────────────────────────────
    setup_logging()
    log.info("=" * 60)
    log.info("  3rd Touch Trendline Bot — starting up")
    log.info("  Exchange  : Binance Spot")
    log.info("  AI Model  : %s", config.CLAUDE_MODEL)
    log.info("  Top pairs : %d", config.TOP_N_PAIRS)
    log.info("  Timeframes: %s", config.TIMEFRAMES)
    log.info("=" * 60)

    # ── 2. Shared objects ─────────────────────────────────────────────────────
    notifier  = Notifier()
    registry  = BufferRegistry(max_size=config.CANDLE_BUFFER_SIZE)
    math_agent = MathematicianAgent()
    ai_manager = AIManager()
    warden     = WardenAgent(notifier=notifier)

    # Shared exchange instance (REST + WS)
    exchange = ccxtpro.binance(
        {
            "apiKey":  config.BINANCE_API_KEY,
            "secret":  config.BINANCE_API_SECRET,
            "options": {"defaultType": "spot"},
            "enableRateLimit": True,
        }
    )

    executioner = ExecutionerAgent(
        exchange=exchange,
        registry=registry,
        mathematician=math_agent,
        ai_manager=ai_manager,
        warden=warden,
        notifier=notifier,
    )

    # ── 3. Scout ──────────────────────────────────────────────────────────────
    scout = ScoutAgent(
        registry=registry,
        on_close_cb=executioner.on_candle_close,
    )
    # Give scout the authenticated exchange for WebSocket streams (trades/orders)
    # and initialise its public exchange for market data reads
    scout._exchange = exchange
    await scout.start()
    log.info("ScoutAgent initialised with shared exchange connection")

    # ── 4. Seed BTC 1h (for AI macro context) ────────────────────────────────
    await seed_btc_1h(registry)

    # ── 5. Initial symbol discovery + REST seed + WebSocket open ─────────────
    symbols = await scout.discover_top_symbols()
    await scout.update_symbol_pool(symbols)
    await bot_state.set_symbols_tracked(len(symbols))
    log.info("Initial pool: %d symbols × %d timeframes", len(symbols), len(config.TIMEFRAMES))

    # ── 6. Notify bot is live ─────────────────────────────────────────────────
    await notifier.send(
        f"🤖 <b>Trading Bot ONLINE</b>\n"
        f"Exchange  : Binance Spot\n"
        f"Pairs     : {len(symbols)} USDT symbols\n"
        f"Timeframes: {', '.join(config.TIMEFRAMES)}\n"
        f"AI model  : <code>{config.CLAUDE_MODEL}</code>"
    )

    # ── 7. Background tasks ───────────────────────────────────────────────────
    port = int(os.environ.get("PORT", 8000))
    tasks = [
        asyncio.create_task(scout.run_discovery_loop(),          name="discovery"),
        asyncio.create_task(btc_stream_loop(scout, registry), name="btc-1h"),
        asyncio.create_task(print_status_loop(warden),           name="status"),
        asyncio.create_task(run_server(host="0.0.0.0", port=port), name="web-ui"),
    ]

    log.info("All tasks started — bot is running.  Press Ctrl-C to stop.")
    log.info("📊 Dashboard → http://localhost:8000")

    # ── 8. Run until cancelled ────────────────────────────────────────────────
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        log.info("Shutdown signal received — cleaning up...")
        for t in tasks:
            t.cancel()
        await bot_state.set_stopped()
        await scout.stop()
        await exchange.close()
        await notifier.send("🛑 <b>Trading Bot OFFLINE</b> — graceful shutdown complete.")
        await notifier.close()
        log.info("Bot stopped cleanly.")


def _handle_signal(signum, frame):
    log.info("Signal %s received — initiating shutdown", signum)
    loop = asyncio.get_event_loop()
    for task in asyncio.all_tasks(loop):
        task.cancel()


if __name__ == "__main__":
    # Register Ctrl-C / SIGTERM handlers
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        sys.exit(0)
