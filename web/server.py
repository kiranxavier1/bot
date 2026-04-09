"""
web/server.py
─────────────────────────────────────────────────────────────────────────────
FastAPI web server — provides:

  REST endpoints
  ─────────────
  GET  /api/status     → bot meta (uptime, symbols tracked, running flag)
  GET  /api/positions  → current open positions
  GET  /api/trades     → closed trade history
  GET  /api/ai-log     → AI decision log
  GET  /api/stats      → aggregate win/loss/P&L stats
  GET  /api/snapshot   → full state in one call

  WebSocket
  ─────────
  WS   /ws             → streams full state JSON on every change

  Static files
  ────────────
  GET  /               → serves web/static/index.html  (the React dashboard)

Started as an asyncio task inside main.py using uvicorn programmatically,
so it shares the same event loop as the trading bot — no separate process.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from web.state import bot_state
from agents.post_mortem import load_analysis, load_strategy_data, list_all_strategies

log = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Trading Bot Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = Path(__file__).parent / "static"

# Serve static assets (CSS, JS if added later)
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_dashboard():
    """Serve the single-page React dashboard."""
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/api/snapshot")
async def get_snapshot():
    return await bot_state.snapshot()


@app.get("/api/status")
async def get_status():
    snap = await bot_state.snapshot()
    return snap["meta"]


@app.get("/api/positions")
async def get_positions():
    snap = await bot_state.snapshot()
    return snap["positions"]


@app.get("/api/trades")
async def get_trades():
    snap = await bot_state.snapshot()
    return snap["trades"]


@app.get("/api/ai-log")
async def get_ai_log():
    snap = await bot_state.snapshot()
    return snap["ai_log"]


@app.get("/api/stats")
async def get_stats():
    snap = await bot_state.snapshot()
    return snap["stats"]


@app.get("/api/trade-analysis/{trade_id}")
async def api_get_analysis(trade_id: str):
    data = load_analysis(trade_id)
    if data is None:
        return JSONResponse(status_code=404, content={"error": "Analysis not found"})
    return data


@app.get("/api/strategy-lessons/{strategy}")
async def api_get_strategy_lessons(strategy: str):
    data = load_strategy_data(strategy)
    return data


@app.get("/api/strategies")
async def api_list_strategies():
    return list_all_strategies()

@app.post("/api/dashboard/reset")
async def api_reset_dashboard():
    """Wipe all trade history, stats, AI log, and P&L from the dashboard."""
    await bot_state.reset_dashboard()
    return {"success": True}


@app.post("/api/strategy-lessons/reset")
async def api_reset_strategy_lessons():
    from agents.post_mortem import clear_all_strategy_rules
    clear_all_strategy_rules()
    return {"success": True}

class AllocationUpdate(BaseModel):
    pct: float

@app.post("/api/settings/allocation")
async def api_update_allocation(data: AllocationUpdate):
    if 1.0 <= data.pct <= 100.0:
        bot_state.trade_allocation_pct = round(data.pct, 2)
        bot_state._save_to_disk()
        await bot_state._broadcast()
        return {"success": True, "trade_allocation_pct": bot_state.trade_allocation_pct}
    return JSONResponse(status_code=400, content={"error": "Percentage must be between 1 and 100"})


# ── Test trade (order-flow verification) ──────────────────────────────────────
# Set by main.py after exchange is initialised
_test_exchange = None

def set_test_exchange(exchange) -> None:
    global _test_exchange
    _test_exchange = exchange

@app.post("/api/test-trade")
async def api_test_trade():
    """
    Fires a real market buy on DOGE/USDT (minimum notional, ~$1 worth),
    waits 4 seconds, then market sells the full qty.
    Use this to verify order placement is working end-to-end.
    """
    import config as _cfg

    if _test_exchange is None:
        return JSONResponse(status_code=503, content={"error": "Exchange not initialised yet — bot still starting up."})

    symbol = "DOGE/USDT"
    try:
        # Step 1: get current price
        ticker = await _test_exchange.fetch_ticker(symbol)
        price  = float(ticker.get("last") or ticker.get("close") or 0)
        if price <= 0:
            return JSONResponse(status_code=500, content={"error": f"Could not fetch price for {symbol}"})

        # Step 2: calculate minimum viable qty (~$1.50 notional to clear Binance's $5 min)
        # Use $6 notional to be safe with leverage
        notional = 6.0
        raw_qty  = notional / price
        qty = float(_test_exchange.amount_to_precision(symbol, raw_qty))

        # Step 3: set leverage (use 1x for test — safest)
        try:
            await _test_exchange.set_leverage(1, symbol)
        except Exception:
            pass  # ignore if already set

        # Step 4: market BUY
        log.info("🧪 TEST TRADE: Market Buy %s qty=%.4g @ ~%.4g", symbol, qty, price)
        buy_order = await _test_exchange.create_order(
            symbol, "market", "buy", qty,
            params={"positionSide": "BOTH"},
        )
        buy_price = float(buy_order.get("average") or buy_order.get("price") or price)
        log.info("🧪 TEST BUY filled: %s qty=%.4g @ %.4g | id=%s", symbol, qty, buy_price, buy_order.get("id"))

        # Step 5: wait 4 seconds
        await asyncio.sleep(4)

        # Step 6: market SELL (close)
        log.info("🧪 TEST TRADE: Market Sell %s qty=%.4g", symbol, qty)
        sell_order = await _test_exchange.create_order(
            symbol, "market", "sell", qty,
            params={"positionSide": "BOTH"},
        )
        sell_price = float(sell_order.get("average") or sell_order.get("price") or price)
        pnl        = (sell_price - buy_price) * qty
        log.info("🧪 TEST SELL filled: %s qty=%.4g @ %.4g | pnl=%+.4f USDT | id=%s",
                 symbol, qty, sell_price, pnl, sell_order.get("id"))

        return {
            "success":    True,
            "symbol":     symbol,
            "qty":        qty,
            "buy_price":  buy_price,
            "sell_price": sell_price,
            "pnl_usdt":   round(pnl, 4),
            "buy_id":     buy_order.get("id"),
            "sell_id":    sell_order.get("id"),
        }

    except Exception as exc:
        log.error("🧪 TEST TRADE FAILED: %s", exc, exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(exc)})



class UpdatePayload(BaseModel):
    action: str

@app.post("/api/system/update")
async def auto_update_bot(payload: UpdatePayload):
    """
    Executes a `git pull` to fetch the latest codebase changes,
    and then immediately replaces the python process with a fresh boot to apply them.
    """
    import os
    import sys
    import subprocess
    
    if payload.action != "restart":
        return JSONResponse(status_code=400, content={"error": "Invalid action"})

    try:
        log.info("📥 Executing git pull...")
        # Run git pull synchronously
        result = subprocess.run(["git", "pull"], capture_output=True, text=True, check=True)
        log.info(f"Git Pull Result: {result.stdout}")
        
        # Give operations 1 second to breathe, execute the restart via background task
        # We can't await it here because the server will die before returning the 200 OK.
        log.warning("🔄 Process replacing itself for core update...")
        os.execv(sys.executable, [sys.executable] + sys.argv)
        
    except subprocess.CalledProcessError as e:
        log.error(f"Git pull failed: {e.stderr}")
        return JSONResponse(status_code=500, content={"error": "Git pull failed."})

# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Accepts a WebSocket connection and streams state snapshots.

    The client receives a full JSON snapshot:
      • immediately on connect
      • whenever any bot event triggers a state change

    The server keeps the connection alive with periodic heartbeat pings.
    """
    await websocket.accept()
    queue = bot_state.subscribe()
    log.info("Dashboard WebSocket connected: %s", websocket.client)

    try:
        # Send initial full snapshot immediately
        initial = await bot_state.snapshot()
        await websocket.send_text(json.dumps({"type": "snapshot", "data": initial}))

        # Stream updates + heartbeat
        while True:
            try:
                # Wait up to 20s for a state update, then send a heartbeat
                snapshot = await asyncio.wait_for(queue.get(), timeout=20.0)
                await websocket.send_text(
                    json.dumps({"type": "snapshot", "data": snapshot})
                )
            except asyncio.TimeoutError:
                # Heartbeat to keep connection alive
                await websocket.send_text(json.dumps({"type": "ping"}))

    except WebSocketDisconnect:
        log.info("Dashboard WebSocket disconnected: %s", websocket.client)
    except Exception as exc:
        log.warning("WebSocket error: %s", exc)
    finally:
        bot_state.unsubscribe(queue)


# ── Uvicorn runner (called from main.py) ──────────────────────────────────────

async def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """
    Start the uvicorn ASGI server inside the existing asyncio event loop.
    Call this as: asyncio.create_task(run_server())
    """
    import uvicorn  # imported here so it's only required when the web UI is used

    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="warning",    # uvicorn's own logs — keep quiet
        access_log=False,
    )
    server = uvicorn.Server(config)
    log.info("Dashboard available at http://%s:%d", host if host != "0.0.0.0" else "localhost", port)
    await server.serve()
