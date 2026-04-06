import sys
import os
import json
import asyncio
from pathlib import Path

# Add root to path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from web.state import bot_state
from agents.post_mortem import RetrainingAgent

async def trigger_retraining():
    print("Feeding historical losses to AI Retrainer...")
    
    # Ensure RetrainingAgent is initialized
    retrainer = RetrainingAgent()
    
    # 1. Load the seeded state
    state_path = os.path.join(os.path.dirname(__file__), "data", "state.json")
    if not os.path.exists(state_path):
        print("❌ Error: data/state.json not found.")
        return
        
    with open(state_path, "r") as f:
        state_data = json.load(f)
        
    sim_trades = state_data.get("sim_trades", [])
    
    # 2. Find losses (pnl_gbp < 0 or reason == "SL")
    losing_trades = [t for t in sim_trades if t.get("pnl_gbp", 0) < 0 or t.get("reason") == "SL"]
    
    if not losing_trades:
        print("No losing trades found in state.json.")
        return
        
    print(f"Found {len(losing_trades)} losing trade(s). Pushing to queue...")
    
    for trade in losing_trades:
        # Push to the bot_state queue
        # RetrainingAgent listens to this queue
        bot_state.losing_trades_queue.put_nowait(trade)
        print(f"Queued: {trade.get('symbol')} ({trade.get('strategy')})")
        
    # 3. Start the retrainer just long enough to process the queue
    # We'll run it manually for a bit
    print("Starting AI analysis for the queued trades...")
    while not bot_state.losing_trades_queue.empty():
        trade = await bot_state.losing_trades_queue.get()
        print(f"Processing {trade.get('symbol')}...")
        try:
            await retrainer._analyze_loss(trade)
            print(f"✅ Analyzed: {trade.get('symbol')}")
        except Exception as e:
            print(f"❌ Analysis failed for {trade.get('symbol')}: {e}")
        bot_state.losing_trades_queue.task_done()

if __name__ == "__main__":
    # Mock environment for config if needed
    os.environ.setdefault("BINANCE_API_KEY", "1")
    os.environ.setdefault("BINANCE_API_SECRET", "1")
    os.environ.setdefault("GEMINI_API_KEY", "1") # Use a real one if testing AI call
    os.environ.setdefault("TELEGRAM_BOT_TOKEN", "1")
    os.environ.setdefault("TELEGRAM_CHAT_ID", "1")
    
    asyncio.run(trigger_retraining())
