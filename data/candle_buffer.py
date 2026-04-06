"""
data/candle_buffer.py
─────────────────────────────────────────────────────────────────────────────
Thread-safe rolling OHLCV buffer for a single (symbol, timeframe) pair.

Each candle is stored as a dict:
    {
        "timestamp": int (ms epoch),
        "open":  float,
        "high":  float,
        "low":   float,
        "close": float,
        "volume": float,
    }

The buffer is capped at `max_size` candles (default 200) to minimise RAM.
Incoming candles from the WebSocket stream are either appended (new candle)
or used to update the last entry (same-timestamp candle update from exchange).
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Deque, Dict, List, Optional

import pandas as pd


# ── OHLCV field positions in the raw CCXT list ───────────────────────────────
_TS, _O, _H, _L, _C, _V = 0, 1, 2, 3, 4, 5


def _candle_from_list(raw: list) -> Dict:
    return {
        "timestamp": int(raw[_TS]),
        "open":      float(raw[_O]),
        "high":      float(raw[_H]),
        "low":       float(raw[_L]),
        "close":     float(raw[_C]),
        "volume":    float(raw[_V]),
    }


class CandleBuffer:
    """Rolling OHLCV buffer for one (symbol, timeframe) pair."""

    def __init__(self, symbol: str, timeframe: str, max_size: int = 200):
        self.symbol    = symbol
        self.timeframe = timeframe
        self.max_size  = max_size
        self._buf: Deque[Dict] = deque(maxlen=max_size)
        self._lock = asyncio.Lock()

    # ── Seed from REST bootstrap ──────────────────────────────────────────────
    async def seed(self, raw_candles: List[list]) -> None:
        """Populate from a list of CCXT OHLCV lists (oldest → newest)."""
        async with self._lock:
            self._buf.clear()
            for raw in raw_candles[-self.max_size:]:
                self._buf.append(_candle_from_list(raw))

    # ── Live update from WebSocket ────────────────────────────────────────────
    async def update(self, raw_candles: List[list]) -> None:
        """
        Called on every WebSocket tick.  `raw_candles` is the CCXT Pro array
        of recently updated candles (usually 1–3 items).
        """
        async with self._lock:
            for raw in raw_candles:
                candle = _candle_from_list(raw)
                if self._buf and self._buf[-1]["timestamp"] == candle["timestamp"]:
                    # Same-period update → overwrite the last candle in place
                    self._buf[-1] = candle
                else:
                    self._buf.append(candle)

    # ── Accessors ─────────────────────────────────────────────────────────────
    async def snapshot(self) -> List[Dict]:
        """Return a copy of the current buffer as a list (oldest → newest)."""
        async with self._lock:
            return list(self._buf)

    async def last_closed(self, n: int = 1) -> Optional[List[Dict]]:
        """
        Return the last `n` CLOSED candles (excludes the live, still-forming
        candle at index -1 which is the current incomplete period).
        """
        async with self._lock:
            closed = list(self._buf)[:-1]   # drop the live candle
            if len(closed) < n:
                return None
            return closed[-n:]

    async def to_dataframe(self) -> pd.DataFrame:
        """Convert buffer to a pandas DataFrame for indicator calculations."""
        async with self._lock:
            data = list(self._buf)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("datetime").sort_index()
        return df

    def __len__(self) -> int:
        return len(self._buf)

    def __repr__(self) -> str:
        return (
            f"CandleBuffer({self.symbol!r}, {self.timeframe!r}, "
            f"size={len(self._buf)}/{self.max_size})"
        )


# ── Registry of all active buffers ───────────────────────────────────────────
class BufferRegistry:
    """
    Singleton-style registry that maps (symbol, timeframe) → CandleBuffer.
    Created once in main.py and passed to the agents that need it.
    """

    def __init__(self, max_size: int = 200):
        self._max_size = max_size
        self._buffers: Dict[tuple, CandleBuffer] = {}

    def get_or_create(self, symbol: str, timeframe: str) -> CandleBuffer:
        key = (symbol, timeframe)
        if key not in self._buffers:
            self._buffers[key] = CandleBuffer(symbol, timeframe, self._max_size)
        return self._buffers[key]

    def get(self, symbol: str, timeframe: str) -> Optional[CandleBuffer]:
        return self._buffers.get((symbol, timeframe))

    def remove_symbol(self, symbol: str) -> None:
        """Drop all buffers for a symbol (called when the symbol leaves top-N)."""
        keys = [k for k in self._buffers if k[0] == symbol]
        for k in keys:
            del self._buffers[k]

    def all_symbols(self) -> List[str]:
        seen: set = set()
        return [k[0] for k in self._buffers if not (k[0] in seen or seen.add(k[0]))]

    def __repr__(self) -> str:
        return f"BufferRegistry(buffers={len(self._buffers)})"
