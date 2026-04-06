"""
utils/notifications.py
─────────────────────────────────────────────────────────────────────────────
Async Telegram Bot notification helper.

Usage:
    from utils.notifications import Notifier
    notifier = Notifier()
    await notifier.send("✅ Trade opened: BTC/USDT @ 67,500")

The notifier silently swallows errors so a Telegram outage never crashes
the bot.  All errors are logged at WARNING level.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

import config

log = logging.getLogger(__name__)

_TELEGRAM_URL = "https://api.telegram.org/bot{token}/sendMessage"


class Notifier:
    """Lightweight async Telegram notifier."""

    def __init__(
        self,
        token: str = config.TELEGRAM_BOT_TOKEN,
        chat_id: str = config.TELEGRAM_CHAT_ID,
    ):
        self._token   = token
        self._chat_id = chat_id
        self._url     = _TELEGRAM_URL.format(token=token)
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def send(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Send a message to the configured Telegram chat.

        Returns True on success, False on failure.
        Messages longer than 4096 chars are automatically split.
        """
        chunks = [text[i : i + 4096] for i in range(0, len(text), 4096)]
        ok = True
        for chunk in chunks:
            ok = ok and await self._send_chunk(chunk, parse_mode)
        return ok

    async def _send_chunk(self, text: str, parse_mode: str) -> bool:
        try:
            client = await self._get_client()
            resp = await client.post(
                self._url,
                json={
                    "chat_id":    self._chat_id,
                    "text":       text,
                    "parse_mode": parse_mode,
                },
            )
            resp.raise_for_status()
            return True
        except Exception as exc:
            log.warning("Telegram send failed: %s", exc)
            return False

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── Convenience formatters ────────────────────────────────────────────────
    @staticmethod
    def trade_opened_msg(symbol: str, side: str, entry: float,
                         sl: float, tp: float, size: float) -> str:
        return (
            f"🟢 <b>TRADE OPENED</b>\n"
            f"Symbol : <code>{symbol}</code>\n"
            f"Side   : <code>{side.upper()}</code>\n"
            f"Entry  : <code>{entry:.6g}</code>\n"
            f"SL     : <code>{sl:.6g}</code>  "
            f"({abs(entry - sl) / entry * 100:.2f}%)\n"
            f"TP     : <code>{tp:.6g}</code>  "
            f"({abs(tp - entry) / entry * 100:.2f}%)\n"
            f"Size   : <code>{size:.6g}</code>"
        )

    @staticmethod
    def trade_closed_msg(symbol: str, reason: str,
                         entry: float, exit_price: float,
                         pnl_pct: float) -> str:
        emoji = "✅" if pnl_pct >= 0 else "❌"
        return (
            f"{emoji} <b>TRADE CLOSED — {reason.upper()}</b>\n"
            f"Symbol  : <code>{symbol}</code>\n"
            f"Entry   : <code>{entry:.6g}</code>\n"
            f"Exit    : <code>{exit_price:.6g}</code>\n"
            f"P&L     : <code>{pnl_pct:+.2f}%</code>"
        )

    @staticmethod
    def watcher_alert_msg(symbol: str, timeframe: str,
                          current_price: float, trendline_price: float,
                          proximity_pct: float) -> str:
        return (
            f"👁️ <b>WATCHER ARMED</b>\n"
            f"Symbol     : <code>{symbol}</code>\n"
            f"Timeframe  : <code>{timeframe}</code>\n"
            f"Price      : <code>{current_price:.6g}</code>\n"
            f"Trendline  : <code>{trendline_price:.6g}</code>\n"
            f"Proximity  : <code>{proximity_pct:.3f}%</code>"
        )

    @staticmethod
    def ai_rejected_msg(symbol: str, reason: str, confidence: float) -> str:
        return (
            f"🤖 <b>AI REJECTED TRADE</b>\n"
            f"Symbol     : <code>{symbol}</code>\n"
            f"Confidence : <code>{confidence:.2f}</code>\n"
            f"Reason     : {reason}"
        )

    @staticmethod
    def error_msg(context: str, exc: Exception) -> str:
        return (
            f"⚠️ <b>BOT ERROR</b>\n"
            f"Context : {context}\n"
            f"Error   : <code>{type(exc).__name__}: {exc}</code>"
        )
