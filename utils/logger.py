"""
utils/logger.py
─────────────────────────────────────────────────────────────────────────────
Configures a root logger that writes to:
  1. A rotating file  (bot.log  by default)
  2. Coloured console output

Call `setup_logging()` once at startup (main.py).  Every other module then
just does:

    import logging
    log = logging.getLogger(__name__)
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import logging.handlers
import sys
from pathlib import Path

try:
    import colorlog  # type: ignore
    _HAS_COLORLOG = True
except ImportError:
    _HAS_COLORLOG = False

import config


def setup_logging() -> None:
    """Configure the root logger.  Call this once at bot start-up."""
    log_level = getattr(logging, config.LOG_LEVEL, logging.INFO)
    root = logging.getLogger()
    root.setLevel(log_level)

    # Silence noisy third-party loggers
    for noisy in ("ccxt", "asyncio", "urllib3", "httpx", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    fmt_str = (
        "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
    )
    date_fmt = "%Y-%m-%d %H:%M:%S"

    # ── Console handler ──────────────────────────────────────────────────────
    if _HAS_COLORLOG:
        color_fmt = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s | %(levelname)-8s%(reset)s | "
            "%(cyan)s%(name)-30s%(reset)s | %(message)s",
            datefmt=date_fmt,
            log_colors={
                "DEBUG":    "white",
                "INFO":     "green",
                "WARNING":  "yellow",
                "ERROR":    "red",
                "CRITICAL": "bold_red",
            },
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(color_fmt)
    else:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter(fmt_str, datefmt=date_fmt))

    root.addHandler(ch)

    # ── Rotating file handler ────────────────────────────────────────────────
    log_path = Path(config.LOG_FILE)
    fh = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,   # 10 MB per file
        backupCount=5,
        encoding="utf-8",
    )
    fh.setFormatter(logging.Formatter(fmt_str, datefmt=date_fmt))
    root.addHandler(fh)

    root.info("Logging initialised  →  file=%s  level=%s", log_path, config.LOG_LEVEL)
