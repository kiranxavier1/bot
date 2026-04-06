"""
utils/indicators.py  (v2)
─────────────────────────────────────────────────────────────────────────────
Technical indicator library — all computed natively, no pandas-ta dependency.

Functions
─────────
calc_atr(df, period)             → ATR value  [FIX #4]
calc_adx_full(df, period)        → (ADX, DI+, DI-)  [FIX #3]
calc_rsi(df, period)             → latest RSI
calc_ema(series, period)         → EMA Series
price_above_ema200(df_htf)       → bool  [FIX #2]
volume_ratio(candle, df)         → float (candle vol / avg vol)  [FIX #1]
btc_trend(df_1h)                 → "bullish" | "bearish" | "neutral"
market_regime(df)                → ("trending"|"ranging", adx, di_plus, di_minus)
fetch_news_sentiment(ccy, key)   → bool
calc_bollinger_bands(df, p, std) → (upper, mid, lower)
detect_fvg(df, lookback)         → List of active Bullish FVGs
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)


# ── EMA (Wilder-compatible, used throughout) ──────────────────────────────────
def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ── ATR (FIX #4) ──────────────────────────────────────────────────────────────
def calc_atr(df: pd.DataFrame, period: int = None) -> float:
    """
    Average True Range — used for ATR-based stop loss placement.
    Returns the most recent ATR value, or 0.0 on failure.
    """
    period = period or config.ATR_PERIOD
    try:
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        close = df["close"].astype(float)
        prev_close = close.shift(1)

        tr = pd.concat(
            [high - low,
             (high - prev_close).abs(),
             (low  - prev_close).abs()],
            axis=1,
        ).max(axis=1)

        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        return float(atr.iloc[-1])
    except Exception as exc:
        log.warning("ATR calculation failed: %s", exc)
        return 0.0


# ── Full ADX with DI+ and DI- (FIX #3) ───────────────────────────────────────
def calc_adx_full(df: pd.DataFrame, period: int = 14) -> Tuple[float, float, float]:
    """
    Returns (ADX, DI+, DI-).
    DI+ > DI- confirms upward trend direction.
    Wilder smoothing used throughout.
    Returns (0, 0, 0) on failure.
    """
    try:
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        close = df["close"].astype(float)
        prev_close = close.shift(1)

        tr = pd.concat(
            [high - low,
             (high - prev_close).abs(),
             (low  - prev_close).abs()],
            axis=1,
        ).max(axis=1)

        up_move   = high.diff()
        down_move = -low.diff()
        plus_dm  = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=df.index,
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=df.index,
        )

        alpha = 1 / period
        atr_s   = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_s  = plus_dm.ewm(alpha=alpha, adjust=False).mean()
        minus_s = minus_dm.ewm(alpha=alpha, adjust=False).mean()

        plus_di  = 100 * plus_s  / atr_s.replace(0, np.nan)
        minus_di = 100 * minus_s / atr_s.replace(0, np.nan)

        dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.fillna(0).ewm(alpha=alpha, adjust=False).mean()

        return (
            float(adx.iloc[-1]),
            float(plus_di.iloc[-1]),
            float(minus_di.iloc[-1]),
        )
    except Exception as exc:
        log.warning("ADX full calculation failed: %s", exc)
        return (0.0, 0.0, 0.0)


def calc_adx(df: pd.DataFrame, period: int = 14) -> float:
    """Backward-compatible wrapper — returns ADX only."""
    adx, _, _ = calc_adx_full(df, period)
    return adx


# ── RSI ───────────────────────────────────────────────────────────────────────
def calc_rsi(df: pd.DataFrame, period: int = 14) -> float:
    try:
        delta    = df["close"].astype(float).diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs  = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])
    except Exception as exc:
        log.warning("RSI calculation failed: %s", exc)
        return 50.0


# ── 200 EMA higher-timeframe filter (FIX #2) ─────────────────────────────────
def price_above_ema200(df_htf: pd.DataFrame, period: int = None) -> bool:
    """
    Returns True if the coin's latest close is above its EMA-200 on the
    higher timeframe (1h).  This ensures we only buy in macro uptrends.
    """
    period = period or config.HTF_EMA_PERIOD
    try:
        close = df_htf["close"].astype(float)
        ema   = calc_ema(close, period)
        result = float(close.iloc[-1]) > float(ema.iloc[-1])
        log.debug("EMA%d filter: close=%.6g ema=%.6g → %s",
                  period, close.iloc[-1], ema.iloc[-1], result)
        return result
    except Exception as exc:
        log.warning("EMA-200 filter failed: %s — defaulting to True", exc)
        return True   # fail-open: don't block on calculation error


# ── Volume confirmation ratio (FIX #1) ───────────────────────────────────────
def volume_ratio(candle: dict, df: pd.DataFrame) -> float:
    """
    Returns candle_volume / avg_volume over the last VOLUME_LOOKBACK periods.
    A value ≥ VOLUME_CONFIRM_MULTIPLIER (default 1.3) = confirmed bounce.
    """
    try:
        avg = df["volume"].astype(float).iloc[-config.VOLUME_LOOKBACK:].mean()
        if avg <= 0:
            return 1.0
        return float(candle.get("volume", 0)) / avg
    except Exception as exc:
        log.warning("Volume ratio failed: %s", exc)
        return 1.0


# ── BTC 1h trend ──────────────────────────────────────────────────────────────
def btc_trend(df_1h: pd.DataFrame) -> str:
    """
    EMA-20 vs EMA-50 crossover on BTC 1h.
    Returns "bullish" | "bearish" | "neutral".
    """
    try:
        close  = df_1h["close"].astype(float)
        ema20  = calc_ema(close, 20)
        ema50  = calc_ema(close, 50)
        last_c = float(close.iloc[-1])
        last20 = float(ema20.iloc[-1])
        last50 = float(ema50.iloc[-1])
        if last20 > last50 and last_c > last20:
            return "bullish"
        if last20 < last50 and last_c < last20:
            return "bearish"
        return "neutral"
    except Exception as exc:
        log.warning("BTC trend failed: %s", exc)
        return "neutral"


# ── Market regime (FIX #3 — now returns direction too) ───────────────────────
def market_regime(
    df: pd.DataFrame,
    adx_threshold: int = None,
) -> Tuple[str, float, float, float]:
    """
    Returns (regime, adx, di_plus, di_minus).
      regime = "trending_up" | "trending_down" | "ranging"

    "trending_up"   → ADX ≥ threshold AND DI+ > DI-
    "trending_down" → ADX ≥ threshold AND DI- > DI+
    "ranging"       → ADX < threshold
    """
    threshold = adx_threshold or config.ADX_TREND_THRESHOLD
    adx, di_plus, di_minus = calc_adx_full(df)

    if adx >= threshold:
        regime = "trending_up" if di_plus > di_minus else "trending_down"
    else:
        regime = "ranging"

    log.debug(
        "Market regime: ADX=%.1f DI+=%.1f DI-=%.1f → %s",
        adx, di_plus, di_minus, regime,
    )
    return regime, adx, di_plus, di_minus


# ── CryptoPanic news filter ───────────────────────────────────────────────────
def fetch_news_sentiment(base_currency: str, api_key: Optional[str] = None) -> bool:
    """
    Query CryptoPanic for recent high-impact news about `base_currency`.
    Returns True = safe, False = negative news found.
    Skipped (returns True) if no API key is configured.
    """
    if not api_key:
        return True
    try:
        import requests  # pylint: disable=import-outside-toplevel
        url = (
            f"https://cryptopanic.com/api/v1/posts/"
            f"?auth_token={api_key}&currencies={base_currency}"
            f"&filter=important&kind=news&public=true"
        )
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        for item in resp.json().get("results", []):
            title = (item.get("title") or "").lower()
            if any(kw in title for kw in config.NEGATIVE_KEYWORDS):
                log.warning("⚠️  Negative news for %s: %r", base_currency, title)
                return False
        return True
    except Exception as exc:
        log.warning("News fetch failed for %s: %s — defaulting safe", base_currency, exc)
        return True


# ── Bollinger Bands ───────────────────────────────────────────────────────────
def calc_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float]:
    """
    Returns the most recent (Upper, Middle, Lower) Bollinger Band values.
    Used for Mean Reversion strategy.
    """
    try:
        close = df["close"].astype(float)
        mid = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = mid + (std * std_dev)
        lower = mid - (std * std_dev)
        return float(upper.iloc[-1]), float(mid.iloc[-1]), float(lower.iloc[-1])
    except Exception as exc:
        log.warning("Bollinger Bands failed: %s", exc)
        return 0.0, 0.0, 0.0


# ── Fair Value Gaps (SMC) ─────────────────────────────────────────────────────
def detect_fvg(df: pd.DataFrame, lookback: int = 20) -> list[dict]:
    """
    Detects recent Bullish Fair Value Gaps (FVG) / Imbalances.
    A bullish FVG occurs over 3 candles when Candle 1 High < Candle 3 Low.
    Returns a list of active gap zones (not yet mitigated/filled by price).
    """
    try:
        highs = df["high"].astype(float).values
        lows = df["low"].astype(float).values
        closes = df["close"].astype(float).values
        timestamps = df["timestamp"].values

        n = len(df)
        active_fvgs = []

        start_idx = max(0, n - lookback)
        # We need sequences of 3 candles (i-2, i-1, i)
        for i in range(start_idx + 2, n):
            c1_high = highs[i-2]
            c3_low = lows[i]

            # Bullish FVG: Strong move up leaving a gap between C1 high and C3 low
            if c3_low > c1_high:
                gap_top = c3_low
                gap_bottom = c1_high
                gap_size = (gap_top - gap_bottom) / gap_bottom

                # Only consider meaningful gaps (>0.1%)
                if gap_size > 0.001:
                    # Check if mitigated by any candle AFTER candle 3
                    mitigated = False
                    for j in range(i + 1, n):
                        if lows[j] <= gap_bottom:
                            mitigated = True
                            break
                    
                    if not mitigated:
                        active_fvgs.append({
                            "top": gap_top,
                            "bottom": gap_bottom,
                            "mid": (gap_top + gap_bottom) / 2,
                            "timestamp": int(timestamps[i-1]), # The impulse candle
                            "size_pct": round(gap_size * 100, 3)
                        })

        # Return the most recent FVGs first
        return active_fvgs[::-1]
    except Exception as exc:
        log.warning("FVG detection failed: %s", exc)
        return []

