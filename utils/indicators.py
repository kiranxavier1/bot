"""
utils/indicators.py  (v3)
─────────────────────────────────────────────────────────────────────────────
Technical indicator library — all computed natively, no pandas-ta dependency.

Functions
─────────
calc_atr(df, period)             → ATR value  [FIX #4]
calc_atr_pct(df, period)         → ATR as % of price (volatility filter)
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
calc_vwap(df, session_candles)   → float VWAP (24h rolling by default)
calc_stoch_rsi(df, ...)          → (K, D) Stochastic RSI 0-100
detect_rsi_divergence(df, ...)   → dict with divergence details | None
detect_liquidity_sweep(df, ...)  → dict with sweep details | None
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
        return list(reversed(active_fvgs))
    except Exception as exc:
        log.warning("FVG detection failed: %s", exc)
        return []


# ── VWAP (Volume Weighted Average Price) ──────────────────────────────────────
def calc_vwap(df: pd.DataFrame, session_candles: int = None) -> float:
    """
    Rolling VWAP over the last `session_candles` periods.
    Default 96 candles = 24h on 15m timeframe.

    Institutional traders treat VWAP as the 'fair value' intraday anchor.
    Price pulling back to VWAP during an uptrend is one of the highest-
    probability scalping entries because large players re-accumulate there.
    """
    try:
        n = session_candles or getattr(config, "VWAP_SESSION_CANDLES", 96)
        n = min(n, len(df))
        sub = df.iloc[-n:]
        typical = (sub["high"].astype(float) + sub["low"].astype(float) + sub["close"].astype(float)) / 3
        vol = sub["volume"].astype(float)
        total_vol = vol.sum()
        if total_vol <= 0:
            return float(df["close"].iloc[-1])
        return float((typical * vol).sum() / total_vol)
    except Exception as exc:
        log.warning("VWAP calculation failed: %s", exc)
        return 0.0


# ── Stochastic RSI ────────────────────────────────────────────────────────────
def calc_stoch_rsi(
    df: pd.DataFrame,
    rsi_period: int = 14,
    stoch_period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> Tuple[float, float]:
    """
    Stochastic RSI — returns (%K, %D), both scaled 0–100.

    More responsive than plain RSI on volatile coins.  Particularly useful
    for mean-reversion entries: K < 20 and crossing above D = oversold reversal.
    """
    try:
        delta = df["close"].astype(float).diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        rsi_min = rsi.rolling(stoch_period).min()
        rsi_max = rsi.rolling(stoch_period).max()
        stoch = 100 * (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
        k = stoch.rolling(smooth_k).mean()
        d = k.rolling(smooth_d).mean()

        return float(k.fillna(50).iloc[-1]), float(d.fillna(50).iloc[-1])
    except Exception as exc:
        log.warning("Stochastic RSI failed: %s", exc)
        return 50.0, 50.0


# ── RSI Divergence ────────────────────────────────────────────────────────────
def detect_rsi_divergence(
    df: pd.DataFrame,
    rsi_period: int = 14,
    lookback: int = None,
    oversold_level: float = None,
) -> Optional[dict]:
    """
    Detects BULLISH RSI divergence over the last `lookback` candles.

    Bullish divergence = price makes a lower low but RSI makes a higher low.
    This means selling pressure is exhausting even as price falls — a leading
    indicator of reversal.  Most reliable on 15m–4h for swing entries.

    Returns dict with divergence details, or None if not found.
    """
    try:
        lookback = lookback or getattr(config, "RSI_DIVERGENCE_LOOKBACK", 20)
        oversold = oversold_level or getattr(config, "RSI_DIVERGENCE_OVERSOLD", 45)

        window = min(lookback, len(df) - rsi_period - 2)
        if window < 6:
            return None

        close = df["close"].astype(float)
        lows = df["low"].astype(float)

        # Compute RSI over full df for accuracy, then slice to window
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi_series = (100 - (100 / (1 + rs))).fillna(50)

        # Work on the recent window only
        price_vals = lows.iloc[-window:].values
        rsi_vals = rsi_series.iloc[-window:].values

        # Find local price minima (valleys)
        minima = [
            i for i in range(1, len(price_vals) - 1)
            if price_vals[i] < price_vals[i - 1] and price_vals[i] < price_vals[i + 1]
        ]
        if len(minima) < 2:
            return None

        prev_i = minima[-2]
        curr_i = minima[-1]

        price_lower_low = price_vals[curr_i] < price_vals[prev_i]
        rsi_higher_low = rsi_vals[curr_i] > rsi_vals[prev_i]
        rsi_in_oversold = rsi_vals[curr_i] < oversold

        if price_lower_low and rsi_higher_low and rsi_in_oversold:
            return {
                "type": "bullish",
                "price_low1": float(price_vals[prev_i]),
                "price_low2": float(price_vals[curr_i]),
                "rsi_low1": round(float(rsi_vals[prev_i]), 2),
                "rsi_low2": round(float(rsi_vals[curr_i]), 2),
                "current_rsi": round(float(rsi_series.iloc[-1]), 2),
                "rsi_gain": round(float(rsi_vals[curr_i] - rsi_vals[prev_i]), 2),
            }
        return None
    except Exception as exc:
        log.warning("RSI divergence detection failed: %s", exc)
        return None


# ── Liquidity Sweep (Stop Hunt) ───────────────────────────────────────────────
def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 20) -> Optional[dict]:
    """
    Detects a bullish liquidity sweep (stop-hunt below a prior swing low).

    Pattern: institutional players push price below a visible swing low to
    trigger retail stop-losses (buy orders), absorb that liquidity, then
    rapidly reverse upward.  A candle that wicks below the swing low but
    closes back above it is the key signal.

    Returns dict with sweep details if found in the last 3 candles, else None.
    """
    try:
        n = min(lookback, len(df))
        sub = df.iloc[-n:]
        lows = sub["low"].astype(float).values
        closes = sub["close"].astype(float).values

        if len(lows) < 6:
            return None

        # Find the most recent swing low in the PRIOR section (exclude last 3 candles)
        prior_lows = lows[:-3]
        swing_low = None
        for i in range(len(prior_lows) - 1, 0, -1):
            left_ok = i == 0 or prior_lows[i] < prior_lows[i - 1]
            right_ok = (i + 1 >= len(prior_lows)) or prior_lows[i] < prior_lows[i + 1]
            if left_ok and right_ok:
                swing_low = prior_lows[i]
                break

        if swing_low is None:
            return None

        # Check last 3 candles for sweep + reclaim
        for low, close in zip(lows[-3:], closes[-3:]):
            if low < swing_low and close > swing_low:
                sweep_depth_pct = (swing_low - low) / swing_low * 100
                if sweep_depth_pct >= 0.1:
                    return {
                        "swing_low": float(swing_low),
                        "sweep_low": float(low),
                        "reclaim_close": float(close),
                        "sweep_depth_pct": round(sweep_depth_pct, 3),
                    }
        return None
    except Exception as exc:
        log.warning("Liquidity sweep detection failed: %s", exc)
        return None


# ── ATR as % of price ─────────────────────────────────────────────────────────
def calc_atr_pct(df: pd.DataFrame, period: int = None) -> float:
    """
    ATR expressed as a percentage of current price.
    Used as a relative volatility filter for coin selection:
    only trade coins with enough intraday range to capture scalp/swing gains.
    """
    atr = calc_atr(df, period)
    price = float(df["close"].iloc[-1])
    if price <= 0:
        return 0.0
    return (atr / price) * 100

