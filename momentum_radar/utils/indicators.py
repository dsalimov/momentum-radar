"""
indicators.py – Technical indicator calculations.

Pure functions that operate on pandas DataFrames; no external dependencies
beyond pandas and numpy.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_atr(daily: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Compute the Average True Range over the last *period* days.

    True Range = max(high − low, |high − prev_close|, |low − prev_close|)

    Args:
        daily: Daily OHLCV DataFrame with ``high``, ``low``, ``close`` columns.
        period: Look-back period in days.

    Returns:
        ATR as a float, or ``None`` if there is insufficient data.
    """
    required = {"high", "low", "close"}
    if not required.issubset(daily.columns):
        logger.warning("ATR requires 'high', 'low', 'close' columns.")
        return None
    if len(daily) < 2:
        return None

    df = daily[["high", "low", "close"]].copy().tail(period + 1)
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    tr = tr.dropna()
    if len(tr) < period:
        return float(tr.mean()) if not tr.empty else None
    return float(tr.tail(period).mean())


def compute_vwap(bars: pd.DataFrame) -> Optional[float]:
    """Compute the Volume-Weighted Average Price for the given bars.

    VWAP = cumsum(typical_price × volume) / cumsum(volume)
    where typical_price = (high + low + close) / 3

    Args:
        bars: Intraday OHLCV DataFrame with ``high``, ``low``, ``close``,
            ``volume`` columns.

    Returns:
        Current VWAP as a float, or ``None`` if computation is not possible.
    """
    required = {"high", "low", "close", "volume"}
    if not required.issubset(bars.columns):
        return None
    if bars.empty:
        return None

    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    cum_vol = bars["volume"].cumsum()
    cum_tp_vol = (typical * bars["volume"]).cumsum()

    if cum_vol.iloc[-1] == 0:
        return None
    return float(cum_tp_vol.iloc[-1] / cum_vol.iloc[-1])


def compute_rvol(
    bars: pd.DataFrame,
    daily: pd.DataFrame,
    lookback_days: int = 30,
) -> Optional[float]:
    """Compute Relative Volume (RVOL).

    RVOL = current cumulative intraday volume / 30-day average daily volume

    Args:
        bars: Intraday OHLCV DataFrame.
        daily: Daily OHLCV DataFrame.
        lookback_days: Number of trading days for the average.

    Returns:
        RVOL as a float, or ``None`` if data is insufficient.
    """
    if bars is None or bars.empty or "volume" not in bars.columns:
        return None
    if daily is None or daily.empty or "volume" not in daily.columns:
        return None

    avg_daily = daily["volume"].iloc[-(lookback_days + 1):-1].mean()
    if avg_daily <= 0:
        return None
    current_vol = float(bars["volume"].sum())
    return current_vol / avg_daily


def compute_ema(series: pd.Series, period: int) -> Optional[pd.Series]:
    """Compute Exponential Moving Average."""
    if series is None or len(series) < period:
        return None
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(closes: pd.Series, period: int = 14) -> Optional[float]:
    """Compute RSI (Relative Strength Index) for the most recent bar."""
    if closes is None or len(closes) < period + 1:
        return None
    delta = closes.diff().dropna()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    val = float(rsi.iloc[-1])
    return val if not pd.isna(val) else None


def compute_macd(
    closes: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Optional[dict]:
    """Compute MACD line, signal line, and histogram.

    Returns dict with keys ``macd``, ``signal``, ``histogram``, or ``None``
    if there is insufficient data.
    """
    if closes is None or len(closes) < slow + signal:
        return None
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return {
        "macd": float(macd_line.iloc[-1]),
        "signal": float(signal_line.iloc[-1]),
        "histogram": float(histogram.iloc[-1]),
    }


def compute_bollinger_bands(
    closes: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> Optional[dict]:
    """Compute Bollinger Bands (upper, middle, lower).

    Returns dict with keys ``upper``, ``middle``, ``lower``, or ``None``
    if there is insufficient data.
    """
    if closes is None or len(closes) < period:
        return None
    middle = closes.rolling(period).mean()
    std = closes.rolling(period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return {
        "upper": float(upper.iloc[-1]),
        "middle": float(middle.iloc[-1]),
        "lower": float(lower.iloc[-1]),
    }
