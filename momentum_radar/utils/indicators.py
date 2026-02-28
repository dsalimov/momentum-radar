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
