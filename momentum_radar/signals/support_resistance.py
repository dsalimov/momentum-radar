"""
signals/support_resistance.py – Support/resistance touch and failed breakout signals.

Registered signals
------------------
- ``third_touch_support``  – third approach of a horizontal support level
                             (historically high-probability reversal setup)
- ``failed_breakout``      – price breaks a key level then immediately reverses
                             (bull trap / bear trap / liquidity sweep)
- ``resistance_break``     – price closes above a prior resistance level on volume
                             (bullish momentum signal — buyers overtake sellers)
- ``support_break``        – price closes below a prior support level on volume
                             (bearish momentum signal — sellers overtake buyers)
"""

import logging
from typing import List, Optional

import pandas as pd

from momentum_radar.signals.base import SignalResult
from momentum_radar.signals.scoring import register_signal

logger = logging.getLogger(__name__)

# Price tolerance for grouping nearby support/resistance levels (2 %)
_LEVEL_TOLERANCE: float = 0.02
# Minimum number of prior bounces to classify a level as established
_MIN_BOUNCES: int = 2
# How many daily bars to look back when finding levels
_LOOKBACK: int = 60


def _find_local_lows(daily: pd.DataFrame, lookback: int = _LOOKBACK) -> List[float]:
    """Return distinct local-low price levels from the last *lookback* daily bars.

    A bar qualifies as a local low when its low is less than the adjacent bars.
    Nearby levels (within ``_LEVEL_TOLERANCE``) are merged into a single level.

    Args:
        daily:    Daily OHLCV DataFrame.
        lookback: Number of bars to search.

    Returns:
        Sorted list of distinct support price levels.
    """
    if len(daily) < 5:
        return []

    lows = daily["low"].values[-lookback:]
    raw: List[float] = []

    for i in range(1, len(lows) - 1):
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            raw.append(float(lows[i]))

    # Merge nearby levels
    merged: List[float] = []
    for lvl in sorted(raw):
        if not merged or abs(lvl - merged[-1]) / merged[-1] > _LEVEL_TOLERANCE:
            merged.append(lvl)

    return merged


def _count_bounces(daily: pd.DataFrame, level: float) -> int:
    """Count how many times price has touched and bounced from *level*.

    A touch-and-bounce is identified when:
    - The daily low comes within ``_LEVEL_TOLERANCE`` of *level*
    - The subsequent closing price recovers above level × (1 + tolerance/2)

    Args:
        daily: Daily OHLCV DataFrame.
        level: Price level to evaluate.

    Returns:
        Number of confirmed bounces.
    """
    lows = daily["low"].values
    closes = daily["close"].values
    bounces = 0
    in_zone = False

    for i in range(len(lows)):
        near_level = abs(lows[i] - level) / (level if level != 0 else 1) <= _LEVEL_TOLERANCE
        if near_level and not in_zone:
            in_zone = True
            if i + 1 < len(closes) and closes[i + 1] > level * (1 + _LEVEL_TOLERANCE * 0.5):
                bounces += 1
        elif not near_level:
            in_zone = False

    return bounces


@register_signal("third_touch_support")
def third_touch_support(
    ticker: str,
    bars: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
    **kwargs,
) -> SignalResult:
    """Detect the third (or later) touch of a horizontal support level.

    A third touch of a well-established support level is historically a
    high-probability reversal setup because the level has absorbed selling
    pressure at least twice before.

    Score:
    - +2 when ≥ 3 prior bounces confirmed (strong level)
    - +1 when exactly 2 prior bounces (moderate level)

    Args:
        ticker: Stock symbol.
        bars:   Intraday 1-min OHLCV DataFrame (used for current price).
        daily:  Daily OHLCV DataFrame (at least 20 bars).

    Returns:
        :class:`~momentum_radar.signals.base.SignalResult`
    """
    if daily is None or len(daily) < 20:
        return SignalResult(triggered=False, score=0, details="Insufficient daily data")

    support_levels = _find_local_lows(daily)
    if not support_levels:
        return SignalResult(triggered=False, score=0, details="No support levels detected")

    # Use intraday low if available, else daily last-bar low
    if bars is not None and not bars.empty and "low" in bars.columns:
        current_low = float(bars["low"].min())
    elif "low" in daily.columns:
        current_low = float(daily["low"].iloc[-1])
    else:
        return SignalResult(triggered=False, score=0, details="No price data")

    for level in support_levels:
        near_level = level > 0 and abs(current_low - level) / level <= _LEVEL_TOLERANCE
        if not near_level:
            continue

        # Count bounces on all bars *except* the current approach
        bounces = _count_bounces(daily.iloc[:-1], level)
        if bounces >= _MIN_BOUNCES:
            score = 2 if bounces >= 3 else 1
            return SignalResult(
                triggered=True,
                score=score,
                details=(
                    f"Third+ touch of support ${level:.2f} "
                    f"({bounces} prior bounces) — high-probability reversal setup"
                ),
            )

    return SignalResult(
        triggered=False,
        score=0,
        details="No third-touch support pattern detected",
    )


@register_signal("failed_breakout")
def failed_breakout(
    ticker: str,
    bars: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
    **kwargs,
) -> SignalResult:
    """Detect failed breakouts (liquidity traps).

    A failed breakout occurs when price breaks a key level but then immediately
    reverses with a prominent wick, trapping breakout traders.

    - **Bull trap**: price breaks *above* prior-bar high → closes back *below* it
      with a long upper wick and above-average volume.
    - **Bear trap**: price breaks *below* prior-bar low → closes back *above* it
      with a long lower wick and above-average volume.

    Score: +2 on confirmed trap (clear rejection + volume confirmation).

    Args:
        ticker: Stock symbol.
        bars:   Intraday 1-min OHLCV DataFrame.
        daily:  Daily OHLCV DataFrame (at least 5 bars).

    Returns:
        :class:`~momentum_radar.signals.base.SignalResult`
    """
    if daily is None or len(daily) < 5:
        return SignalResult(triggered=False, score=0, details="Insufficient daily data")

    last = daily.iloc[-1]
    prev = daily.iloc[-2]

    last_open = float(last["open"])
    last_close = float(last["close"])
    last_high = float(last["high"])
    last_low = float(last["low"])
    last_vol = float(last["volume"]) if "volume" in daily.columns else 0.0

    prev_high = float(prev["high"])
    prev_low = float(prev["low"])

    # Average volume of the 20 bars before the current bar
    avg_vol = (
        float(daily["volume"].iloc[-21:-1].mean())
        if "volume" in daily.columns and len(daily) >= 21
        else last_vol
    )

    body = abs(last_close - last_open)
    upper_wick = last_high - max(last_open, last_close)
    lower_wick = min(last_open, last_close) - last_low
    total_range = last_high - last_low if last_high > last_low else 1e-9

    # Bull trap: broke above prev high but reversed back below with wick rejection
    if (
        last_high > prev_high
        and last_close < prev_high
        and total_range > 0
        and upper_wick > body * 1.5
        and (avg_vol <= 0 or last_vol >= avg_vol * 0.8)
    ):
        return SignalResult(
            triggered=True,
            score=2,
            details=(
                f"Bull trap: broke above ${prev_high:.2f} but closed ${last_close:.2f} "
                f"— wick rejection, potential reversal (bear signal)"
            ),
        )

    # Bear trap: wicked below prev low but recovered back above it
    if (
        last_low < prev_low
        and last_close > prev_low
        and total_range > 0
        and lower_wick > body * 1.5
        and (avg_vol <= 0 or last_vol >= avg_vol * 0.8)
    ):
        return SignalResult(
            triggered=True,
            score=2,
            details=(
                f"Bear trap: wicked below ${prev_low:.2f} but closed ${last_close:.2f} "
                f"— liquidity sweep, potential bounce (bull signal)"
            ),
        )

    return SignalResult(
        triggered=False,
        score=0,
        details="No failed breakout pattern detected",
    )


def _find_local_highs(daily: pd.DataFrame, lookback: int = _LOOKBACK) -> List[float]:
    """Return distinct local-high price levels from the last *lookback* daily bars.

    A bar qualifies as a local high when its high is greater than the adjacent
    bars.  Nearby levels (within ``_LEVEL_TOLERANCE``) are merged.

    Args:
        daily:    Daily OHLCV DataFrame.
        lookback: Number of bars to search.

    Returns:
        Sorted list of distinct resistance price levels (ascending).
    """
    if len(daily) < 5:
        return []

    highs = daily["high"].values[-lookback:]
    raw: List[float] = []

    for i in range(1, len(highs) - 1):
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            raw.append(float(highs[i]))

    merged: List[float] = []
    for lvl in sorted(raw):
        if not merged or abs(lvl - merged[-1]) / merged[-1] > _LEVEL_TOLERANCE:
            merged.append(lvl)

    return merged


@register_signal("resistance_break")
def resistance_break(
    ticker: str,
    bars: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
    **kwargs,
) -> SignalResult:
    """Detect a confirmed break above a resistance level (bullish signal).

    As described in the problem statement: "As soon as the key resistance
    level was taken out by buyers, sellers stepped out of the picture and
    the stock moved up on strong bullish momentum/buying."

    A resistance break is confirmed when:
    - The current daily close is above a prior resistance level.
    - The prior daily close was below or at that resistance level.
    - Current volume exceeds the 20-bar average.

    Score: +2 on strong-volume break, +1 on moderate-volume break.

    Args:
        ticker: Stock symbol.
        bars:   Intraday 1-min OHLCV DataFrame (unused; daily preferred).
        daily:  Daily OHLCV DataFrame (at least 5 bars).

    Returns:
        :class:`~momentum_radar.signals.base.SignalResult`
    """
    if daily is None or len(daily) < 5:
        return SignalResult(triggered=False, score=0, details="Insufficient daily data")

    resistance_levels = _find_local_highs(daily)
    if not resistance_levels:
        return SignalResult(triggered=False, score=0, details="No resistance levels detected")

    last_close = float(daily["close"].iloc[-1])
    prev_close = float(daily["close"].iloc[-2])

    last_vol = float(daily["volume"].iloc[-1]) if "volume" in daily.columns else 0.0
    avg_vol = (
        float(daily["volume"].iloc[-21:-1].mean())
        if "volume" in daily.columns and len(daily) >= 21
        else last_vol
    )

    for level in resistance_levels:
        above_now = last_close > level
        below_before = prev_close <= level * (1 + _LEVEL_TOLERANCE)
        if above_now and below_before:
            volume_mult = last_vol / avg_vol if avg_vol > 0 else 1.0
            score = 2 if volume_mult >= 1.5 else 1
            return SignalResult(
                triggered=True,
                score=score,
                details=(
                    f"Resistance break: closed ${last_close:.2f} above ${level:.2f} "
                    f"on {volume_mult:.1f}x volume — bullish momentum signal"
                ),
            )

    return SignalResult(
        triggered=False,
        score=0,
        details="No resistance break detected",
    )


@register_signal("support_break")
def support_break(
    ticker: str,
    bars: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
    **kwargs,
) -> SignalResult:
    """Detect a confirmed break below a support level (bearish signal).

    As described in the problem statement: "After the sellers took over
    and the buyers stepped down, the stock fell through [support] and
    collapsed."

    A support break is confirmed when:
    - The current daily close is below a prior support level.
    - The prior daily close was above or at that support level.
    - Current volume exceeds the 20-bar average.

    Score: +2 on strong-volume break, +1 on moderate-volume break.

    Args:
        ticker: Stock symbol.
        bars:   Intraday 1-min OHLCV DataFrame (unused; daily preferred).
        daily:  Daily OHLCV DataFrame (at least 5 bars).

    Returns:
        :class:`~momentum_radar.signals.base.SignalResult`
    """
    if daily is None or len(daily) < 5:
        return SignalResult(triggered=False, score=0, details="Insufficient daily data")

    support_levels = _find_local_lows(daily)
    if not support_levels:
        return SignalResult(triggered=False, score=0, details="No support levels detected")

    last_close = float(daily["close"].iloc[-1])
    prev_close = float(daily["close"].iloc[-2])

    last_vol = float(daily["volume"].iloc[-1]) if "volume" in daily.columns else 0.0
    avg_vol = (
        float(daily["volume"].iloc[-21:-1].mean())
        if "volume" in daily.columns and len(daily) >= 21
        else last_vol
    )

    for level in sorted(support_levels, reverse=True):
        below_now = last_close < level
        above_before = prev_close >= level * (1 - _LEVEL_TOLERANCE)
        if below_now and above_before:
            volume_mult = last_vol / avg_vol if avg_vol > 0 else 1.0
            score = 2 if volume_mult >= 1.5 else 1
            return SignalResult(
                triggered=True,
                score=score,
                details=(
                    f"Support break: closed ${last_close:.2f} below ${level:.2f} "
                    f"on {volume_mult:.1f}x volume — bearish momentum signal"
                ),
            )

    return SignalResult(
        triggered=False,
        score=0,
        details="No support break detected",
    )
