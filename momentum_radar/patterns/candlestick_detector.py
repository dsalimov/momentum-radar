"""
candlestick_detector.py - Candlestick pattern detection engine.

Detects single, double, and triple candlestick reversal and continuation
patterns. Patterns are detected on the most recent candles (last 1-3 bars).

Pattern types
-------------
Single candle  : Hammer, Inverted Hammer, Hanging Man, Shooting Star,
                 Doji (Regular / Dragonfly / Gravestone),
                 Bullish/Bearish Marubozu, Spinning Top
Two candle     : Bullish/Bearish Engulfing, Bullish/Bearish Harami,
                 Tweezer Top/Bottom, Piercing Line, Dark Cloud Cover
Three candle   : Morning/Evening Star, Three White Soldiers,
                 Three Black Crows, Three Inside Up/Down
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _candle_props(
    o: float, h: float, l: float, c: float
) -> Tuple[float, float, float, float, float, bool, bool]:
    """Return (body, upper_shadow, lower_shadow, total_range, body_ratio,
    is_bullish, is_bearish)."""
    body = abs(c - o)
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l
    total_range = h - l
    body_ratio = body / total_range if total_range > 0 else 0.0
    return body, upper_shadow, lower_shadow, total_range, body_ratio, c > o, c < o


def _trend_context(close: np.ndarray, n: int) -> str:
    """Return 'uptrend', 'downtrend', or 'neutral' based on 5-bar comparison."""
    if n < 6:
        return "neutral"
    ref = close[n - 6]
    if ref == 0:
        return "neutral"
    pct = (close[n - 1] - ref) / abs(ref)
    if pct >= 0.02:
        return "uptrend"
    if pct <= -0.02:
        return "downtrend"
    return "neutral"


def _make_result(
    pattern: str,
    bias: str,
    confidence: float,
    key_points: List[Tuple],
    candle_indices: List[int],
    description: str,
) -> Dict:
    return {
        "pattern": pattern,
        "pattern_type": "candlestick",
        "confidence": round(confidence, 1),
        "bias": bias,
        "direction": bias.upper(),  # "BULLISH" | "BEARISH" — normalised for alert formatter
        "key_points": key_points,
        "lines": [],
        "candle_indices": candle_indices,
        "description": description,
        "state": "forming",
        "breakout_level_upper": None,
        "breakout_level_lower": None,
        "compression_ratio": None,
        "distance_to_breakout": None,
    }


# ---------------------------------------------------------------------------
# Single-candle patterns
# ---------------------------------------------------------------------------


def _detect_hammer(df: pd.DataFrame) -> Optional[Dict]:
    n = len(df)
    if n < 6:
        return None
    trend = _trend_context(df["close"].values, n)
    if trend != "downtrend":
        return None

    idx = n - 1
    o = float(df["open"].iloc[idx])
    h = float(df["high"].iloc[idx])
    l = float(df["low"].iloc[idx])
    c = float(df["close"].iloc[idx])
    body, upper_shadow, lower_shadow, total_range, _, _, _ = _candle_props(o, h, l, c)

    if total_range <= 0 or body <= 0:
        return None
    if lower_shadow < 2 * body:
        return None
    if upper_shadow > 0.1 * total_range:
        return None

    shadow_ratio = lower_shadow / body
    confidence = min(95.0, 65.0 + shadow_ratio * 3)

    pct_decline = (
        (df["close"].iloc[idx] - df["close"].iloc[n - 6]) / df["close"].iloc[n - 6] * 100
    )
    vol = df["volume"].values if "volume" in df.columns else None
    vol_note = ""
    if vol is not None and idx >= 20:
        avg_vol = float(np.mean(vol[max(0, idx - 20):idx]))
        if avg_vol > 0:
            vol_ratio = vol[idx] / avg_vol
            if vol_ratio >= 1.5:
                confidence = min(100.0, confidence + 5)
            vol_note = f"\nVolume: {vol_ratio:.1f}x 20-bar average"

    description = (
        f"BULLISH REVERSAL: Hammer candle at support after {pct_decline:.1f}% decline\n"
        f"O ${o:.2f} H ${h:.2f} L ${l:.2f} C ${c:.2f}\n"
        f"Body: ${body:.2f} | Lower shadow: ${lower_shadow:.2f} ({shadow_ratio:.1f}x body)"
        f" | Upper shadow: ${upper_shadow:.2f}{vol_note}\n"
        f"Prior trend: {pct_decline:.1f}% decline over 5 bars - reversal setup valid\n"
        f"Key level: Hammer low at ${l:.2f} acts as support\n"
        f"Watch for: Bullish follow-through above ${h:.2f} to confirm"
    )
    date = df.index[idx]
    return _make_result(
        "Hammer", "bullish", confidence,
        [(date, l, "Hammer Low"), (date, c, "Hammer Close")],
        [idx],
        description,
    )


def _detect_inverted_hammer(df: pd.DataFrame) -> Optional[Dict]:
    n = len(df)
    if n < 6:
        return None
    trend = _trend_context(df["close"].values, n)
    if trend != "downtrend":
        return None

    idx = n - 1
    o = float(df["open"].iloc[idx])
    h = float(df["high"].iloc[idx])
    l = float(df["low"].iloc[idx])
    c = float(df["close"].iloc[idx])
    body, upper_shadow, lower_shadow, total_range, _, _, _ = _candle_props(o, h, l, c)

    if total_range <= 0 or body <= 0:
        return None
    if upper_shadow < 2 * body:
        return None
    if lower_shadow > 0.1 * total_range:
        return None

    shadow_ratio = upper_shadow / body
    confidence = min(95.0, 60.0 + shadow_ratio * 3)
    pct_decline = (
        (df["close"].iloc[idx] - df["close"].iloc[n - 6]) / df["close"].iloc[n - 6] * 100
    )
    date = df.index[idx]
    description = (
        f"BULLISH REVERSAL: Inverted Hammer after {pct_decline:.1f}% decline\n"
        f"O ${o:.2f} H ${h:.2f} L ${l:.2f} C ${c:.2f}\n"
        f"Upper shadow: {shadow_ratio:.1f}x body - potential buying pressure\n"
        f"Watch for: Confirmation on next bar above ${h:.2f}"
    )
    return _make_result(
        "Inverted Hammer", "bullish", confidence,
        [(date, l, "Pattern Low"), (date, h, "Pattern High")],
        [idx],
        description,
    )


def _detect_hanging_man(df: pd.DataFrame) -> Optional[Dict]:
    n = len(df)
    if n < 6:
        return None
    trend = _trend_context(df["close"].values, n)
    if trend != "uptrend":
        return None

    idx = n - 1
    o = float(df["open"].iloc[idx])
    h = float(df["high"].iloc[idx])
    l = float(df["low"].iloc[idx])
    c = float(df["close"].iloc[idx])
    body, upper_shadow, lower_shadow, total_range, _, _, _ = _candle_props(o, h, l, c)

    if total_range <= 0 or body <= 0:
        return None
    if lower_shadow < 2 * body:
        return None
    if upper_shadow > 0.1 * total_range:
        return None

    shadow_ratio = lower_shadow / body
    confidence = min(95.0, 62.0 + shadow_ratio * 2)
    pct_rise = (
        (df["close"].iloc[idx] - df["close"].iloc[n - 6]) / df["close"].iloc[n - 6] * 100
    )
    date = df.index[idx]
    description = (
        f"BEARISH REVERSAL WARNING: Hanging Man after {pct_rise:.1f}% rise\n"
        f"O ${o:.2f} H ${h:.2f} L ${l:.2f} C ${c:.2f}\n"
        f"Long lower shadow ({shadow_ratio:.1f}x body) shows selling pressure\n"
        f"Watch for: Bearish confirmation on next candle below ${l:.2f}"
    )
    return _make_result(
        "Hanging Man", "bearish", confidence,
        [(date, h, "Hanging Man High"), (date, l, "Shadow Low")],
        [idx],
        description,
    )


def _detect_shooting_star(df: pd.DataFrame) -> Optional[Dict]:
    n = len(df)
    if n < 6:
        return None
    trend = _trend_context(df["close"].values, n)
    if trend != "uptrend":
        return None

    idx = n - 1
    o = float(df["open"].iloc[idx])
    h = float(df["high"].iloc[idx])
    l = float(df["low"].iloc[idx])
    c = float(df["close"].iloc[idx])
    body, upper_shadow, lower_shadow, total_range, _, _, _ = _candle_props(o, h, l, c)

    if total_range <= 0 or body <= 0:
        return None
    if upper_shadow < 2 * body:
        return None
    if lower_shadow > 0.1 * total_range:
        return None

    shadow_ratio = upper_shadow / body
    confidence = min(95.0, 65.0 + shadow_ratio * 2)
    pct_rise = (
        (df["close"].iloc[idx] - df["close"].iloc[n - 6]) / df["close"].iloc[n - 6] * 100
    )
    date = df.index[idx]
    description = (
        f"BEARISH REVERSAL: Shooting Star after {pct_rise:.1f}% rise\n"
        f"O ${o:.2f} H ${h:.2f} L ${l:.2f} C ${c:.2f}\n"
        f"Upper shadow {shadow_ratio:.1f}x body - strong rejection of highs\n"
        f"Watch for: Bearish follow-through below ${l:.2f}"
    )
    return _make_result(
        "Shooting Star", "bearish", confidence,
        [(date, h, "Star High"), (date, l, "Pattern Low")],
        [idx],
        description,
    )


def _detect_doji(df: pd.DataFrame) -> Optional[Dict]:
    n = len(df)
    if n < 2:
        return None

    idx = n - 1
    o = float(df["open"].iloc[idx])
    h = float(df["high"].iloc[idx])
    l = float(df["low"].iloc[idx])
    c = float(df["close"].iloc[idx])
    body, upper_shadow, lower_shadow, total_range, body_ratio, _, _ = _candle_props(o, h, l, c)

    if total_range <= 0 or body_ratio >= 0.10:
        return None

    if lower_shadow > 2 * upper_shadow and upper_shadow < 0.05 * total_range:
        subtype = "Dragonfly Doji"
        bias = "bullish"
        desc_extra = "Long lower shadow with no upper shadow - potential bullish reversal"
    elif upper_shadow > 2 * lower_shadow and lower_shadow < 0.05 * total_range:
        subtype = "Gravestone Doji"
        bias = "bearish"
        desc_extra = "Long upper shadow with no lower shadow - potential bearish reversal"
    else:
        subtype = "Doji"
        bias = "neutral"
        desc_extra = "Equal shadows - indecision, watch for directional break"

    date = df.index[idx]
    description = (
        f"INDECISION: {subtype} candle - {desc_extra}\n"
        f"O ${o:.2f} H ${h:.2f} L ${l:.2f} C ${c:.2f}\n"
        f"Body ratio: {body_ratio * 100:.1f}% of range\n"
        f"Upper shadow: ${upper_shadow:.2f} | Lower shadow: ${lower_shadow:.2f}"
    )
    return _make_result(
        subtype, bias, 65.0,
        [(date, l, "Doji Low"), (date, h, "Doji High")],
        [idx],
        description,
    )


def _detect_dragonfly_doji(df: pd.DataFrame) -> Optional[Dict]:
    n = len(df)
    if n < 6:
        return None
    trend = _trend_context(df["close"].values, n)
    if trend != "downtrend":
        return None

    idx = n - 1
    o = float(df["open"].iloc[idx])
    h = float(df["high"].iloc[idx])
    l = float(df["low"].iloc[idx])
    c = float(df["close"].iloc[idx])
    body, upper_shadow, lower_shadow, total_range, body_ratio, _, _ = _candle_props(o, h, l, c)

    if total_range <= 0 or body_ratio >= 0.10:
        return None
    if not (lower_shadow > 2 * upper_shadow and upper_shadow < 0.05 * total_range):
        return None

    pct_decline = (
        (df["close"].iloc[idx] - df["close"].iloc[n - 6]) / df["close"].iloc[n - 6] * 100
    )
    date = df.index[idx]
    description = (
        f"BULLISH REVERSAL: Dragonfly Doji after {pct_decline:.1f}% decline\n"
        f"O ${o:.2f} H ${h:.2f} L ${l:.2f} C ${c:.2f}\n"
        f"Long lower shadow with tiny/no upper shadow - strong rejection of lows\n"
        f"Watch for: Bullish follow-through above ${h:.2f}"
    )
    return _make_result(
        "Dragonfly Doji", "bullish", 72.0,
        [(date, l, "Dragon Low"), (date, h, "Dragon High")],
        [idx],
        description,
    )


def _detect_gravestone_doji(df: pd.DataFrame) -> Optional[Dict]:
    n = len(df)
    if n < 6:
        return None
    trend = _trend_context(df["close"].values, n)
    if trend != "uptrend":
        return None

    idx = n - 1
    o = float(df["open"].iloc[idx])
    h = float(df["high"].iloc[idx])
    l = float(df["low"].iloc[idx])
    c = float(df["close"].iloc[idx])
    body, upper_shadow, lower_shadow, total_range, body_ratio, _, _ = _candle_props(o, h, l, c)

    if total_range <= 0 or body_ratio >= 0.10:
        return None
    if not (upper_shadow > 2 * lower_shadow and lower_shadow < 0.05 * total_range):
        return None

    pct_rise = (
        (df["close"].iloc[idx] - df["close"].iloc[n - 6]) / df["close"].iloc[n - 6] * 100
    )
    date = df.index[idx]
    description = (
        f"BEARISH REVERSAL: Gravestone Doji after {pct_rise:.1f}% rise\n"
        f"O ${o:.2f} H ${h:.2f} L ${l:.2f} C ${c:.2f}\n"
        f"Long upper shadow with tiny/no lower shadow - strong rejection of highs\n"
        f"Watch for: Bearish follow-through below ${l:.2f}"
    )
    return _make_result(
        "Gravestone Doji", "bearish", 72.0,
        [(date, h, "Gravestone High"), (date, l, "Gravestone Low")],
        [idx],
        description,
    )


def _detect_bullish_marubozu(df: pd.DataFrame) -> Optional[Dict]:
    n = len(df)
    if n < 1:
        return None

    idx = n - 1
    o = float(df["open"].iloc[idx])
    h = float(df["high"].iloc[idx])
    l = float(df["low"].iloc[idx])
    c = float(df["close"].iloc[idx])
    body, upper_shadow, lower_shadow, total_range, body_ratio, is_bullish, _ = _candle_props(
        o, h, l, c
    )

    if not is_bullish or total_range <= 0:
        return None
    if upper_shadow > 0.05 * total_range or lower_shadow > 0.05 * total_range:
        return None
    if body_ratio < 0.90:
        return None

    date = df.index[idx]
    description = (
        f"STRONG BULLISH: Bullish Marubozu - full body candle, no wicks\n"
        f"O ${o:.2f} H ${h:.2f} L ${l:.2f} C ${c:.2f}\n"
        f"Body: {body_ratio * 100:.0f}% of range - strong buying pressure throughout session\n"
        f"Watch for: Continuation above ${h:.2f}"
    )
    return _make_result(
        "Bullish Marubozu", "bullish", 78.0,
        [(date, o, "Marubozu Open"), (date, c, "Marubozu Close")],
        [idx],
        description,
    )


def _detect_bearish_marubozu(df: pd.DataFrame) -> Optional[Dict]:
    n = len(df)
    if n < 1:
        return None

    idx = n - 1
    o = float(df["open"].iloc[idx])
    h = float(df["high"].iloc[idx])
    l = float(df["low"].iloc[idx])
    c = float(df["close"].iloc[idx])
    body, upper_shadow, lower_shadow, total_range, body_ratio, _, is_bearish = _candle_props(
        o, h, l, c
    )

    if not is_bearish or total_range <= 0:
        return None
    if upper_shadow > 0.05 * total_range or lower_shadow > 0.05 * total_range:
        return None
    if body_ratio < 0.90:
        return None

    date = df.index[idx]
    description = (
        f"STRONG BEARISH: Bearish Marubozu - full body candle, no wicks\n"
        f"O ${o:.2f} H ${h:.2f} L ${l:.2f} C ${c:.2f}\n"
        f"Body: {body_ratio * 100:.0f}% of range - strong selling pressure throughout session\n"
        f"Watch for: Continuation below ${l:.2f}"
    )
    return _make_result(
        "Bearish Marubozu", "bearish", 78.0,
        [(date, c, "Marubozu Close"), (date, o, "Marubozu Open")],
        [idx],
        description,
    )


def _detect_spinning_top(df: pd.DataFrame) -> Optional[Dict]:
    n = len(df)
    if n < 1:
        return None

    idx = n - 1
    o = float(df["open"].iloc[idx])
    h = float(df["high"].iloc[idx])
    l = float(df["low"].iloc[idx])
    c = float(df["close"].iloc[idx])
    body, upper_shadow, lower_shadow, total_range, body_ratio, _, _ = _candle_props(o, h, l, c)

    if total_range <= 0:
        return None
    if not (0.10 <= body_ratio <= 0.35):
        return None
    if upper_shadow < 0.20 * total_range or lower_shadow < 0.20 * total_range:
        return None

    date = df.index[idx]
    description = (
        f"INDECISION: Spinning Top - small body with upper and lower shadows\n"
        f"O ${o:.2f} H ${h:.2f} L ${l:.2f} C ${c:.2f}\n"
        f"Body: {body_ratio * 100:.0f}% of range - indecision between buyers and sellers\n"
        f"Wait for directional confirmation on next candle"
    )
    return _make_result(
        "Spinning Top", "neutral", 62.0,
        [(date, l, "Shadow Low"), (date, h, "Shadow High")],
        [idx],
        description,
    )


# ---------------------------------------------------------------------------
# Two-candle patterns
# ---------------------------------------------------------------------------


def _detect_bullish_engulfing(df: pd.DataFrame) -> Optional[Dict]:
    n = len(df)
    if n < 7:
        return None
    trend = _trend_context(df["close"].values, n)
    if trend != "downtrend":
        return None

    i1, i2 = n - 2, n - 1
    o1 = float(df["open"].iloc[i1])
    h1 = float(df["high"].iloc[i1])
    l1 = float(df["low"].iloc[i1])
    c1 = float(df["close"].iloc[i1])
    o2 = float(df["open"].iloc[i2])
    h2 = float(df["high"].iloc[i2])
    l2 = float(df["low"].iloc[i2])
    c2 = float(df["close"].iloc[i2])

    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)

    if c1 >= o1:  # candle 1 must be bearish
        return None
    if c2 <= o2:  # candle 2 must be bullish
        return None
    if body2 <= body1:
        return None
    # candle 2 opens below candle 1's close and closes above candle 1's open
    if o2 >= c1 or c2 <= o1:
        return None

    engulf_ratio = body2 / body1 if body1 > 0 else 1.0
    confidence = min(95.0, 70.0 + engulf_ratio * 5)

    pct_decline = (
        (df["close"].iloc[i2] - df["close"].iloc[n - 6]) / df["close"].iloc[n - 6] * 100
    )
    vol = df["volume"].values if "volume" in df.columns else None
    vol_note = ""
    if vol is not None and vol[i1] > 0:
        vol_ratio = vol[i2] / vol[i1]
        if vol_ratio >= 1.5:
            confidence = min(100.0, confidence + 5)
        vol_note = (
            f"\nVolume: Candle 2 volume {vol_ratio:.1f}x candle 1 (strong confirmation)"
            if vol_ratio >= 1.5 else ""
        )

    date1, date2 = df.index[i1], df.index[i2]
    description = (
        f"BULLISH REVERSAL: Strong engulfing after 5-bar decline of {pct_decline:.1f}%\n"
        f"Candle 1 (bearish): O ${o1:.2f} H ${h1:.2f} L ${l1:.2f} C ${c1:.2f}"
        f" (body: ${body1:.2f})\n"
        f"Candle 2 (bullish): O ${o2:.2f} H ${h2:.2f} L ${l2:.2f} C ${c2:.2f}"
        f" (body: ${body2:.2f})\n"
        f"Engulfing ratio: {engulf_ratio:.1f}x{vol_note}\n"
        f"Prior trend: {pct_decline:.1f}% decline over 5 bars - reversal context valid\n"
        f"Watch for: Follow-through above ${h2:.2f} on next bar"
    )
    return _make_result(
        "Bullish Engulfing", "bullish", confidence,
        [(date1, l1, "Engulfed Candle"), (date2, c2, "Engulfing Candle")],
        [i1, i2],
        description,
    )


def _detect_bearish_engulfing(df: pd.DataFrame) -> Optional[Dict]:
    n = len(df)
    if n < 7:
        return None
    trend = _trend_context(df["close"].values, n)
    if trend != "uptrend":
        return None

    i1, i2 = n - 2, n - 1
    o1 = float(df["open"].iloc[i1])
    h1 = float(df["high"].iloc[i1])
    l1 = float(df["low"].iloc[i1])
    c1 = float(df["close"].iloc[i1])
    o2 = float(df["open"].iloc[i2])
    h2 = float(df["high"].iloc[i2])
    l2 = float(df["low"].iloc[i2])
    c2 = float(df["close"].iloc[i2])

    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)

    if c1 <= o1:  # candle 1 must be bullish
        return None
    if c2 >= o2:  # candle 2 must be bearish
        return None
    if body2 <= body1:
        return None
    # candle 2 opens above candle 1's close and closes below candle 1's open
    if o2 <= c1 or c2 >= o1:
        return None

    engulf_ratio = body2 / body1 if body1 > 0 else 1.0
    confidence = min(95.0, 70.0 + engulf_ratio * 5)

    pct_rise = (
        (df["close"].iloc[i2] - df["close"].iloc[n - 6]) / df["close"].iloc[n - 6] * 100
    )
    date1, date2 = df.index[i1], df.index[i2]
    description = (
        f"BEARISH REVERSAL: Strong engulfing after 5-bar rise of {pct_rise:.1f}%\n"
        f"Candle 1 (bullish): O ${o1:.2f} H ${h1:.2f} L ${l1:.2f} C ${c1:.2f}"
        f" (body: ${body1:.2f})\n"
        f"Candle 2 (bearish): O ${o2:.2f} H ${h2:.2f} L ${l2:.2f} C ${c2:.2f}"
        f" (body: ${body2:.2f})\n"
        f"Engulfing ratio: {engulf_ratio:.1f}x\n"
        f"Prior trend: {pct_rise:.1f}% rise over 5 bars - reversal context valid\n"
        f"Watch for: Follow-through below ${l2:.2f} on next bar"
    )
    return _make_result(
        "Bearish Engulfing", "bearish", confidence,
        [(date1, h1, "Engulfed Candle"), (date2, c2, "Engulfing Candle")],
        [i1, i2],
        description,
    )


def _detect_bullish_harami(df: pd.DataFrame) -> Optional[Dict]:
    n = len(df)
    if n < 7:
        return None
    trend = _trend_context(df["close"].values, n)
    if trend != "downtrend":
        return None

    i1, i2 = n - 2, n - 1
    o1 = float(df["open"].iloc[i1])
    h1 = float(df["high"].iloc[i1])
    l1 = float(df["low"].iloc[i1])
    c1 = float(df["close"].iloc[i1])
    o2 = float(df["open"].iloc[i2])
    h2 = float(df["high"].iloc[i2])
    l2 = float(df["low"].iloc[i2])
    c2 = float(df["close"].iloc[i2])

    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)

    if c1 >= o1:  # candle 1 must be bearish
        return None
    if c2 <= o2:  # candle 2 must be bullish
        return None
    if body2 >= body1:
        return None
    # candle 2 body contained within candle 1 body
    if min(o2, c2) < min(o1, c1) or max(o2, c2) > max(o1, c1):
        return None

    date1, date2 = df.index[i1], df.index[i2]
    description = (
        f"BULLISH POTENTIAL REVERSAL: Bullish Harami after downtrend\n"
        f"Candle 1 (large bearish): O ${o1:.2f} H ${h1:.2f} L ${l1:.2f} C ${c1:.2f}\n"
        f"Candle 2 (small bullish): O ${o2:.2f} H ${h2:.2f} L ${l2:.2f} C ${c2:.2f}\n"
        f"Small bullish candle contained within prior bearish candle - potential reversal\n"
        f"Watch for: Bullish confirmation above ${h2:.2f}"
    )
    return _make_result(
        "Bullish Harami", "bullish", 68.0,
        [(date1, l1, "Prior Bearish"), (date2, c2, "Harami Candle")],
        [i1, i2],
        description,
    )


def _detect_bearish_harami(df: pd.DataFrame) -> Optional[Dict]:
    n = len(df)
    if n < 7:
        return None
    trend = _trend_context(df["close"].values, n)
    if trend != "uptrend":
        return None

    i1, i2 = n - 2, n - 1
    o1 = float(df["open"].iloc[i1])
    h1 = float(df["high"].iloc[i1])
    l1 = float(df["low"].iloc[i1])
    c1 = float(df["close"].iloc[i1])
    o2 = float(df["open"].iloc[i2])
    h2 = float(df["high"].iloc[i2])
    l2 = float(df["low"].iloc[i2])
    c2 = float(df["close"].iloc[i2])

    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)

    if c1 <= o1:  # candle 1 must be bullish
        return None
    if c2 >= o2:  # candle 2 must be bearish
        return None
    if body2 >= body1:
        return None
    if min(o2, c2) < min(o1, c1) or max(o2, c2) > max(o1, c1):
        return None

    date1, date2 = df.index[i1], df.index[i2]
    description = (
        f"BEARISH POTENTIAL REVERSAL: Bearish Harami after uptrend\n"
        f"Candle 1 (large bullish): O ${o1:.2f} H ${h1:.2f} L ${l1:.2f} C ${c1:.2f}\n"
        f"Candle 2 (small bearish): O ${o2:.2f} H ${h2:.2f} L ${l2:.2f} C ${c2:.2f}\n"
        f"Small bearish candle contained within prior bullish candle - potential reversal\n"
        f"Watch for: Bearish confirmation below ${l2:.2f}"
    )
    return _make_result(
        "Bearish Harami", "bearish", 68.0,
        [(date1, h1, "Prior Bullish"), (date2, c2, "Harami Candle")],
        [i1, i2],
        description,
    )


def _detect_tweezer_top(df: pd.DataFrame) -> Optional[Dict]:
    n = len(df)
    if n < 7:
        return None
    trend = _trend_context(df["close"].values, n)
    if trend != "uptrend":
        return None

    i1, i2 = n - 2, n - 1
    h1 = float(df["high"].iloc[i1])
    h2 = float(df["high"].iloc[i2])
    l1 = float(df["low"].iloc[i1])
    l2 = float(df["low"].iloc[i2])
    o2 = float(df["open"].iloc[i2])
    c2 = float(df["close"].iloc[i2])

    if h1 <= 0 or abs(h1 - h2) / h1 > 0.001:
        return None
    if c2 >= o2:  # second candle should be bearish
        return None

    date1, date2 = df.index[i1], df.index[i2]
    description = (
        f"BEARISH REVERSAL: Tweezer Top - matching highs at ${h1:.2f}\n"
        f"Two candles failing at same high - strong resistance\n"
        f"Watch for: Breakdown below ${min(l1, l2):.2f}"
    )
    return _make_result(
        "Tweezer Top", "bearish", 70.0,
        [(date1, h1, "Tweezer High 1"), (date2, h2, "Tweezer High 2")],
        [i1, i2],
        description,
    )


def _detect_tweezer_bottom(df: pd.DataFrame) -> Optional[Dict]:
    n = len(df)
    if n < 7:
        return None
    trend = _trend_context(df["close"].values, n)
    if trend != "downtrend":
        return None

    i1, i2 = n - 2, n - 1
    l1 = float(df["low"].iloc[i1])
    l2 = float(df["low"].iloc[i2])
    h1 = float(df["high"].iloc[i1])
    h2 = float(df["high"].iloc[i2])
    o2 = float(df["open"].iloc[i2])
    c2 = float(df["close"].iloc[i2])

    if l1 <= 0 or abs(l1 - l2) / l1 > 0.001:
        return None
    if c2 <= o2:  # second candle should be bullish
        return None

    date1, date2 = df.index[i1], df.index[i2]
    description = (
        f"BULLISH REVERSAL: Tweezer Bottom - matching lows at ${l1:.2f}\n"
        f"Two candles bouncing from same low - strong support\n"
        f"Watch for: Breakout above ${max(h1, h2):.2f}"
    )
    return _make_result(
        "Tweezer Bottom", "bullish", 70.0,
        [(date1, l1, "Tweezer Low 1"), (date2, l2, "Tweezer Low 2")],
        [i1, i2],
        description,
    )


def _detect_piercing_line(df: pd.DataFrame) -> Optional[Dict]:
    n = len(df)
    if n < 7:
        return None
    trend = _trend_context(df["close"].values, n)
    if trend != "downtrend":
        return None

    i1, i2 = n - 2, n - 1
    o1 = float(df["open"].iloc[i1])
    h1 = float(df["high"].iloc[i1])
    l1 = float(df["low"].iloc[i1])
    c1 = float(df["close"].iloc[i1])
    o2 = float(df["open"].iloc[i2])
    h2 = float(df["high"].iloc[i2])
    l2 = float(df["low"].iloc[i2])
    c2 = float(df["close"].iloc[i2])

    if c1 >= o1:  # candle 1 must be bearish
        return None
    if c2 <= o2:  # candle 2 must be bullish
        return None
    if o2 >= l1:  # candle 2 opens below prior low
        return None
    mid1 = (o1 + c1) / 2
    if c2 <= mid1:  # candle 2 closes above midpoint of candle 1
        return None
    if c2 >= o1:  # but not a full engulf (stays below prior open)
        return None

    date1, date2 = df.index[i1], df.index[i2]
    description = (
        f"BULLISH REVERSAL: Piercing Line pattern\n"
        f"Candle 1 (bearish): O ${o1:.2f} H ${h1:.2f} L ${l1:.2f} C ${c1:.2f}\n"
        f"Candle 2 opens at ${o2:.2f} (below prior low ${l1:.2f}),"
        f" closes at ${c2:.2f} (above midpoint ${mid1:.2f})\n"
        f"Strong recovery indicates buying pressure overwhelming sellers\n"
        f"Watch for: Confirmation above ${h2:.2f}"
    )
    return _make_result(
        "Piercing Line", "bullish", 75.0,
        [(date1, l1, "Prior Bearish Low"), (date2, c2, "Piercing Close")],
        [i1, i2],
        description,
    )


def _detect_dark_cloud_cover(df: pd.DataFrame) -> Optional[Dict]:
    n = len(df)
    if n < 7:
        return None
    trend = _trend_context(df["close"].values, n)
    if trend != "uptrend":
        return None

    i1, i2 = n - 2, n - 1
    o1 = float(df["open"].iloc[i1])
    h1 = float(df["high"].iloc[i1])
    l1 = float(df["low"].iloc[i1])
    c1 = float(df["close"].iloc[i1])
    o2 = float(df["open"].iloc[i2])
    h2 = float(df["high"].iloc[i2])
    l2 = float(df["low"].iloc[i2])
    c2 = float(df["close"].iloc[i2])

    if c1 <= o1:  # candle 1 must be bullish
        return None
    if c2 >= o2:  # candle 2 must be bearish
        return None
    if o2 <= h1:  # candle 2 opens above prior high
        return None
    mid1 = (o1 + c1) / 2
    if c2 >= mid1:  # candle 2 closes below midpoint of candle 1
        return None
    if c2 <= o1:  # but not a full engulf (stays above prior open)
        return None

    date1, date2 = df.index[i1], df.index[i2]
    description = (
        f"BEARISH REVERSAL: Dark Cloud Cover pattern\n"
        f"Candle 1 (bullish): O ${o1:.2f} H ${h1:.2f} L ${l1:.2f} C ${c1:.2f}\n"
        f"Candle 2 opens at ${o2:.2f} (above prior high ${h1:.2f}),"
        f" closes at ${c2:.2f} (below midpoint ${mid1:.2f})\n"
        f"Strong reversal after gap up - sellers take control\n"
        f"Watch for: Confirmation below ${l2:.2f}"
    )
    return _make_result(
        "Dark Cloud Cover", "bearish", 75.0,
        [(date1, h1, "Prior Bullish High"), (date2, c2, "Dark Cloud Close")],
        [i1, i2],
        description,
    )


# ---------------------------------------------------------------------------
# Three-candle patterns
# ---------------------------------------------------------------------------


def _detect_morning_star(df: pd.DataFrame) -> Optional[Dict]:
    n = len(df)
    if n < 8:
        return None
    trend = _trend_context(df["close"].values, n)
    if trend != "downtrend":
        return None

    i1, i2, i3 = n - 3, n - 2, n - 1
    o1 = float(df["open"].iloc[i1])
    c1 = float(df["close"].iloc[i1])
    o2 = float(df["open"].iloc[i2])
    h2 = float(df["high"].iloc[i2])
    l2 = float(df["low"].iloc[i2])
    c2 = float(df["close"].iloc[i2])
    o3 = float(df["open"].iloc[i3])
    h3 = float(df["high"].iloc[i3])
    l3 = float(df["low"].iloc[i3])
    c3 = float(df["close"].iloc[i3])

    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)

    if c1 >= o1:  # candle 1 must be bearish
        return None
    if o2 >= c1:  # star must gap down (open below candle 1 close)
        return None
    if body2 >= 0.5 * body1:  # star must be small
        return None
    if c3 <= o3:  # candle 3 must be bullish
        return None
    if c3 <= (o1 + c1) / 2:  # candle 3 closes above midpoint of candle 1
        return None

    pct_decline = (
        (df["close"].iloc[i3] - df["close"].iloc[n - 6]) / df["close"].iloc[n - 6] * 100
    )
    date1, date2, date3 = df.index[i1], df.index[i2], df.index[i3]
    description = (
        f"BULLISH REVERSAL: Morning Star pattern after {pct_decline:.1f}% decline\n"
        f"Candle 1 (bearish): C ${c1:.2f} (large bearish candle)\n"
        f"Candle 2 (star): O ${o2:.2f} C ${c2:.2f} (small body, gaps down)\n"
        f"Candle 3 (bullish): O ${o3:.2f} C ${c3:.2f} (strong recovery)\n"
        f"Classic 3-candle bottom reversal - watch for follow-through above ${h3:.2f}"
    )
    return _make_result(
        "Morning Star", "bullish", 78.0,
        [(date1, c1, "Bearish Candle"), (date2, l2, "Star"), (date3, c3, "Bullish Candle")],
        [i1, i2, i3],
        description,
    )


def _detect_evening_star(df: pd.DataFrame) -> Optional[Dict]:
    n = len(df)
    if n < 8:
        return None
    trend = _trend_context(df["close"].values, n)
    if trend != "uptrend":
        return None

    i1, i2, i3 = n - 3, n - 2, n - 1
    o1 = float(df["open"].iloc[i1])
    h1 = float(df["high"].iloc[i1])
    l1 = float(df["low"].iloc[i1])
    c1 = float(df["close"].iloc[i1])
    o2 = float(df["open"].iloc[i2])
    h2 = float(df["high"].iloc[i2])
    l2 = float(df["low"].iloc[i2])
    c2 = float(df["close"].iloc[i2])
    o3 = float(df["open"].iloc[i3])
    h3 = float(df["high"].iloc[i3])
    l3 = float(df["low"].iloc[i3])
    c3 = float(df["close"].iloc[i3])

    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)

    if c1 <= o1:  # candle 1 must be bullish
        return None
    if o2 <= c1:  # star must gap up (open above candle 1 close)
        return None
    if body2 >= 0.5 * body1:  # star must be small
        return None
    if c3 >= o3:  # candle 3 must be bearish
        return None
    if c3 >= (o1 + c1) / 2:  # candle 3 closes below midpoint of candle 1
        return None

    pct_rise = (
        (df["close"].iloc[i3] - df["close"].iloc[n - 6]) / df["close"].iloc[n - 6] * 100
    )
    date1, date2, date3 = df.index[i1], df.index[i2], df.index[i3]
    description = (
        f"BEARISH REVERSAL: Evening Star pattern after {pct_rise:.1f}% rise\n"
        f"Candle 1 (bullish): C ${c1:.2f} (large bullish candle)\n"
        f"Candle 2 (star): O ${o2:.2f} C ${c2:.2f} (small body, gaps up)\n"
        f"Candle 3 (bearish): O ${o3:.2f} C ${c3:.2f} (strong reversal)\n"
        f"Classic 3-candle top reversal - watch for follow-through below ${l3:.2f}"
    )
    return _make_result(
        "Evening Star", "bearish", 78.0,
        [(date1, c1, "Bullish Candle"), (date2, h2, "Star"), (date3, c3, "Bearish Candle")],
        [i1, i2, i3],
        description,
    )


def _detect_three_white_soldiers(df: pd.DataFrame) -> Optional[Dict]:
    n = len(df)
    if n < 3:
        return None

    i1, i2, i3 = n - 3, n - 2, n - 1
    o1 = float(df["open"].iloc[i1])
    h1 = float(df["high"].iloc[i1])
    c1 = float(df["close"].iloc[i1])
    o2 = float(df["open"].iloc[i2])
    h2 = float(df["high"].iloc[i2])
    c2 = float(df["close"].iloc[i2])
    o3 = float(df["open"].iloc[i3])
    h3 = float(df["high"].iloc[i3])
    c3 = float(df["close"].iloc[i3])

    if c1 <= o1 or c2 <= o2 or c3 <= o3:  # all three must be bullish
        return None
    if not (o1 <= o2 <= c1 and o2 <= o3 <= c2):  # each opens within prior body
        return None
    if not (c1 < c2 < c3):  # each closes higher
        return None

    total_gain = (c3 - o1) / o1 * 100 if o1 > 0 else 0.0
    date1, date2, date3 = df.index[i1], df.index[i2], df.index[i3]
    description = (
        f"STRONG BULLISH: Three White Soldiers - {total_gain:.1f}% gain over 3 candles\n"
        f"Candle 1: O ${o1:.2f} C ${c1:.2f} | Candle 2: O ${o2:.2f} C ${c2:.2f}"
        f" | Candle 3: O ${o3:.2f} C ${c3:.2f}\n"
        f"Three consecutive bullish candles, each opening within prior body and closing higher\n"
        f"Strong buying momentum - continuation likely above ${h3:.2f}"
    )
    return _make_result(
        "Three White Soldiers", "bullish", 80.0,
        [(date1, o1, "Soldier 1 Open"), (date2, c2, "Soldier 2 Close"), (date3, c3, "Soldier 3 Close")],
        [i1, i2, i3],
        description,
    )


def _detect_three_black_crows(df: pd.DataFrame) -> Optional[Dict]:
    n = len(df)
    if n < 3:
        return None

    i1, i2, i3 = n - 3, n - 2, n - 1
    o1 = float(df["open"].iloc[i1])
    l1 = float(df["low"].iloc[i1])
    c1 = float(df["close"].iloc[i1])
    o2 = float(df["open"].iloc[i2])
    l2 = float(df["low"].iloc[i2])
    c2 = float(df["close"].iloc[i2])
    o3 = float(df["open"].iloc[i3])
    l3 = float(df["low"].iloc[i3])
    c3 = float(df["close"].iloc[i3])

    if c1 >= o1 or c2 >= o2 or c3 >= o3:  # all three must be bearish
        return None
    if not (c1 <= o2 <= o1 and c2 <= o3 <= o2):  # each opens within prior body
        return None
    if not (c1 > c2 > c3):  # each closes lower
        return None

    total_loss = (c3 - o1) / o1 * 100 if o1 > 0 else 0.0
    date1, date2, date3 = df.index[i1], df.index[i2], df.index[i3]
    description = (
        f"STRONG BEARISH: Three Black Crows - {total_loss:.1f}% loss over 3 candles\n"
        f"Candle 1: O ${o1:.2f} C ${c1:.2f} | Candle 2: O ${o2:.2f} C ${c2:.2f}"
        f" | Candle 3: O ${o3:.2f} C ${c3:.2f}\n"
        f"Three consecutive bearish candles, each opening within prior body and closing lower\n"
        f"Strong selling pressure - continuation likely below ${l3:.2f}"
    )
    return _make_result(
        "Three Black Crows", "bearish", 80.0,
        [(date1, o1, "Crow 1 Open"), (date2, c2, "Crow 2 Close"), (date3, c3, "Crow 3 Close")],
        [i1, i2, i3],
        description,
    )


def _detect_three_inside_up(df: pd.DataFrame) -> Optional[Dict]:
    n = len(df)
    if n < 8:
        return None
    trend = _trend_context(df["close"].values, n)
    if trend != "downtrend":
        return None

    i1, i2, i3 = n - 3, n - 2, n - 1
    o1 = float(df["open"].iloc[i1])
    h1 = float(df["high"].iloc[i1])
    l1 = float(df["low"].iloc[i1])
    c1 = float(df["close"].iloc[i1])
    o2 = float(df["open"].iloc[i2])
    h2 = float(df["high"].iloc[i2])
    c2 = float(df["close"].iloc[i2])
    o3 = float(df["open"].iloc[i3])
    h3 = float(df["high"].iloc[i3])
    c3 = float(df["close"].iloc[i3])

    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)

    if c1 >= o1:  # candle 1: large bearish
        return None
    if c2 <= o2:  # candle 2: small bullish
        return None
    if body2 >= body1:
        return None
    if min(o2, c2) < min(o1, c1) or max(o2, c2) > max(o1, c1):
        return None
    if c3 <= o3:  # candle 3: bullish, closes above candle 2
        return None
    if c3 <= c2:
        return None

    date1, date2, date3 = df.index[i1], df.index[i2], df.index[i3]
    description = (
        f"BULLISH CONFIRMATION: Three Inside Up\n"
        f"Bearish harami (candles 1-2) confirmed by higher close on candle 3\n"
        f"C1: ${c1:.2f} | C2: ${c2:.2f} | C3: ${c3:.2f} (higher close confirms reversal)\n"
        f"Watch for: Continuation above ${h3:.2f}"
    )
    return _make_result(
        "Three Inside Up", "bullish", 76.0,
        [(date1, l1, "Prior Bearish"), (date2, c2, "Harami"), (date3, c3, "Confirmation")],
        [i1, i2, i3],
        description,
    )


def _detect_three_inside_down(df: pd.DataFrame) -> Optional[Dict]:
    n = len(df)
    if n < 8:
        return None
    trend = _trend_context(df["close"].values, n)
    if trend != "uptrend":
        return None

    i1, i2, i3 = n - 3, n - 2, n - 1
    o1 = float(df["open"].iloc[i1])
    h1 = float(df["high"].iloc[i1])
    l1 = float(df["low"].iloc[i1])
    c1 = float(df["close"].iloc[i1])
    o2 = float(df["open"].iloc[i2])
    l2 = float(df["low"].iloc[i2])
    c2 = float(df["close"].iloc[i2])
    o3 = float(df["open"].iloc[i3])
    l3 = float(df["low"].iloc[i3])
    c3 = float(df["close"].iloc[i3])

    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)

    if c1 <= o1:  # candle 1: large bullish
        return None
    if c2 >= o2:  # candle 2: small bearish
        return None
    if body2 >= body1:
        return None
    if min(o2, c2) < min(o1, c1) or max(o2, c2) > max(o1, c1):
        return None
    if c3 >= o3:  # candle 3: bearish, closes below candle 2
        return None
    if c3 >= c2:
        return None

    date1, date2, date3 = df.index[i1], df.index[i2], df.index[i3]
    description = (
        f"BEARISH CONFIRMATION: Three Inside Down\n"
        f"Bullish harami (candles 1-2) confirmed by lower close on candle 3\n"
        f"C1: ${c1:.2f} | C2: ${c2:.2f} | C3: ${c3:.2f} (lower close confirms reversal)\n"
        f"Watch for: Continuation below ${l3:.2f}"
    )
    return _make_result(
        "Three Inside Down", "bearish", 76.0,
        [(date1, h1, "Prior Bullish"), (date2, c2, "Harami"), (date3, c3, "Confirmation")],
        [i1, i2, i3],
        description,
    )


# ---------------------------------------------------------------------------
# Registry and public API
# ---------------------------------------------------------------------------

CANDLESTICK_PATTERNS: Dict[str, Callable[[pd.DataFrame], Optional[Dict]]] = {
    "hammer": _detect_hammer,
    "inverted hammer": _detect_inverted_hammer,
    "hanging man": _detect_hanging_man,
    "shooting star": _detect_shooting_star,
    "doji": _detect_doji,
    "dragonfly doji": _detect_dragonfly_doji,
    "gravestone doji": _detect_gravestone_doji,
    "bullish marubozu": _detect_bullish_marubozu,
    "bearish marubozu": _detect_bearish_marubozu,
    "spinning top": _detect_spinning_top,
    "bullish engulfing": _detect_bullish_engulfing,
    "bearish engulfing": _detect_bearish_engulfing,
    "bullish harami": _detect_bullish_harami,
    "bearish harami": _detect_bearish_harami,
    "tweezer top": _detect_tweezer_top,
    "tweezer bottom": _detect_tweezer_bottom,
    "piercing line": _detect_piercing_line,
    "rising sun": _detect_piercing_line,
    "dark cloud cover": _detect_dark_cloud_cover,
    "morning star": _detect_morning_star,
    "evening star": _detect_evening_star,
    "three white soldiers": _detect_three_white_soldiers,
    "three black crows": _detect_three_black_crows,
    "three inside up": _detect_three_inside_up,
    "three inside down": _detect_three_inside_down,
}


def detect_candlestick_pattern(
    pattern_name: str,
    df: pd.DataFrame,
) -> Optional[Dict]:
    """Detect a specific candlestick pattern in OHLCV data.

    Each detector looks only at the most recent 1-3 candles plus 5 bars of
    trend context, so a minimum of 8 bars is sufficient for all patterns.

    Args:
        pattern_name: Case-insensitive pattern name (e.g. ``"hammer"``).
        df:           Daily OHLCV DataFrame with ``open``, ``high``, ``low``,
                      ``close``, ``volume`` columns.

    Returns:
        A result dict with ``pattern``, ``pattern_type``, ``confidence``,
        ``bias``, ``key_points``, ``lines``, ``candle_indices``,
        ``description``, ``state``, and breakout/compression fields set
        to ``None``; or ``None`` if no pattern is detected.
    """
    key = pattern_name.strip().lower()
    detector = CANDLESTICK_PATTERNS.get(key)
    if detector is None:
        logger.warning("Unknown candlestick pattern %r", pattern_name)
        return None

    if df is None or df.empty or len(df) < 3:
        return None

    try:
        return detector(df)
    except Exception as exc:
        logger.debug("Candlestick detection error (%s): %s", pattern_name, exc)
        return None
