"""
detector.py - Compression-tracking chart pattern detection engine.

Detects patterns IN PROGRESS (before breakout), tracking compression and
consolidation live.  Alerts BEFORE breakout, invalidates AFTER breakout.

Pattern States
--------------
FORMING    : Structure detected, still building
NEAR_BREAK : Price approaching breakout level (alert zone)
BROKEN     : Price closed outside structure - invalidated (never returned)

Supported patterns
------------------
- Bull Flag / Bear Flag
- Ascending Triangle / Descending Triangle
- Rising Wedge / Falling Wedge
- Double Bottom / Double Top
- Head and Shoulders / Inverse Head and Shoulders
- Cup and Handle
- Symmetrical Triangle
- Pennant
- Channel Up / Channel Down
- Flat Base
- Broadening Formation
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

KeyPoint = Tuple["pd.Timestamp", float, str]   # (date, price, label)
PatternResult = Dict                           # {"pattern", "confidence", "key_points", ...}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PATTERN_LOOKBACK = 30  # use only the last 30 candles for detection
FLAG_WIDTH_THRESHOLD = 0.05  # max flag channel width as fraction of pole top price


# ---------------------------------------------------------------------------
# Pattern State Enum
# ---------------------------------------------------------------------------

class PatternState(Enum):
    FORMING = "forming"           # Structure detected, still building
    NEAR_BREAK = "near_break"     # Price approaching breakout level (ALERT ZONE)
    BROKEN = "broken"             # Price closed outside structure - INVALIDATE


# ---------------------------------------------------------------------------
# ActivePattern dataclass
# ---------------------------------------------------------------------------

@dataclass
class ActivePattern:
    pattern_type: str
    state: PatternState
    start_index: int
    ticker: str
    pivot_highs: List[Tuple[datetime, float]]
    pivot_lows: List[Tuple[datetime, float]]
    upper_trendline: Tuple[float, float]      # (slope, intercept) from linreg
    lower_trendline: Tuple[float, float]      # (slope, intercept) from linreg
    breakout_level_upper: float
    breakout_level_lower: float
    compression_ratio: float
    pattern_age: int
    created_at: datetime
    last_updated: datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_pivot_high(highs: np.ndarray, idx: int) -> bool:
    """Strict pivot high: high must be higher than previous 2 AND next 2 candles."""
    if idx < 2 or idx >= len(highs) - 2:
        return False
    return (
        highs[idx] > highs[idx - 1] and highs[idx] > highs[idx - 2]
        and highs[idx] > highs[idx + 1] and highs[idx] > highs[idx + 2]
    )


def _is_pivot_low(lows: np.ndarray, idx: int) -> bool:
    """Strict pivot low: low must be lower than previous 2 AND next 2 candles."""
    if idx < 2 or idx >= len(lows) - 2:
        return False
    return (
        lows[idx] < lows[idx - 1] and lows[idx] < lows[idx - 2]
        and lows[idx] < lows[idx + 1] and lows[idx] < lows[idx + 2]
    )


def _find_strict_pivots(df: pd.DataFrame) -> Tuple[List[int], List[int]]:
    """Return (pivot_high_indices, pivot_low_indices) using strict 2-bar rule."""
    highs = df["high"].values
    lows = df["low"].values
    n = len(highs)
    pivot_highs = [i for i in range(2, n - 2) if _is_pivot_high(highs, i)]
    pivot_lows = [i for i in range(2, n - 2) if _is_pivot_low(lows, i)]
    return pivot_highs, pivot_lows


def _linreg(y: np.ndarray, x: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Return (slope, intercept) of linear regression through *y*."""
    if x is None:
        x = np.arange(len(y), dtype=float)
    if len(x) < 2:
        return 0.0, float(y[0]) if len(y) > 0 else 0.0
    coeffs = np.polyfit(x, y, 1)
    return float(coeffs[0]), float(coeffs[1])


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Compute the *period*-bar Average True Range."""
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(close)
    if n < 2:
        return float(high[-1] - low[-1]) if n > 0 else 1.0
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ),
    )
    if len(tr) == 0:
        return 1.0
    return float(np.mean(tr[-period:])) if len(tr) >= period else float(np.mean(tr))


def _determine_state(
    compression_ratio: float,
    current_price: float,
    upper_at_current: float,
    lower_at_current: float,
    atr: float,
    breakout_level_upper: float,
    breakout_level_lower: float,
) -> Optional[PatternState]:
    """Return the pattern state, or ``None`` if the pattern should be discarded.

    Returns ``None`` when:
    - Price closes outside the trendlines (BROKEN)
    - Compression ratio > 0.60 (too early - not yet a pattern)
    """
    # Structure validity check - mandatory on every call
    if current_price > upper_at_current or current_price < lower_at_current:
        return None  # BROKEN - invalidate

    # Too early - compression has not started yet
    if compression_ratio > 0.60:
        return None

    # ATR-based proximity or tight compression -> NEAR_BREAK
    distance_to_upper = abs(current_price - breakout_level_upper)
    distance_to_lower = abs(current_price - breakout_level_lower)
    distance_to_breakout = min(distance_to_upper, distance_to_lower)

    if distance_to_breakout <= 0.25 * atr or compression_ratio < 0.35:
        return PatternState.NEAR_BREAK

    return PatternState.FORMING


def _compression_confidence(compression_ratio: float, state: PatternState) -> float:
    """Compute a confidence score from compression ratio and state."""
    base = 75.0 if state == PatternState.NEAR_BREAK else 65.0
    compression_score = max(0.0, (0.60 - compression_ratio) / 0.60) * 25.0
    return min(100.0, base + compression_score)


def _vol_trend_desc(volume: np.ndarray, start_idx: int) -> str:
    """Return a descriptive string for volume trend during the pattern window."""
    pat_vol = volume[start_idx:].astype(float)
    if len(pat_vol) < 3:
        return "Insufficient data"
    slope, _ = _linreg(pat_vol)
    avg = float(np.mean(pat_vol))
    if avg == 0:
        return "N/A"
    pct_change = slope * len(pat_vol) / avg
    if pct_change < -0.15:
        return "Declining (confirming)"
    elif pct_change > 0.15:
        return "Increasing"
    return "Flat/neutral"


def _fmt_date(ts) -> str:
    """Return a short ``Mon DD`` date string from a timestamp."""
    try:
        return pd.Timestamp(ts).strftime("%b %d")
    except Exception:
        return str(ts)


# ---------------------------------------------------------------------------
# Individual pattern detectors (compression-based, in-progress only)
# ---------------------------------------------------------------------------


def _detect_triangle(df: pd.DataFrame, ascending: bool) -> Optional[PatternResult]:
    """Detect Ascending or Descending Triangle IN PROGRESS using pivot regression.

    Ascending triangle: flat resistance, rising support (higher lows).
    Descending triangle: flat support, falling resistance (lower highs).
    """
    n = len(df)
    if n < 15:
        return None

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    pivot_highs, pivot_lows = _find_strict_pivots(df)

    if len(pivot_highs) < 2 or len(pivot_lows) < 2:
        return None

    ph_x = np.array(pivot_highs, dtype=float)
    ph_y = high[pivot_highs]
    pl_x = np.array(pivot_lows, dtype=float)
    pl_y = low[pivot_lows]

    resist_slope, resist_intercept = _linreg(ph_y, ph_x)
    support_slope, support_intercept = _linreg(pl_y, pl_x)

    pattern_name = "Ascending Triangle" if ascending else "Descending Triangle"

    if ascending:
        # Flat resistance (near-zero slope), rising support (positive slope)
        if not (-0.05 < resist_slope < 0.05) or support_slope <= 0:
            return None
    else:
        # Flat support (near-zero slope), falling resistance (negative slope)
        if not (-0.05 < support_slope < 0.05) or resist_slope >= 0:
            return None

    # Compression ratio = current_width / initial_width
    start_idx = float(min(pivot_highs[0], pivot_lows[0]))
    end_idx = float(n - 1)

    upper_at_start = resist_slope * start_idx + resist_intercept
    lower_at_start = support_slope * start_idx + support_intercept
    initial_width = abs(upper_at_start - lower_at_start)

    upper_at_current = resist_slope * end_idx + resist_intercept
    lower_at_current = support_slope * end_idx + support_intercept
    current_width = abs(upper_at_current - lower_at_current)

    if initial_width <= 0:
        return None

    compression_ratio = current_width / initial_width

    # Expanding structure - not a valid pattern
    if compression_ratio >= 1.0:
        return None

    current_price = float(close[-1])
    atr = _compute_atr(df)

    state = _determine_state(
        compression_ratio, current_price,
        upper_at_current, lower_at_current,
        atr, upper_at_current, lower_at_current,
    )
    if state is None:
        return None

    start_idx_int = int(start_idx)
    start_date = df.index[start_idx_int]
    end_date = df.index[-1]

    key_points: List[KeyPoint] = [
        (start_date, upper_at_start, "Resistance Start"),
        (end_date, upper_at_current, "Resistance End"),
        (start_date, lower_at_start, "Support Start"),
        (end_date, lower_at_current, "Support End"),
    ]

    distance_to_breakout = min(
        abs(current_price - upper_at_current),
        abs(current_price - lower_at_current),
    )
    confidence = _compression_confidence(compression_ratio, state)

    volume_arr = df["volume"].values if "volume" in df.columns else None
    vol_desc = "N/A"
    if volume_arr is not None:
        vol_desc = _vol_trend_desc(volume_arr, start_idx_int)
    dist_pct = distance_to_breakout / current_price * 100 if current_price > 0 else 0

    if ascending:
        stop_loss = round(lower_at_current - atr, 2)
        description = (
            f"BULLISH: Flat resistance with rising support — accumulation pattern\n"
            f"Resistance: ${upper_at_current:.2f} (tested {len(pivot_highs)}x) | "
            f"Support rising from ${lower_at_start:.2f} to ${lower_at_current:.2f}\n"
            f"Compression: {compression_ratio * 100:.0f}% "
            f"(from ${initial_width:.2f} spread to ${current_width:.2f})\n"
            f"Breakout level: ${upper_at_current:.2f} | Current: ${current_price:.2f} | "
            f"Distance: ${distance_to_breakout:.2f} ({dist_pct:.1f}%)\n"
            f"14-day ATR: ${atr:.2f} | Suggested stop: ${stop_loss:.2f} (below rising support)\n"
            f"Volume: {vol_desc}\n"
            f"Watch for: Close above ${upper_at_current:.2f} with volume surge"
        )
    else:
        stop_loss = round(upper_at_current + atr, 2)
        description = (
            f"BEARISH: Flat support with falling resistance — distribution pattern\n"
            f"Support: ${lower_at_current:.2f} (tested {len(pivot_lows)}x) | "
            f"Resistance falling from ${upper_at_start:.2f} to ${upper_at_current:.2f}\n"
            f"Compression: {compression_ratio * 100:.0f}% "
            f"(from ${initial_width:.2f} spread to ${current_width:.2f})\n"
            f"Breakdown level: ${lower_at_current:.2f} | Current: ${current_price:.2f} | "
            f"Distance: ${distance_to_breakout:.2f} ({dist_pct:.1f}%)\n"
            f"14-day ATR: ${atr:.2f} | Suggested stop: ${stop_loss:.2f} (above resistance)\n"
            f"Volume: {vol_desc}\n"
            f"Watch for: Close below ${lower_at_current:.2f} on heavy volume"
        )

    return {
        "pattern": pattern_name,
        "confidence": round(confidence, 1),
        "key_points": key_points,
        "description": description,
        "state": state,
        "compression_ratio": round(compression_ratio, 4),
        "breakout_level_upper": round(upper_at_current, 4),
        "breakout_level_lower": round(lower_at_current, 4),
        "distance_to_breakout": round(distance_to_breakout, 4),
        "lines": [
            [(start_date, upper_at_start), (end_date, upper_at_current)],
            [(start_date, lower_at_start), (end_date, lower_at_current)],
        ],
        "upper_trendline": (resist_slope, resist_intercept),
        "lower_trendline": (support_slope, support_intercept),
    }


def _detect_wedge(df: pd.DataFrame, rising: bool) -> Optional[PatternResult]:
    """Detect Rising or Falling Wedge IN PROGRESS using pivot regression.

    Rising wedge: both trendlines slope up and converge (bearish).
    Falling wedge: both trendlines slope down and converge (bullish).
    """
    n = len(df)
    if n < 15:
        return None

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    pivot_highs, pivot_lows = _find_strict_pivots(df)

    if len(pivot_highs) < 2 or len(pivot_lows) < 2:
        return None

    ph_x = np.array(pivot_highs, dtype=float)
    ph_y = high[pivot_highs]
    pl_x = np.array(pivot_lows, dtype=float)
    pl_y = low[pivot_lows]

    resist_slope, resist_intercept = _linreg(ph_y, ph_x)
    support_slope, support_intercept = _linreg(pl_y, pl_x)

    pattern_name = "Rising Wedge" if rising else "Falling Wedge"

    if rising:
        # Both slopes positive, support slope > resistance slope (converging upward)
        if resist_slope <= 0 or support_slope <= 0:
            return None
        if support_slope <= resist_slope:
            return None
    else:
        # Both slopes negative, resistance slope > support slope (converging downward)
        if resist_slope >= 0 or support_slope >= 0:
            return None
        if resist_slope <= support_slope:
            return None

    start_idx = float(min(pivot_highs[0], pivot_lows[0]))
    end_idx = float(n - 1)

    upper_at_start = resist_slope * start_idx + resist_intercept
    lower_at_start = support_slope * start_idx + support_intercept
    initial_width = abs(upper_at_start - lower_at_start)

    upper_at_current = resist_slope * end_idx + resist_intercept
    lower_at_current = support_slope * end_idx + support_intercept
    current_width = abs(upper_at_current - lower_at_current)

    if initial_width <= 0:
        return None

    compression_ratio = current_width / initial_width

    if compression_ratio >= 1.0:
        return None

    current_price = float(close[-1])
    atr = _compute_atr(df)

    state = _determine_state(
        compression_ratio, current_price,
        upper_at_current, lower_at_current,
        atr, upper_at_current, lower_at_current,
    )
    if state is None:
        return None

    start_idx_int = int(start_idx)
    start_date = df.index[start_idx_int]
    end_date = df.index[-1]

    key_points: List[KeyPoint] = [
        (start_date, upper_at_start, "Resistance Start"),
        (end_date, upper_at_current, "Resistance End"),
        (start_date, lower_at_start, "Support Start"),
        (end_date, lower_at_current, "Support End"),
    ]

    distance_to_breakout = min(
        abs(current_price - upper_at_current),
        abs(current_price - lower_at_current),
    )
    confidence = _compression_confidence(compression_ratio, state)

    volume_arr = df["volume"].values if "volume" in df.columns else None
    vol_desc = "N/A"
    if volume_arr is not None:
        vol_desc = _vol_trend_desc(volume_arr, start_idx_int)
    dist_pct = distance_to_breakout / current_price * 100 if current_price > 0 else 0

    if rising:
        stop_loss = round(upper_at_current + atr, 2)
        description = (
            f"BEARISH: Rising wedge — both trendlines rising and converging (reversal)\n"
            f"Resistance: ${upper_at_start:.2f} → ${upper_at_current:.2f} (+{resist_slope:.3f}/bar) | "
            f"Support: ${lower_at_start:.2f} → ${lower_at_current:.2f} (+{support_slope:.3f}/bar)\n"
            f"Compression: {compression_ratio * 100:.0f}% "
            f"(from ${initial_width:.2f} spread to ${current_width:.2f})\n"
            f"Breakdown level: ${lower_at_current:.2f} | Current: ${current_price:.2f} | "
            f"Distance: ${distance_to_breakout:.2f} ({dist_pct:.1f}%)\n"
            f"14-day ATR: ${atr:.2f} | Suggested stop: ${stop_loss:.2f} (above resistance)\n"
            f"Volume: {vol_desc}\n"
            f"Watch for: Close below ${lower_at_current:.2f} to confirm bearish breakdown"
        )
    else:
        stop_loss = round(lower_at_current - atr, 2)
        description = (
            f"BULLISH: Falling wedge — both trendlines falling and converging (reversal)\n"
            f"Resistance: ${upper_at_start:.2f} → ${upper_at_current:.2f} ({resist_slope:.3f}/bar) | "
            f"Support: ${lower_at_start:.2f} → ${lower_at_current:.2f} ({support_slope:.3f}/bar)\n"
            f"Compression: {compression_ratio * 100:.0f}% "
            f"(from ${initial_width:.2f} spread to ${current_width:.2f})\n"
            f"Breakout level: ${upper_at_current:.2f} | Current: ${current_price:.2f} | "
            f"Distance: ${distance_to_breakout:.2f} ({dist_pct:.1f}%)\n"
            f"14-day ATR: ${atr:.2f} | Suggested stop: ${stop_loss:.2f} (below support)\n"
            f"Volume: {vol_desc}\n"
            f"Watch for: Close above ${upper_at_current:.2f} to confirm bullish breakout"
        )

    return {
        "pattern": pattern_name,
        "confidence": round(confidence, 1),
        "key_points": key_points,
        "description": description,
        "state": state,
        "compression_ratio": round(compression_ratio, 4),
        "breakout_level_upper": round(upper_at_current, 4),
        "breakout_level_lower": round(lower_at_current, 4),
        "distance_to_breakout": round(distance_to_breakout, 4),
        "lines": [
            [(start_date, upper_at_start), (end_date, upper_at_current)],
            [(start_date, lower_at_start), (end_date, lower_at_current)],
        ],
        "upper_trendline": (resist_slope, resist_intercept),
        "lower_trendline": (support_slope, support_intercept),
    }


def _detect_bull_flag(df: pd.DataFrame) -> Optional[PatternResult]:
    """Detect Bull Flag IN PROGRESS - price must still be inside the channel.

    Rules
    -----
    1. Impulse leg: >8% rise in 3-12 candles
    2. Consolidation: 5-15 candles with slight downward drift (<5%)
    3. Price is STILL inside the flag channel (not broken out yet)
    4. Volume decreasing during flag
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values if "volume" in df.columns else None
    n = len(close)
    if n < 15:
        return None

    # Consolidation must include the current bar (n-1)
    for consol_start in range(max(1, n - 15), n - 4):
        consol_end = n - 1
        consol_len = consol_end - consol_start

        if consol_len < 5 or consol_len > 15:
            continue

        consol_high_vals = high[consol_start: consol_end + 1]
        consol_low_vals = low[consol_start: consol_end + 1]
        consol_close_vals = close[consol_start: consol_end + 1]

        chan_high = float(np.max(consol_high_vals))
        chan_low = float(np.min(consol_low_vals))
        chan_width = chan_high - chan_low
        if chan_width <= 0:
            continue

        current_close = float(close[-1])
        # Price must be STILL inside the channel - if already broke out, REJECT
        if current_close > chan_high or current_close < chan_low:
            continue

        # Slight downward drift (flag): between -5% and +1%
        flag_drift = (consol_close_vals[-1] - consol_close_vals[0]) / consol_close_vals[0]
        if not (-0.05 <= flag_drift <= 0.01):
            continue

        # Volume decreasing during flag
        if volume is not None and consol_len >= 3:
            vol_slope = _linreg(volume[consol_start: consol_end + 1])[0]
            if vol_slope > 0:
                continue

        # Look for impulse leg (pole) immediately before consolidation
        for pole_len in range(3, 13):
            pole_end = consol_start
            pole_start = pole_end - pole_len
            if pole_start < 0:
                break

            pole_return = (close[pole_end] - close[pole_start]) / close[pole_start]
            if pole_return < 0.08:
                continue

            # Consolidation range <= 5% of pole top price
            if chan_width / close[pole_end] > FLAG_WIDTH_THRESHOLD:
                continue

            breakout_upper = chan_high
            breakout_lower = chan_low
            atr = _compute_atr(df)
            distance_to_breakout = min(
                abs(current_close - breakout_upper),
                abs(current_close - breakout_lower),
            )

            state = (
                PatternState.NEAR_BREAK
                if distance_to_breakout <= 0.25 * atr
                else PatternState.FORMING
            )

            compression_ratio = chan_width / (chan_high * FLAG_WIDTH_THRESHOLD) if chan_high > 0 else 0.5
            compression_ratio = min(compression_ratio, 1.0)

            confidence = min(100.0, 65.0 + min(pole_return * 100, 20.0))
            if state == PatternState.NEAR_BREAK:
                confidence = min(confidence + 10, 100.0)

            key_points: List[KeyPoint] = [
                (df.index[pole_start], float(close[pole_start]), "Pole Start"),
                (df.index[pole_end], float(close[pole_end]), "Pole Top"),
                (df.index[-1], current_close, "Flag End"),
            ]

            chan_width_pct = chan_width / close[pole_end] * 100 if close[pole_end] > 0 else 0
            dist_to_break = abs(current_close - breakout_upper)
            dist_pct = dist_to_break / current_close * 100 if current_close > 0 else 0
            vol_flag = "Decreasing during flag (textbook)" if volume is not None else "N/A"
            stop_loss = round(breakout_lower - atr, 2)
            description = (
                f"BULLISH: Strong impulse +{pole_return * 100:.1f}% in {pole_len} bars, now consolidating\n"
                f"Pole: ${close[pole_start]:.2f} to ${close[pole_end]:.2f} (+{pole_return * 100:.1f}%)\n"
                f"Flag channel: ${chan_low:.2f} - ${chan_high:.2f} "
                f"(range: ${chan_width:.2f}, {chan_width_pct:.1f}% of price)\n"
                f"Flag drift: {flag_drift * 100:.1f}% over {consol_len} bars (healthy pullback)\n"
                f"Volume: {vol_flag}\n"
                f"Breakout level: ${breakout_upper:.2f} | Current: ${current_close:.2f} | "
                f"Distance: ${dist_to_break:.2f} ({dist_pct:.1f}%)\n"
                f"14-day ATR: ${atr:.2f} | Suggested stop: ${stop_loss:.2f} (below flag)\n"
                f"Watch for: Close above ${breakout_upper:.2f} to trigger continuation"
            )

            return {
                "pattern": "Bull Flag",
                "confidence": round(confidence, 1),
                "key_points": key_points,
                "description": description,
                "state": state,
                "compression_ratio": round(compression_ratio, 4),
                "breakout_level_upper": round(breakout_upper, 4),
                "breakout_level_lower": round(breakout_lower, 4),
                "distance_to_breakout": round(distance_to_breakout, 4),
                "lines": [
                    [(df.index[consol_start], chan_high), (df.index[-1], chan_high)],
                    [(df.index[consol_start], chan_low), (df.index[-1], chan_low)],
                ],
                "upper_trendline": (0.0, breakout_upper),
                "lower_trendline": (0.0, breakout_lower),
            }

    return None


def _detect_bear_flag(df: pd.DataFrame) -> Optional[PatternResult]:
    """Detect Bear Flag IN PROGRESS - price must still be inside the channel.

    Rules
    -----
    1. Impulse leg: >8% drop in 3-12 candles
    2. Consolidation: 5-15 candles with slight upward drift (<5%)
    3. Price is STILL inside the flag channel (not broken out yet)
    4. Volume decreasing during flag
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values if "volume" in df.columns else None
    n = len(close)
    if n < 15:
        return None

    for consol_start in range(max(1, n - 15), n - 4):
        consol_end = n - 1
        consol_len = consol_end - consol_start

        if consol_len < 5 or consol_len > 15:
            continue

        consol_high_vals = high[consol_start: consol_end + 1]
        consol_low_vals = low[consol_start: consol_end + 1]
        consol_close_vals = close[consol_start: consol_end + 1]

        chan_high = float(np.max(consol_high_vals))
        chan_low = float(np.min(consol_low_vals))
        chan_width = chan_high - chan_low
        if chan_width <= 0:
            continue

        current_close = float(close[-1])
        if current_close > chan_high or current_close < chan_low:
            continue

        # Slight upward drift (bear flag): between -1% and +5%
        flag_drift = (consol_close_vals[-1] - consol_close_vals[0]) / consol_close_vals[0]
        if not (-0.01 <= flag_drift <= 0.05):
            continue

        if volume is not None and consol_len >= 3:
            vol_slope = _linreg(volume[consol_start: consol_end + 1])[0]
            if vol_slope > 0:
                continue

        for pole_len in range(3, 13):
            pole_end = consol_start
            pole_start = pole_end - pole_len
            if pole_start < 0:
                break

            pole_return = (close[pole_end] - close[pole_start]) / close[pole_start]
            if pole_return > -0.08:
                continue

            if chan_width / abs(close[pole_start]) > FLAG_WIDTH_THRESHOLD:
                continue

            breakout_upper = chan_high
            breakout_lower = chan_low
            atr = _compute_atr(df)
            distance_to_breakout = min(
                abs(current_close - breakout_upper),
                abs(current_close - breakout_lower),
            )

            state = (
                PatternState.NEAR_BREAK
                if distance_to_breakout <= 0.25 * atr
                else PatternState.FORMING
            )

            compression_ratio = (
                chan_width / (abs(close[pole_start]) * FLAG_WIDTH_THRESHOLD)
                if close[pole_start] != 0 else 0.5
            )
            compression_ratio = min(compression_ratio, 1.0)

            confidence = min(100.0, 65.0 + min(abs(pole_return) * 100, 20.0))
            if state == PatternState.NEAR_BREAK:
                confidence = min(confidence + 10, 100.0)

            key_points: List[KeyPoint] = [
                (df.index[pole_start], float(close[pole_start]), "Pole Start"),
                (df.index[pole_end], float(close[pole_end]), "Pole Bottom"),
                (df.index[-1], current_close, "Flag End"),
            ]

            chan_width_pct = chan_width / abs(close[pole_start]) * 100 if close[pole_start] != 0 else 0
            dist_to_break = abs(current_close - breakout_lower)
            dist_pct = dist_to_break / current_close * 100 if current_close > 0 else 0
            vol_flag = "Decreasing during flag (textbook)" if volume is not None else "N/A"
            stop_loss = round(breakout_upper + atr, 2)
            description = (
                f"BEARISH: Strong drop {pole_return * 100:.1f}% in {pole_len} bars, now consolidating\n"
                f"Pole: ${close[pole_start]:.2f} to ${close[pole_end]:.2f} ({pole_return * 100:.1f}%)\n"
                f"Flag channel: ${chan_low:.2f} - ${chan_high:.2f} "
                f"(range: ${chan_width:.2f}, {chan_width_pct:.1f}% of price)\n"
                f"Flag drift: +{flag_drift * 100:.1f}% over {consol_len} bars (healthy bounce)\n"
                f"Volume: {vol_flag}\n"
                f"Breakdown level: ${breakout_lower:.2f} | Current: ${current_close:.2f} | "
                f"Distance: ${dist_to_break:.2f} ({dist_pct:.1f}%)\n"
                f"14-day ATR: ${atr:.2f} | Suggested stop: ${stop_loss:.2f} (above flag)\n"
                f"Watch for: Close below ${breakout_lower:.2f} to trigger continuation"
            )

            return {
                "pattern": "Bear Flag",
                "confidence": round(confidence, 1),
                "key_points": key_points,
                "description": description,
                "state": state,
                "compression_ratio": round(compression_ratio, 4),
                "breakout_level_upper": round(breakout_upper, 4),
                "breakout_level_lower": round(breakout_lower, 4),
                "distance_to_breakout": round(distance_to_breakout, 4),
                "lines": [
                    [(df.index[consol_start], chan_high), (df.index[-1], chan_high)],
                    [(df.index[consol_start], chan_low), (df.index[-1], chan_low)],
                ],
                "upper_trendline": (0.0, breakout_upper),
                "lower_trendline": (0.0, breakout_lower),
            }

    return None


# ---------------------------------------------------------------------------
# New pattern detectors
# ---------------------------------------------------------------------------


def _detect_double_bottom(df: pd.DataFrame) -> Optional[PatternResult]:
    """Detect Double Bottom (W-shape) IN PROGRESS — before breakout above neckline."""
    n = len(df)
    if n < 15:
        return None

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values if "volume" in df.columns else None

    _, pivot_lows = _find_strict_pivots(df)
    if len(pivot_lows) < 2:
        return None

    atr = _compute_atr(df)
    current_price = float(close[-1])
    end_date = df.index[-1]

    for j in range(len(pivot_lows) - 1, 0, -1):
        for i in range(j - 1, -1, -1):
            idx1, idx2 = pivot_lows[i], pivot_lows[j]
            bar_distance = idx2 - idx1
            if not (7 <= bar_distance <= 30):
                continue

            trough1 = float(low[idx1])
            trough2 = float(low[idx2])
            avg_trough = (trough1 + trough2) / 2
            if avg_trough <= 0:
                continue
            spread = abs(trough1 - trough2) / avg_trough
            if spread > 0.03:
                continue

            neckline = float(np.max(high[idx1:idx2 + 1]))
            if current_price >= neckline:
                continue
            if current_price < avg_trough - atr:
                continue

            pattern_height = neckline - avg_trough
            if pattern_height <= 0:
                continue

            distance_to_breakout = abs(current_price - neckline)
            dist_pct = distance_to_breakout / current_price * 100 if current_price > 0 else 0

            state = (
                PatternState.NEAR_BREAK
                if distance_to_breakout <= 0.5 * atr
                else PatternState.FORMING
            )

            compression_ratio = min(distance_to_breakout / pattern_height, 1.0)
            confidence = _compression_confidence(1.0 - compression_ratio, state)
            if confidence < 60:
                continue

            date1, date2 = df.index[idx1], df.index[idx2]
            start_date = date1
            vol_desc = "N/A"
            if volume is not None:
                vol_desc = _vol_trend_desc(volume, idx1)

            stop_loss = round(min(trough1, trough2) - atr, 2)
            description = (
                f"BULLISH: W-shape forming at ${avg_trough:.2f} support zone\n"
                f"Trough 1: ${trough1:.2f} ({_fmt_date(date1)}) | "
                f"Trough 2: ${trough2:.2f} ({_fmt_date(date2)}) | Spread: {spread * 100:.1f}%\n"
                f"Neckline (breakout level): ${neckline:.2f}\n"
                f"Current price: ${current_price:.2f} | "
                f"Distance to breakout: ${distance_to_breakout:.2f} ({dist_pct:.1f}%)\n"
                f"Volume: {vol_desc}\n"
                f"14-day ATR: ${atr:.2f} | Suggested stop: ${stop_loss:.2f} (below troughs)\n"
                f"Watch for: Close above ${neckline:.2f} with volume > 20-day avg to confirm"
            )

            key_points: List[KeyPoint] = [
                (date1, trough1, "Trough 1"),
                (date2, trough2, "Trough 2"),
                (end_date, current_price, "Current"),
            ]

            return {
                "pattern": "Double Bottom",
                "confidence": round(confidence, 1),
                "key_points": key_points,
                "description": description,
                "state": state,
                "compression_ratio": round(compression_ratio, 4),
                "breakout_level_upper": round(neckline, 4),
                "breakout_level_lower": round(avg_trough, 4),
                "distance_to_breakout": round(distance_to_breakout, 4),
                "lines": [
                    [(start_date, neckline), (end_date, neckline)],
                    [(start_date, avg_trough), (end_date, avg_trough)],
                ],
                "upper_trendline": (0.0, neckline),
                "lower_trendline": (0.0, avg_trough),
            }

    return None


def _detect_double_top(df: pd.DataFrame) -> Optional[PatternResult]:
    """Detect Double Top (M-shape) IN PROGRESS — before breakdown below neckline."""
    n = len(df)
    if n < 15:
        return None

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values if "volume" in df.columns else None

    pivot_highs, _ = _find_strict_pivots(df)
    if len(pivot_highs) < 2:
        return None

    atr = _compute_atr(df)
    current_price = float(close[-1])
    end_date = df.index[-1]

    for j in range(len(pivot_highs) - 1, 0, -1):
        for i in range(j - 1, -1, -1):
            idx1, idx2 = pivot_highs[i], pivot_highs[j]
            bar_distance = idx2 - idx1
            if not (7 <= bar_distance <= 30):
                continue

            peak1 = float(high[idx1])
            peak2 = float(high[idx2])
            avg_peak = (peak1 + peak2) / 2
            if avg_peak <= 0:
                continue
            spread = abs(peak1 - peak2) / avg_peak
            if spread > 0.03:
                continue

            neckline = float(np.min(low[idx1:idx2 + 1]))
            if current_price <= neckline:
                continue
            if current_price > avg_peak + atr:
                continue

            pattern_height = avg_peak - neckline
            if pattern_height <= 0:
                continue

            distance_to_breakout = abs(current_price - neckline)
            dist_pct = distance_to_breakout / current_price * 100 if current_price > 0 else 0

            state = (
                PatternState.NEAR_BREAK
                if distance_to_breakout <= 0.5 * atr
                else PatternState.FORMING
            )

            compression_ratio = min(distance_to_breakout / pattern_height, 1.0)
            confidence = _compression_confidence(1.0 - compression_ratio, state)
            if confidence < 60:
                continue

            date1, date2 = df.index[idx1], df.index[idx2]
            start_date = date1
            vol_desc = "N/A"
            if volume is not None:
                vol_desc = _vol_trend_desc(volume, idx1)

            stop_loss = round(max(peak1, peak2) + atr, 2)
            description = (
                f"BEARISH: M-shape forming at ${avg_peak:.2f} resistance zone\n"
                f"Peak 1: ${peak1:.2f} ({_fmt_date(date1)}) | "
                f"Peak 2: ${peak2:.2f} ({_fmt_date(date2)}) | Spread: {spread * 100:.1f}%\n"
                f"Neckline (breakdown level): ${neckline:.2f}\n"
                f"Current price: ${current_price:.2f} | "
                f"Distance to breakdown: ${distance_to_breakout:.2f} ({dist_pct:.1f}%)\n"
                f"Volume: {vol_desc}\n"
                f"14-day ATR: ${atr:.2f} | Suggested stop: ${stop_loss:.2f} (above peaks)\n"
                f"Watch for: Close below ${neckline:.2f} on heavy volume to confirm"
            )

            key_points: List[KeyPoint] = [
                (date1, peak1, "Peak 1"),
                (date2, peak2, "Peak 2"),
                (end_date, current_price, "Current"),
            ]

            return {
                "pattern": "Double Top",
                "confidence": round(confidence, 1),
                "key_points": key_points,
                "description": description,
                "state": state,
                "compression_ratio": round(compression_ratio, 4),
                "breakout_level_upper": round(avg_peak, 4),
                "breakout_level_lower": round(neckline, 4),
                "distance_to_breakout": round(distance_to_breakout, 4),
                "lines": [
                    [(start_date, avg_peak), (end_date, avg_peak)],
                    [(start_date, neckline), (end_date, neckline)],
                ],
                "upper_trendline": (0.0, avg_peak),
                "lower_trendline": (0.0, neckline),
            }

    return None


def _detect_head_and_shoulders(df: pd.DataFrame) -> Optional[PatternResult]:
    """Detect Head and Shoulders IN PROGRESS — before breakdown below neckline."""
    n = len(df)
    if n < 15:
        return None

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values if "volume" in df.columns else None

    pivot_highs, _ = _find_strict_pivots(df)
    if len(pivot_highs) < 3:
        return None

    atr = _compute_atr(df)
    current_price = float(close[-1])
    end_date = df.index[-1]

    for i in range(len(pivot_highs) - 2):
        for j in range(i + 1, len(pivot_highs) - 1):
            for k in range(j + 1, len(pivot_highs)):
                ls_idx = pivot_highs[i]
                head_idx = pivot_highs[j]
                rs_idx = pivot_highs[k]

                ls_price = float(high[ls_idx])
                head_price = float(high[head_idx])
                rs_price = float(high[rs_idx])

                if head_price <= ls_price or head_price <= rs_price:
                    continue

                shoulder_avg = (ls_price + rs_price) / 2
                if shoulder_avg <= 0:
                    continue
                shoulder_sym = abs(ls_price - rs_price) / shoulder_avg
                if shoulder_sym > 0.05:
                    continue

                lhs_trough = float(np.min(low[ls_idx:head_idx + 1]))
                rhs_trough = float(np.min(low[head_idx:rs_idx + 1]))
                neckline = (lhs_trough + rhs_trough) / 2

                if current_price <= neckline:
                    continue

                head_prom = (head_price - shoulder_avg) / shoulder_avg
                distance_to_breakout = current_price - neckline
                dist_pct = distance_to_breakout / current_price * 100 if current_price > 0 else 0

                state = (
                    PatternState.NEAR_BREAK
                    if distance_to_breakout <= 0.5 * atr
                    else PatternState.FORMING
                )

                pattern_height = head_price - neckline
                compression_ratio = min(
                    distance_to_breakout / pattern_height if pattern_height > 0 else 0.5, 1.0
                )

                base_conf = 70.0 if state == PatternState.NEAR_BREAK else 60.0
                head_bonus = min(head_prom * 100, 15.0)
                sym_bonus = max(0.0, (0.05 - shoulder_sym) / 0.05) * 10.0
                confidence = min(100.0, base_conf + head_bonus + sym_bonus)

                date_ls = df.index[ls_idx]
                date_head = df.index[head_idx]
                date_rs = df.index[rs_idx]
                vol_desc = "N/A"
                if volume is not None:
                    vol_desc = _vol_trend_desc(volume, ls_idx)

                stop_loss = round(rs_price + atr, 2)
                symmetry_pct = (1.0 - shoulder_sym) * 100
                description = (
                    f"BEARISH: Classic H&S forming — distribution pattern\n"
                    f"Left Shoulder: ${ls_price:.2f} ({_fmt_date(date_ls)}) | "
                    f"Head: ${head_price:.2f} ({_fmt_date(date_head)}) | "
                    f"Right Shoulder: ${rs_price:.2f} ({_fmt_date(date_rs)})\n"
                    f"Shoulder symmetry: {symmetry_pct:.1f}% | "
                    f"Head prominence: {head_prom * 100:.1f}% above shoulders\n"
                    f"Neckline: ${neckline:.2f} | Current: ${current_price:.2f} | "
                    f"Distance to breakdown: ${distance_to_breakout:.2f} ({dist_pct:.1f}%)\n"
                    f"Volume: {vol_desc}\n"
                    f"14-day ATR: ${atr:.2f} | Suggested stop: ${stop_loss:.2f} (above right shoulder)\n"
                    f"Watch for: Close below ${neckline:.2f} on heavy volume to confirm breakdown"
                )

                key_points: List[KeyPoint] = [
                    (date_ls, ls_price, "Left Shoulder"),
                    (date_head, head_price, "Head"),
                    (date_rs, rs_price, "Right Shoulder"),
                    (end_date, current_price, "Current"),
                ]

                return {
                    "pattern": "Head and Shoulders",
                    "confidence": round(confidence, 1),
                    "key_points": key_points,
                    "description": description,
                    "state": state,
                    "compression_ratio": round(compression_ratio, 4),
                    "breakout_level_upper": round(head_price, 4),
                    "breakout_level_lower": round(neckline, 4),
                    "distance_to_breakout": round(distance_to_breakout, 4),
                    "lines": [
                        [(date_ls, neckline), (end_date, neckline)],
                    ],
                    "upper_trendline": (0.0, head_price),
                    "lower_trendline": (0.0, neckline),
                }

    return None


def _detect_inverse_head_and_shoulders(df: pd.DataFrame) -> Optional[PatternResult]:
    """Detect Inverse Head and Shoulders IN PROGRESS — before breakout above neckline."""
    n = len(df)
    if n < 15:
        return None

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values if "volume" in df.columns else None

    _, pivot_lows = _find_strict_pivots(df)
    if len(pivot_lows) < 3:
        return None

    atr = _compute_atr(df)
    current_price = float(close[-1])
    end_date = df.index[-1]

    for i in range(len(pivot_lows) - 2):
        for j in range(i + 1, len(pivot_lows) - 1):
            for k in range(j + 1, len(pivot_lows)):
                ls_idx = pivot_lows[i]
                head_idx = pivot_lows[j]
                rs_idx = pivot_lows[k]

                ls_price = float(low[ls_idx])
                head_price = float(low[head_idx])
                rs_price = float(low[rs_idx])

                if head_price >= ls_price or head_price >= rs_price:
                    continue

                shoulder_avg = (ls_price + rs_price) / 2
                if shoulder_avg <= 0:
                    continue
                shoulder_sym = abs(ls_price - rs_price) / shoulder_avg
                if shoulder_sym > 0.05:
                    continue

                lhs_peak = float(np.max(high[ls_idx:head_idx + 1]))
                rhs_peak = float(np.max(high[head_idx:rs_idx + 1]))
                neckline = (lhs_peak + rhs_peak) / 2

                if current_price >= neckline:
                    continue

                head_prom = (shoulder_avg - head_price) / shoulder_avg
                distance_to_breakout = neckline - current_price
                dist_pct = distance_to_breakout / current_price * 100 if current_price > 0 else 0

                state = (
                    PatternState.NEAR_BREAK
                    if distance_to_breakout <= 0.5 * atr
                    else PatternState.FORMING
                )

                pattern_height = neckline - head_price
                compression_ratio = min(
                    distance_to_breakout / pattern_height if pattern_height > 0 else 0.5, 1.0
                )

                base_conf = 70.0 if state == PatternState.NEAR_BREAK else 60.0
                head_bonus = min(head_prom * 100, 15.0)
                sym_bonus = max(0.0, (0.05 - shoulder_sym) / 0.05) * 10.0
                confidence = min(100.0, base_conf + head_bonus + sym_bonus)

                date_ls = df.index[ls_idx]
                date_head = df.index[head_idx]
                date_rs = df.index[rs_idx]
                vol_desc = "N/A"
                if volume is not None:
                    vol_desc = _vol_trend_desc(volume, ls_idx)

                stop_loss = round(rs_price - atr, 2)
                symmetry_pct = (1.0 - shoulder_sym) * 100
                description = (
                    f"BULLISH: Inverse H&S forming — accumulation pattern\n"
                    f"Left Shoulder: ${ls_price:.2f} ({_fmt_date(date_ls)}) | "
                    f"Head: ${head_price:.2f} ({_fmt_date(date_head)}) | "
                    f"Right Shoulder: ${rs_price:.2f} ({_fmt_date(date_rs)})\n"
                    f"Shoulder symmetry: {symmetry_pct:.1f}% | "
                    f"Head depth: {head_prom * 100:.1f}% below shoulders\n"
                    f"Neckline: ${neckline:.2f} | Current: ${current_price:.2f} | "
                    f"Distance to breakout: ${distance_to_breakout:.2f} ({dist_pct:.1f}%)\n"
                    f"Volume: {vol_desc}\n"
                    f"14-day ATR: ${atr:.2f} | Suggested stop: ${stop_loss:.2f} (below right shoulder)\n"
                    f"Watch for: Close above ${neckline:.2f} with volume surge to confirm"
                )

                key_points: List[KeyPoint] = [
                    (date_ls, ls_price, "Left Shoulder"),
                    (date_head, head_price, "Head"),
                    (date_rs, rs_price, "Right Shoulder"),
                    (end_date, current_price, "Current"),
                ]

                return {
                    "pattern": "Inverse Head and Shoulders",
                    "confidence": round(confidence, 1),
                    "key_points": key_points,
                    "description": description,
                    "state": state,
                    "compression_ratio": round(compression_ratio, 4),
                    "breakout_level_upper": round(neckline, 4),
                    "breakout_level_lower": round(head_price, 4),
                    "distance_to_breakout": round(distance_to_breakout, 4),
                    "lines": [
                        [(date_ls, neckline), (end_date, neckline)],
                    ],
                    "upper_trendline": (0.0, neckline),
                    "lower_trendline": (0.0, head_price),
                }

    return None


def _detect_cup_and_handle(df: pd.DataFrame) -> Optional[PatternResult]:
    """Detect Cup and Handle IN PROGRESS — handle forming, before breakout above cup rim."""
    n = len(df)
    if n < 20:
        return None

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values if "volume" in df.columns else None

    atr = _compute_atr(df)
    current_price = float(close[-1])
    end_date = df.index[-1]

    for handle_start in range(max(3, n - 10), n - 2):
        handle_len = n - 1 - handle_start
        if handle_len < 3 or handle_len > 10:
            continue

        cup_start = 0
        cup_end = handle_start
        if cup_end - cup_start < 10:
            continue

        half = (cup_end - cup_start) // 2
        left_rim_rel = int(np.argmax(high[cup_start:cup_start + half]))
        left_rim_idx = left_rim_rel + cup_start
        left_rim = float(high[left_rim_idx])

        bottom_rel = int(np.argmin(low[cup_start:cup_end]))
        bottom_idx = bottom_rel + cup_start
        cup_bottom = float(low[bottom_idx])

        cup_depth = left_rim - cup_bottom
        if cup_depth < atr * 2:
            continue

        right_half_start = cup_start + half
        right_rim_rel = int(np.argmax(high[right_half_start:cup_end]))
        right_rim_idx = right_rim_rel + right_half_start
        right_rim = float(high[right_rim_idx])

        rim_diff = abs(right_rim - left_rim) / left_rim if left_rim > 0 else 1.0
        if rim_diff > 0.05:
            continue

        handle_high = float(np.max(high[handle_start:n]))
        handle_low = float(np.min(low[handle_start:n]))
        handle_drop = (right_rim - handle_low) / cup_depth if cup_depth > 0 else 1.0
        if handle_drop > 0.5 or handle_drop <= 0:
            continue

        if current_price > handle_high or current_price < handle_low:
            continue

        breakout_level = (left_rim + right_rim) / 2
        distance_to_breakout = abs(current_price - breakout_level)
        dist_pct = distance_to_breakout / current_price * 100 if current_price > 0 else 0

        state = (
            PatternState.NEAR_BREAK
            if distance_to_breakout <= 0.5 * atr
            else PatternState.FORMING
        )

        handle_width = handle_high - handle_low
        compression_ratio = min(handle_width / cup_depth if cup_depth > 0 else 0.5, 1.0)
        base_conf = 70.0 if state == PatternState.NEAR_BREAK else 62.0
        depth_bonus = min(cup_depth / atr, 5.0) * 2
        confidence = min(100.0, base_conf + depth_bonus)

        vol_desc = "N/A"
        if volume is not None:
            vol_desc = _vol_trend_desc(volume, handle_start)

        date_left_rim = df.index[left_rim_idx]
        date_bottom = df.index[bottom_idx]
        date_handle_start = df.index[handle_start]
        stop_loss = round(handle_low - atr, 2)

        description = (
            f"BULLISH: U-shaped cup with handle forming\n"
            f"Cup: left rim ${left_rim:.2f} ({_fmt_date(date_left_rim)}) → "
            f"bottom ${cup_bottom:.2f} ({_fmt_date(date_bottom)}) → recovery\n"
            f"Cup depth: ${cup_depth:.2f} | Handle drop: {handle_drop * 100:.1f}% of cup (healthy: <50%)\n"
            f"Breakout level (cup rim): ${breakout_level:.2f}\n"
            f"Current price: ${current_price:.2f} | "
            f"Distance to breakout: ${distance_to_breakout:.2f} ({dist_pct:.1f}%)\n"
            f"Volume: {vol_desc} during handle (ideal: declining)\n"
            f"14-day ATR: ${atr:.2f} | Suggested stop: ${stop_loss:.2f} (below handle)\n"
            f"Watch for: Close above ${breakout_level:.2f} with volume surge to confirm"
        )

        key_points: List[KeyPoint] = [
            (date_left_rim, left_rim, "Left Rim"),
            (date_bottom, cup_bottom, "Cup Bottom"),
            (date_handle_start, handle_high, "Handle Start"),
            (end_date, current_price, "Current"),
        ]

        start_date = df.index[cup_start]
        return {
            "pattern": "Cup and Handle",
            "confidence": round(confidence, 1),
            "key_points": key_points,
            "description": description,
            "state": state,
            "compression_ratio": round(compression_ratio, 4),
            "breakout_level_upper": round(breakout_level, 4),
            "breakout_level_lower": round(handle_low, 4),
            "distance_to_breakout": round(distance_to_breakout, 4),
            "lines": [
                [(start_date, breakout_level), (end_date, breakout_level)],
                [(date_handle_start, handle_low), (end_date, handle_low)],
            ],
            "upper_trendline": (0.0, breakout_level),
            "lower_trendline": (0.0, handle_low),
        }

    return None


def _detect_symmetrical_triangle(df: pd.DataFrame) -> Optional[PatternResult]:
    """Detect Symmetrical Triangle — both trendlines converging, neither flat."""
    n = len(df)
    if n < 15:
        return None

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    pivot_highs, pivot_lows = _find_strict_pivots(df)
    if len(pivot_highs) < 2 or len(pivot_lows) < 2:
        return None

    ph_x = np.array(pivot_highs, dtype=float)
    ph_y = high[pivot_highs]
    pl_x = np.array(pivot_lows, dtype=float)
    pl_y = low[pivot_lows]

    resist_slope, resist_intercept = _linreg(ph_y, ph_x)
    support_slope, support_intercept = _linreg(pl_y, pl_x)

    # Symmetrical: resistance falling (negative), support rising (positive)
    if resist_slope >= -0.01 or support_slope <= 0.01:
        return None
    # Neither flat — that would be ascending/descending triangle
    if abs(resist_slope) < 0.01 or abs(support_slope) < 0.01:
        return None

    start_idx = float(min(pivot_highs[0], pivot_lows[0]))
    end_idx = float(n - 1)

    upper_at_start = resist_slope * start_idx + resist_intercept
    lower_at_start = support_slope * start_idx + support_intercept
    initial_width = abs(upper_at_start - lower_at_start)
    upper_at_current = resist_slope * end_idx + resist_intercept
    lower_at_current = support_slope * end_idx + support_intercept
    current_width = abs(upper_at_current - lower_at_current)

    if initial_width <= 0 or current_width <= 0:
        return None
    compression_ratio = current_width / initial_width
    if compression_ratio >= 1.0:
        return None

    current_price = float(close[-1])
    atr = _compute_atr(df)

    state = _determine_state(
        compression_ratio, current_price,
        upper_at_current, lower_at_current,
        atr, upper_at_current, lower_at_current,
    )
    if state is None:
        return None

    start_idx_int = int(start_idx)
    start_date = df.index[start_idx_int]
    end_date = df.index[-1]

    distance_to_breakout = min(
        abs(current_price - upper_at_current),
        abs(current_price - lower_at_current),
    )
    dist_pct = distance_to_breakout / current_price * 100 if current_price > 0 else 0
    confidence = _compression_confidence(compression_ratio, state)

    volume_arr = df["volume"].values if "volume" in df.columns else None
    vol_desc = "N/A"
    if volume_arr is not None:
        vol_desc = _vol_trend_desc(volume_arr, start_idx_int)

    stop_loss_long = round(lower_at_current - atr, 2)
    stop_loss_short = round(upper_at_current + atr, 2)
    description = (
        f"NEUTRAL: Symmetrical triangle — converging trendlines, breakout imminent\n"
        f"Resistance: falling from ${upper_at_start:.2f} to ${upper_at_current:.2f}\n"
        f"Support: rising from ${lower_at_start:.2f} to ${lower_at_current:.2f}\n"
        f"Compression: {compression_ratio * 100:.0f}% "
        f"(from ${initial_width:.2f} to ${current_width:.2f} range)\n"
        f"Current price: ${current_price:.2f} | "
        f"Distance to breakout: ${distance_to_breakout:.2f} ({dist_pct:.1f}%)\n"
        f"Volume: {vol_desc}\n"
        f"14-day ATR: ${atr:.2f} | Stop long: ${stop_loss_long:.2f} | Stop short: ${stop_loss_short:.2f}\n"
        f"Watch for: Close outside triangle with volume surge to determine direction"
    )

    key_points: List[KeyPoint] = [
        (start_date, upper_at_start, "Resistance Start"),
        (end_date, upper_at_current, "Resistance End"),
        (start_date, lower_at_start, "Support Start"),
        (end_date, lower_at_current, "Support End"),
    ]

    return {
        "pattern": "Symmetrical Triangle",
        "confidence": round(confidence, 1),
        "key_points": key_points,
        "description": description,
        "state": state,
        "compression_ratio": round(compression_ratio, 4),
        "breakout_level_upper": round(upper_at_current, 4),
        "breakout_level_lower": round(lower_at_current, 4),
        "distance_to_breakout": round(distance_to_breakout, 4),
        "lines": [
            [(start_date, upper_at_start), (end_date, upper_at_current)],
            [(start_date, lower_at_start), (end_date, lower_at_current)],
        ],
        "upper_trendline": (resist_slope, resist_intercept),
        "lower_trendline": (support_slope, support_intercept),
    }


def _detect_pennant(df: pd.DataFrame) -> Optional[PatternResult]:
    """Detect Pennant IN PROGRESS — small symmetrical triangle after strong impulse."""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values if "volume" in df.columns else None
    n = len(close)
    if n < 15:
        return None

    atr = _compute_atr(df)
    current_price = float(close[-1])

    for consol_start in range(max(1, n - 10), n - 3):
        consol_len = n - 1 - consol_start
        if consol_len < 3 or consol_len > 10:
            continue

        c_high = high[consol_start:]
        c_low = low[consol_start:]
        x = np.arange(len(c_high), dtype=float)
        resist_slope, resist_intercept = _linreg(c_high, x)
        support_slope, support_intercept = _linreg(c_low, x)

        # Pennant: converging (resistance falling, support rising)
        if resist_slope >= 0 or support_slope <= 0:
            continue

        width_start = resist_intercept - support_intercept
        x_end = float(len(c_high) - 1)
        width_end = (resist_slope * x_end + resist_intercept) - (support_slope * x_end + support_intercept)
        if width_end >= width_start or width_start <= 0:
            continue

        upper_current = resist_slope * x_end + resist_intercept
        lower_current = support_slope * x_end + support_intercept
        if current_price > upper_current or current_price < lower_current:
            continue

        for pole_len in range(3, 13):
            pole_end = consol_start
            pole_start = pole_end - pole_len
            if pole_start < 0:
                break
            pole_return = (close[pole_end] - close[pole_start]) / close[pole_start]
            if abs(pole_return) < 0.08:
                continue

            is_bullish = pole_return > 0
            compression_ratio = min(width_end / width_start, 1.0) if width_start > 0 else 0.5
            distance_to_breakout = min(
                abs(current_price - upper_current),
                abs(current_price - lower_current),
            )
            dist_pct = distance_to_breakout / current_price * 100 if current_price > 0 else 0
            state = (
                PatternState.NEAR_BREAK
                if distance_to_breakout <= 0.25 * atr or compression_ratio < 0.35
                else PatternState.FORMING
            )
            confidence = min(100.0, 65.0 + min(abs(pole_return) * 100, 20.0))
            if state == PatternState.NEAR_BREAK:
                confidence = min(confidence + 10, 100.0)

            vol_desc = "N/A"
            if volume is not None:
                vol_desc = _vol_trend_desc(volume, consol_start)

            pole_start_date = df.index[pole_start]
            pole_end_date = df.index[pole_end]
            consol_start_date = df.index[consol_start]
            end_date = df.index[-1]

            direction = "BULLISH" if is_bullish else "BEARISH"
            pole_sign = "+" if is_bullish else ""
            stop_loss = round(lower_current - atr, 2) if is_bullish else round(upper_current + atr, 2)
            description = (
                f"{direction}: Pennant — tight convergence after strong impulse\n"
                f"Pole: ${close[pole_start]:.2f} → ${close[pole_end]:.2f} "
                f"({pole_sign}{pole_return * 100:.1f}% in {pole_len} bars)\n"
                f"Pennant: resistance ${resist_intercept:.2f}↘ | support ${support_intercept:.2f}↗\n"
                f"Compression: {compression_ratio * 100:.0f}% | "
                f"Upper: ${upper_current:.2f} | Lower: ${lower_current:.2f}\n"
                f"Current price: ${current_price:.2f} | "
                f"Distance to breakout: ${distance_to_breakout:.2f} ({dist_pct:.1f}%)\n"
                f"Volume: {vol_desc}\n"
                f"14-day ATR: ${atr:.2f} | Suggested stop: ${stop_loss:.2f}\n"
                f"Watch for: Close outside pennant in direction of impulse"
            )

            key_points: List[KeyPoint] = [
                (pole_start_date, float(close[pole_start]), "Pole Start"),
                (pole_end_date, float(close[pole_end]), "Pole Top" if is_bullish else "Pole Bottom"),
                (end_date, current_price, "Pennant End"),
            ]

            return {
                "pattern": "Pennant",
                "confidence": round(confidence, 1),
                "key_points": key_points,
                "description": description,
                "state": state,
                "compression_ratio": round(compression_ratio, 4),
                "breakout_level_upper": round(upper_current, 4),
                "breakout_level_lower": round(lower_current, 4),
                "distance_to_breakout": round(distance_to_breakout, 4),
                "lines": [
                    [(consol_start_date, float(resist_intercept)), (end_date, float(upper_current))],
                    [(consol_start_date, float(support_intercept)), (end_date, float(lower_current))],
                ],
                "upper_trendline": (resist_slope, resist_intercept),
                "lower_trendline": (support_slope, support_intercept),
            }

    return None


def _detect_channel(df: pd.DataFrame, up: bool) -> Optional[PatternResult]:
    """Detect Channel Up or Channel Down — parallel trendlines in the same direction."""
    n = len(df)
    if n < 15:
        return None

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    pivot_highs, pivot_lows = _find_strict_pivots(df)
    if len(pivot_highs) < 2 or len(pivot_lows) < 2:
        return None

    ph_x = np.array(pivot_highs, dtype=float)
    ph_y = high[pivot_highs]
    pl_x = np.array(pivot_lows, dtype=float)
    pl_y = low[pivot_lows]

    resist_slope, resist_intercept = _linreg(ph_y, ph_x)
    support_slope, support_intercept = _linreg(pl_y, pl_x)

    pattern_name = "Channel Up" if up else "Channel Down"
    if up:
        if resist_slope <= 0 or support_slope <= 0:
            return None
    else:
        if resist_slope >= 0 or support_slope >= 0:
            return None

    avg_slope = (abs(resist_slope) + abs(support_slope)) / 2
    if avg_slope == 0:
        return None
    if abs(resist_slope - support_slope) / avg_slope > 0.5:
        return None

    start_idx = float(min(pivot_highs[0], pivot_lows[0]))
    end_idx = float(n - 1)

    upper_at_start = resist_slope * start_idx + resist_intercept
    lower_at_start = support_slope * start_idx + support_intercept
    initial_width = abs(upper_at_start - lower_at_start)
    upper_at_current = resist_slope * end_idx + resist_intercept
    lower_at_current = support_slope * end_idx + support_intercept
    current_width = abs(upper_at_current - lower_at_current)

    if initial_width <= 0 or current_width <= 0:
        return None

    # Channel: width should be roughly constant (not converging/expanding too much)
    width_ratio = current_width / initial_width
    if width_ratio < 0.7 or width_ratio > 1.4:
        return None

    current_price = float(close[-1])
    if current_price > upper_at_current or current_price < lower_at_current:
        return None

    atr = _compute_atr(df)
    distance_to_upper = abs(current_price - upper_at_current)
    distance_to_lower = abs(current_price - lower_at_current)
    distance_to_breakout = min(distance_to_upper, distance_to_lower)
    dist_pct = distance_to_breakout / current_price * 100 if current_price > 0 else 0

    state = (
        PatternState.NEAR_BREAK
        if distance_to_breakout <= 0.25 * atr
        else PatternState.FORMING
    )

    compression_ratio = 0.5  # Channel maintains width — use midpoint
    confidence = 65.0 if state == PatternState.NEAR_BREAK else 62.0

    start_idx_int = int(start_idx)
    start_date = df.index[start_idx_int]
    end_date = df.index[-1]

    volume_arr = df["volume"].values if "volume" in df.columns else None
    vol_desc = "N/A"
    if volume_arr is not None:
        vol_desc = _vol_trend_desc(volume_arr, start_idx_int)

    direction = "BULLISH" if up else "BEARISH"
    stop_long = round(lower_at_current - atr, 2)
    stop_short = round(upper_at_current + atr, 2)
    description = (
        f"{direction}: {pattern_name} — parallel channel, price inside\n"
        f"Upper channel: ${upper_at_current:.2f} (slope: {resist_slope:+.3f}/bar) | "
        f"Lower channel: ${lower_at_current:.2f} (slope: {support_slope:+.3f}/bar)\n"
        f"Channel width: ${current_width:.2f} | Width ratio: {width_ratio:.2f}x (parallel)\n"
        f"Current price: ${current_price:.2f} | "
        f"Distance to wall: ${distance_to_breakout:.2f} ({dist_pct:.1f}%)\n"
        f"Volume: {vol_desc}\n"
        f"14-day ATR: ${atr:.2f} | Stop long: ${stop_long:.2f} | Stop short: ${stop_short:.2f}\n"
        f"Watch for: Bounce off lower channel (buy) or breakout above upper channel"
    )

    key_points: List[KeyPoint] = [
        (start_date, upper_at_start, "Channel Start High"),
        (end_date, upper_at_current, "Channel End High"),
        (start_date, lower_at_start, "Channel Start Low"),
        (end_date, lower_at_current, "Channel End Low"),
    ]

    return {
        "pattern": pattern_name,
        "confidence": round(confidence, 1),
        "key_points": key_points,
        "description": description,
        "state": state,
        "compression_ratio": round(compression_ratio, 4),
        "breakout_level_upper": round(upper_at_current, 4),
        "breakout_level_lower": round(lower_at_current, 4),
        "distance_to_breakout": round(distance_to_breakout, 4),
        "lines": [
            [(start_date, upper_at_start), (end_date, upper_at_current)],
            [(start_date, lower_at_start), (end_date, lower_at_current)],
        ],
        "upper_trendline": (resist_slope, resist_intercept),
        "lower_trendline": (support_slope, support_intercept),
    }


def _detect_flat_base(df: pd.DataFrame) -> Optional[PatternResult]:
    """Detect Flat Base / Rectangle — tight horizontal consolidation."""
    n = len(df)
    if n < 15:
        return None

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    pivot_highs, pivot_lows = _find_strict_pivots(df)
    if len(pivot_highs) < 2 or len(pivot_lows) < 2:
        return None

    ph_x = np.array(pivot_highs, dtype=float)
    ph_y = high[pivot_highs]
    pl_x = np.array(pivot_lows, dtype=float)
    pl_y = low[pivot_lows]

    resist_slope, resist_intercept = _linreg(ph_y, ph_x)
    support_slope, support_intercept = _linreg(pl_y, pl_x)

    # Both slopes near zero (flat)
    if abs(resist_slope) > 0.05 or abs(support_slope) > 0.05:
        return None

    end_idx = float(n - 1)
    start_idx = float(min(pivot_highs[0], pivot_lows[0]))

    upper_at_current = resist_slope * end_idx + resist_intercept
    lower_at_current = support_slope * end_idx + support_intercept
    current_width = abs(upper_at_current - lower_at_current)

    if current_width <= 0:
        return None

    current_price = float(close[-1])
    atr = _compute_atr(df)

    if current_width > 5 * atr:
        return None
    if current_price > upper_at_current or current_price < lower_at_current:
        return None

    upper_at_start = resist_slope * start_idx + resist_intercept
    lower_at_start = support_slope * start_idx + support_intercept

    distance_to_upper = abs(current_price - upper_at_current)
    distance_to_lower = abs(current_price - lower_at_current)
    distance_to_breakout = min(distance_to_upper, distance_to_lower)
    dist_pct = distance_to_breakout / current_price * 100 if current_price > 0 else 0

    state = (
        PatternState.NEAR_BREAK
        if distance_to_breakout <= 0.25 * atr
        else PatternState.FORMING
    )

    compression_ratio = min(current_width / (5 * atr) if atr > 0 else 0.5, 1.0)
    confidence = _compression_confidence(1.0 - compression_ratio, state)

    start_idx_int = int(start_idx)
    start_date = df.index[start_idx_int]
    end_date = df.index[-1]

    volume_arr = df["volume"].values if "volume" in df.columns else None
    vol_desc = "N/A"
    if volume_arr is not None:
        vol_desc = _vol_trend_desc(volume_arr, start_idx_int)

    range_pct = current_width / current_price * 100 if current_price > 0 else 0
    stop_long = round(lower_at_current - atr, 2)
    stop_short = round(upper_at_current + atr, 2)
    description = (
        f"NEUTRAL: Flat base / rectangle — tight horizontal consolidation\n"
        f"Resistance: ${upper_at_current:.2f} (flat) | Support: ${lower_at_current:.2f} (flat)\n"
        f"Range: ${current_width:.2f} ({range_pct:.1f}% of price) | ATR: ${atr:.2f}\n"
        f"Current price: ${current_price:.2f} | "
        f"Distance to breakout: ${distance_to_breakout:.2f} ({dist_pct:.1f}%)\n"
        f"Volume: {vol_desc}\n"
        f"14-day ATR: ${atr:.2f} | Stop long: ${stop_long:.2f} | Stop short: ${stop_short:.2f}\n"
        f"Watch for: Close outside range with volume surge to confirm direction"
    )

    key_points: List[KeyPoint] = [
        (start_date, upper_at_start, "Resistance"),
        (start_date, lower_at_start, "Support"),
        (end_date, current_price, "Current"),
    ]

    return {
        "pattern": "Flat Base",
        "confidence": round(confidence, 1),
        "key_points": key_points,
        "description": description,
        "state": state,
        "compression_ratio": round(compression_ratio, 4),
        "breakout_level_upper": round(upper_at_current, 4),
        "breakout_level_lower": round(lower_at_current, 4),
        "distance_to_breakout": round(distance_to_breakout, 4),
        "lines": [
            [(start_date, upper_at_start), (end_date, upper_at_current)],
            [(start_date, lower_at_start), (end_date, lower_at_current)],
        ],
        "upper_trendline": (resist_slope, resist_intercept),
        "lower_trendline": (support_slope, support_intercept),
    }


def _detect_broadening_formation(df: pd.DataFrame) -> Optional[PatternResult]:
    """Detect Broadening Formation / Megaphone — expanding volatility range."""
    n = len(df)
    if n < 15:
        return None

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    pivot_highs, pivot_lows = _find_strict_pivots(df)
    if len(pivot_highs) < 2 or len(pivot_lows) < 2:
        return None

    ph_x = np.array(pivot_highs, dtype=float)
    ph_y = high[pivot_highs]
    pl_x = np.array(pivot_lows, dtype=float)
    pl_y = low[pivot_lows]

    resist_slope, resist_intercept = _linreg(ph_y, ph_x)
    support_slope, support_intercept = _linreg(pl_y, pl_x)

    # Broadening: resistance rising, support falling
    if resist_slope <= 0 or support_slope >= 0:
        return None

    start_idx = float(min(pivot_highs[0], pivot_lows[0]))
    end_idx = float(n - 1)

    upper_at_start = resist_slope * start_idx + resist_intercept
    lower_at_start = support_slope * start_idx + support_intercept
    initial_width = abs(upper_at_start - lower_at_start)
    upper_at_current = resist_slope * end_idx + resist_intercept
    lower_at_current = support_slope * end_idx + support_intercept
    current_width = abs(upper_at_current - lower_at_current)

    if initial_width <= 0 or current_width <= initial_width:
        return None

    current_price = float(close[-1])
    if current_price > upper_at_current or current_price < lower_at_current:
        return None

    atr = _compute_atr(df)
    expansion_ratio = current_width / initial_width
    distance_to_upper = abs(current_price - upper_at_current)
    distance_to_lower = abs(current_price - lower_at_current)
    distance_to_breakout = min(distance_to_upper, distance_to_lower)
    dist_pct = distance_to_breakout / current_price * 100 if current_price > 0 else 0

    state = (
        PatternState.NEAR_BREAK
        if distance_to_breakout <= 0.25 * atr
        else PatternState.FORMING
    )

    compression_ratio = min(1.0 / expansion_ratio, 1.0)
    confidence = 62.0 if state == PatternState.NEAR_BREAK else 60.0

    start_idx_int = int(start_idx)
    start_date = df.index[start_idx_int]
    end_date = df.index[-1]

    volume_arr = df["volume"].values if "volume" in df.columns else None
    vol_desc = "N/A"
    if volume_arr is not None:
        vol_desc = _vol_trend_desc(volume_arr, start_idx_int)

    description = (
        f"CAUTION: Broadening formation / Megaphone — expanding volatility\n"
        f"Rising highs: ${upper_at_start:.2f} → ${upper_at_current:.2f} "
        f"(+{resist_slope:.3f}/bar) | "
        f"Falling lows: ${lower_at_start:.2f} → ${lower_at_current:.2f} "
        f"({support_slope:.3f}/bar)\n"
        f"Range expansion: {expansion_ratio:.2f}x "
        f"(from ${initial_width:.2f} to ${current_width:.2f})\n"
        f"Current price: ${current_price:.2f} | "
        f"Distance to boundary: ${distance_to_breakout:.2f} ({dist_pct:.1f}%)\n"
        f"Volume: {vol_desc}\n"
        f"14-day ATR: ${atr:.2f} | High-risk pattern — increased whipsaw probability\n"
        f"Watch for: Touch of upper/lower boundary as fade opportunity, or breakout continuation"
    )

    key_points: List[KeyPoint] = [
        (start_date, upper_at_start, "Upper Start"),
        (end_date, upper_at_current, "Upper End"),
        (start_date, lower_at_start, "Lower Start"),
        (end_date, lower_at_current, "Lower End"),
    ]

    return {
        "pattern": "Broadening Formation",
        "confidence": round(confidence, 1),
        "key_points": key_points,
        "description": description,
        "state": state,
        "compression_ratio": round(compression_ratio, 4),
        "breakout_level_upper": round(upper_at_current, 4),
        "breakout_level_lower": round(lower_at_current, 4),
        "distance_to_breakout": round(distance_to_breakout, 4),
        "lines": [
            [(start_date, upper_at_start), (end_date, upper_at_current)],
            [(start_date, lower_at_start), (end_date, lower_at_current)],
        ],
        "upper_trendline": (resist_slope, resist_intercept),
        "lower_trendline": (support_slope, support_intercept),
    }


# ---------------------------------------------------------------------------
# Registry of available patterns
# ---------------------------------------------------------------------------

_PATTERN_REGISTRY: Dict[str, Callable[["pd.DataFrame"], Optional[PatternResult]]] = {
    "bull flag": _detect_bull_flag,
    "bear flag": _detect_bear_flag,
    "ascending triangle": lambda df: _detect_triangle(df, ascending=True),
    "descending triangle": lambda df: _detect_triangle(df, ascending=False),
    "rising wedge": lambda df: _detect_wedge(df, rising=True),
    "falling wedge": lambda df: _detect_wedge(df, rising=False),
    "double bottom": _detect_double_bottom,
    "double top": _detect_double_top,
    "head and shoulders": _detect_head_and_shoulders,
    "inverse head and shoulders": _detect_inverse_head_and_shoulders,
    "cup and handle": _detect_cup_and_handle,
    "symmetrical triangle": _detect_symmetrical_triangle,
    "pennant": _detect_pennant,
    "channel up": lambda df: _detect_channel(df, up=True),
    "channel down": lambda df: _detect_channel(df, up=False),
    "flat base": _detect_flat_base,
    "broadening formation": _detect_broadening_formation,
}

# Register candlestick patterns
from momentum_radar.patterns.candlestick_detector import CANDLESTICK_PATTERNS  # noqa: E402

for _cs_name, _cs_func in CANDLESTICK_PATTERNS.items():
    _PATTERN_REGISTRY[_cs_name] = _cs_func


def available_patterns() -> List[str]:
    """Return the list of recognised pattern names."""
    return list(_PATTERN_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_pattern(
    pattern_name: str,
    df: pd.DataFrame,
) -> Optional[PatternResult]:
    """Detect a specific named pattern in OHLCV data.

    Uses the last :data:`PATTERN_LOOKBACK` rows of *df* for analysis.
    Only patterns in FORMING or NEAR_BREAK state are returned.

    Args:
        pattern_name: Case-insensitive pattern name (e.g. ``"bull flag"``).
        df:           Daily OHLCV DataFrame with ``open``, ``high``, ``low``,
                      ``close``, ``volume`` columns.

    Returns:
        A result dict with ``pattern``, ``confidence``, ``key_points``,
        ``description``, ``state``, ``compression_ratio``,
        ``breakout_level_upper``, ``breakout_level_lower``,
        ``distance_to_breakout``, and ``lines`` keys; or ``None`` if no
        active pattern is found.
    """
    key = pattern_name.strip().lower()
    detector = _PATTERN_REGISTRY.get(key)
    if detector is None:
        logger.warning("Unknown pattern %r. Available: %s", pattern_name, available_patterns())
        return None

    if df is None or df.empty or len(df) < 15:
        return None

    # Trim to lookback window
    data = df.tail(PATTERN_LOOKBACK).copy()
    try:
        result = detector(data)  # type: ignore[operator]
    except Exception as exc:
        logger.debug("Pattern detection error (%s): %s", pattern_name, exc)
        return None

    if result is None:
        return None
    if result.get("confidence", 0) < 60:
        return None
    # Never return BROKEN patterns
    state = result.get("state")
    if state == PatternState.BROKEN:
        return None
    return result


def scan_for_pattern(
    pattern_name: str,
    tickers: List[str],
    fetcher,
    top_n: int = 5,
) -> List[Dict]:
    """Scan multiple tickers for a pattern, return top *top_n* matches.

    Only patterns in FORMING or NEAR_BREAK state are included.

    Args:
        pattern_name: Pattern to search for.
        tickers:      List of ticker symbols to scan.
        fetcher:      A :class:`~momentum_radar.data.data_fetcher.BaseDataFetcher`
                      instance.
        top_n:        Maximum number of results to return.

    Returns:
        List of result dicts sorted by confidence (highest first), each
        extended with a ``"ticker"`` key.  Only FORMING / NEAR_BREAK patterns
        are included.
    """
    results: List[Dict] = []
    total = len(tickers)
    for idx, ticker in enumerate(tickers, 1):
        logger.info(
            "Scanning %s for %r (%d/%d)...", ticker, pattern_name, idx, total
        )
        try:
            df = fetcher.get_daily_bars(ticker, period=f"{PATTERN_LOOKBACK}d")
            if df is None or df.empty:
                continue
            match = detect_pattern(pattern_name, df)
            if match:
                state = match.get("state")
                # Filter: only FORMING or NEAR_BREAK (accept both enum and string)
                state_val = state.value if hasattr(state, "value") else state
                if state_val not in ("forming", "near_break"):
                    continue
                match["ticker"] = ticker
                match["df"] = df
                results.append(match)
        except Exception as exc:
            logger.debug("Error scanning %s: %s", ticker, exc)

    results.sort(key=lambda r: r.get("confidence", 0), reverse=True)
    return results[:top_n]
