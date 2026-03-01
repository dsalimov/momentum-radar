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

    return {
        "pattern": pattern_name,
        "confidence": round(confidence, 1),
        "key_points": key_points,
        "description": (
            f"Compression: {compression_ratio * 100:.0f}%, State: {state.value}"
        ),
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

    return {
        "pattern": pattern_name,
        "confidence": round(confidence, 1),
        "key_points": key_points,
        "description": (
            f"Compression: {compression_ratio * 100:.0f}%, State: {state.value}"
        ),
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

            return {
                "pattern": "Bull Flag",
                "confidence": round(confidence, 1),
                "key_points": key_points,
                "description": (
                    f"Pole +{pole_return * 100:.1f}% over {pole_len} bars, "
                    f"State: {state.value}"
                ),
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

            return {
                "pattern": "Bear Flag",
                "confidence": round(confidence, 1),
                "key_points": key_points,
                "description": (
                    f"Pole {pole_return * 100:.1f}% over {pole_len} bars, "
                    f"State: {state.value}"
                ),
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
# Registry of available patterns
# ---------------------------------------------------------------------------

_PATTERN_REGISTRY: Dict[str, Callable[["pd.DataFrame"], Optional[PatternResult]]] = {
    "bull flag": _detect_bull_flag,
    "bear flag": _detect_bear_flag,
    "ascending triangle": lambda df: _detect_triangle(df, ascending=True),
    "descending triangle": lambda df: _detect_triangle(df, ascending=False),
    "rising wedge": lambda df: _detect_wedge(df, rising=True),
    "falling wedge": lambda df: _detect_wedge(df, rising=False),
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
