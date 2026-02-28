"""
detector.py – Chart pattern detection engine.

Uses OHLCV daily bar data with a 120-day lookback to identify common chart
patterns via local extrema analysis.  Each detector returns a result dict or
``None`` when no pattern is found.

Supported patterns
------------------
- Double Bottom / Double Top
- Head and Shoulders / Inverse Head and Shoulders
- Bull Flag / Bear Flag
- Cup and Handle
- Ascending Triangle / Descending Triangle
- Rising Wedge / Falling Wedge
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

KeyPoint = Tuple["pd.Timestamp", float, str]   # (date, price, label)
PatternResult = Dict                           # {"pattern", "confidence", "key_points", "description"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PATTERN_LOOKBACK = 120  # trading days


def _find_peaks_troughs(
    series: pd.Series,
    order: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays of peak and trough indices using scipy argrelextrema.

    The series is lightly smoothed (rolling mean window=3) before detection
    to reduce noise.

    Args:
        series: Price series (close prices).
        order:  Number of bars on each side to consider for a local extremum.

    Returns:
        Tuple of (peak_indices, trough_indices) as integer arrays.
    """
    try:
        from scipy.signal import argrelextrema
    except ImportError as exc:
        raise ImportError("scipy is required for pattern detection.") from exc

    smoothed = series.rolling(window=3, center=True, min_periods=1).mean().values
    peaks = argrelextrema(smoothed, np.greater, order=order)[0]
    troughs = argrelextrema(smoothed, np.less, order=order)[0]
    return peaks, troughs


def _linreg_slope(y: np.ndarray) -> float:
    """Return the slope of a linear regression through *y*."""
    x = np.arange(len(y), dtype=float)
    if len(x) < 2:
        return 0.0
    slope = float(np.polyfit(x, y, 1)[0])
    return slope


# ---------------------------------------------------------------------------
# Individual pattern detectors
# ---------------------------------------------------------------------------


def _detect_double_bottom(df: pd.DataFrame) -> Optional[PatternResult]:
    """Detect a Double Bottom (W-shape) pattern.

    Rules
    -----
    - Two troughs within ±2.5 % of each other
    - Separated by at least 10 bars
    - A peak between them at least 3 % above the troughs
    """
    close = df["close"]
    _, troughs = _find_peaks_troughs(close)
    peaks, _ = _find_peaks_troughs(close)

    if len(troughs) < 2:
        return None

    # Examine pairs of troughs (most recent first)
    for i in range(len(troughs) - 1, 0, -1):
        t2_idx = int(troughs[i])
        for j in range(i - 1, -1, -1):
            t1_idx = int(troughs[j])
            if (t2_idx - t1_idx) < 10:
                continue
            t1_price = float(close.iloc[t1_idx])
            t2_price = float(close.iloc[t2_idx])
            # Troughs within 2.5 %
            tol = 0.025
            if abs(t1_price - t2_price) / max(t1_price, t2_price) > tol:
                continue
            # Find a peak between the two troughs
            mid_peaks = peaks[(peaks > t1_idx) & (peaks < t2_idx)]
            if len(mid_peaks) == 0:
                continue
            mid_peak_idx = int(mid_peaks[np.argmax(close.iloc[mid_peaks].values)])
            mid_price = float(close.iloc[mid_peak_idx])
            avg_trough = (t1_price + t2_price) / 2
            if mid_price < avg_trough * 1.03:
                continue

            confidence = _score_double_bottom(t1_price, t2_price, mid_price)
            key_points: List[KeyPoint] = [
                (close.index[t1_idx], t1_price, "Trough 1"),
                (close.index[mid_peak_idx], mid_price, "Peak"),
                (close.index[t2_idx], t2_price, "Trough 2"),
            ]
            return {
                "pattern": "Double Bottom",
                "confidence": round(confidence, 1),
                "key_points": key_points,
                "description": (
                    f"W-shape with troughs at ~{avg_trough:.2f} "
                    f"and peak at {mid_price:.2f}"
                ),
            }
    return None


def _score_double_bottom(t1: float, t2: float, peak: float) -> float:
    avg = (t1 + t2) / 2
    symmetry = 1.0 - abs(t1 - t2) / avg
    height = (peak - avg) / avg
    return min(100.0, 60.0 + symmetry * 20.0 + min(height * 200, 20.0))


def _detect_double_top(df: pd.DataFrame) -> Optional[PatternResult]:
    """Detect a Double Top (M-shape) pattern."""
    close = df["close"]
    peaks, _ = _find_peaks_troughs(close)
    _, troughs = _find_peaks_troughs(close)

    if len(peaks) < 2:
        return None

    for i in range(len(peaks) - 1, 0, -1):
        p2_idx = int(peaks[i])
        for j in range(i - 1, -1, -1):
            p1_idx = int(peaks[j])
            if (p2_idx - p1_idx) < 10:
                continue
            p1_price = float(close.iloc[p1_idx])
            p2_price = float(close.iloc[p2_idx])
            tol = 0.025
            if abs(p1_price - p2_price) / max(p1_price, p2_price) > tol:
                continue
            # Find a trough between the two peaks
            mid_troughs = troughs[(troughs > p1_idx) & (troughs < p2_idx)]
            if len(mid_troughs) == 0:
                continue
            mid_trough_idx = int(mid_troughs[np.argmin(close.iloc[mid_troughs].values)])
            mid_price = float(close.iloc[mid_trough_idx])
            avg_peak = (p1_price + p2_price) / 2
            if mid_price > avg_peak * 0.97:
                continue

            symmetry = 1.0 - abs(p1_price - p2_price) / avg_peak
            depth = (avg_peak - mid_price) / avg_peak
            confidence = min(100.0, 60.0 + symmetry * 20.0 + min(depth * 200, 20.0))
            key_points: List[KeyPoint] = [
                (close.index[p1_idx], p1_price, "Peak 1"),
                (close.index[mid_trough_idx], mid_price, "Trough"),
                (close.index[p2_idx], p2_price, "Peak 2"),
            ]
            return {
                "pattern": "Double Top",
                "confidence": round(confidence, 1),
                "key_points": key_points,
                "description": (
                    f"M-shape with peaks at ~{avg_peak:.2f} "
                    f"and trough at {mid_price:.2f}"
                ),
            }
    return None


def _detect_head_and_shoulders(df: pd.DataFrame) -> Optional[PatternResult]:
    """Detect a Head and Shoulders (bearish) pattern.

    Rules
    -----
    - Three consecutive peaks: left shoulder, head (highest), right shoulder
    - Left and right shoulders within 3 % of each other
    - Head at least 3 % above the shoulders
    """
    close = df["close"]
    peaks, _ = _find_peaks_troughs(close)

    if len(peaks) < 3:
        return None

    for i in range(len(peaks) - 1, 1, -1):
        rs_idx = int(peaks[i])
        head_idx = int(peaks[i - 1])
        ls_idx = int(peaks[i - 2])

        ls_p = float(close.iloc[ls_idx])
        head_p = float(close.iloc[head_idx])
        rs_p = float(close.iloc[rs_idx])

        # Head must be highest
        if head_p <= ls_p or head_p <= rs_p:
            continue
        # Shoulders within 3 %
        if abs(ls_p - rs_p) / max(ls_p, rs_p) > 0.03:
            continue
        # Head at least 3 % above average shoulder
        avg_shoulder = (ls_p + rs_p) / 2
        if head_p < avg_shoulder * 1.03:
            continue

        symmetry = 1.0 - abs(ls_p - rs_p) / avg_shoulder
        height = (head_p - avg_shoulder) / avg_shoulder
        confidence = min(100.0, 60.0 + symmetry * 20.0 + min(height * 200, 20.0))
        key_points: List[KeyPoint] = [
            (close.index[ls_idx], ls_p, "Left Shoulder"),
            (close.index[head_idx], head_p, "Head"),
            (close.index[rs_idx], rs_p, "Right Shoulder"),
        ]
        return {
            "pattern": "Head and Shoulders",
            "confidence": round(confidence, 1),
            "key_points": key_points,
            "description": (
                f"Head at {head_p:.2f}, shoulders at ~{avg_shoulder:.2f}"
            ),
        }
    return None


def _detect_inverse_head_and_shoulders(df: pd.DataFrame) -> Optional[PatternResult]:
    """Detect an Inverse Head and Shoulders (bullish) pattern."""
    close = df["close"]
    _, troughs = _find_peaks_troughs(close)

    if len(troughs) < 3:
        return None

    for i in range(len(troughs) - 1, 1, -1):
        rs_idx = int(troughs[i])
        head_idx = int(troughs[i - 1])
        ls_idx = int(troughs[i - 2])

        ls_p = float(close.iloc[ls_idx])
        head_p = float(close.iloc[head_idx])
        rs_p = float(close.iloc[rs_idx])

        # Head must be the lowest
        if head_p >= ls_p or head_p >= rs_p:
            continue
        if abs(ls_p - rs_p) / max(ls_p, rs_p) > 0.03:
            continue
        avg_shoulder = (ls_p + rs_p) / 2
        if head_p > avg_shoulder * 0.97:
            continue

        symmetry = 1.0 - abs(ls_p - rs_p) / avg_shoulder
        depth = (avg_shoulder - head_p) / avg_shoulder
        confidence = min(100.0, 60.0 + symmetry * 20.0 + min(depth * 200, 20.0))
        key_points: List[KeyPoint] = [
            (close.index[ls_idx], ls_p, "Left Shoulder"),
            (close.index[head_idx], head_p, "Head"),
            (close.index[rs_idx], rs_p, "Right Shoulder"),
        ]
        return {
            "pattern": "Inverse Head and Shoulders",
            "confidence": round(confidence, 1),
            "key_points": key_points,
            "description": (
                f"Inverted head at {head_p:.2f}, shoulders at ~{avg_shoulder:.2f}"
            ),
        }
    return None


def _detect_bull_flag(df: pd.DataFrame) -> Optional[PatternResult]:
    """Detect a Bull Flag pattern.

    Rules
    -----
    - Pole: >10 % rise in 5–15 bars
    - Flag: subsequent 5–20 bars with slight downward drift (<5 %)
    """
    close = df["close"]
    n = len(close)
    if n < 25:
        return None

    for pole_end in range(n - 5, 14, -1):
        for pole_len in range(5, 16):
            pole_start = pole_end - pole_len
            if pole_start < 0:
                break
            pole_return = (
                float(close.iloc[pole_end]) - float(close.iloc[pole_start])
            ) / float(close.iloc[pole_start])
            if pole_return < 0.10:
                continue

            # Flag: look at bars after pole_end
            flag_end = min(pole_end + 20, n - 1)
            if flag_end <= pole_end + 4:
                continue
            flag_prices = close.iloc[pole_end: flag_end + 1].values
            flag_drift = (flag_prices[-1] - flag_prices[0]) / flag_prices[0]
            if not (-0.05 <= flag_drift <= 0.01):
                continue

            confidence = min(
                100.0,
                60.0 + min(pole_return * 100, 20.0) + min(abs(flag_drift) * 200, 20.0),
            )
            key_points: List[KeyPoint] = [
                (close.index[pole_start], float(close.iloc[pole_start]), "Pole Start"),
                (close.index[pole_end], float(close.iloc[pole_end]), "Pole Top"),
                (close.index[flag_end], float(close.iloc[flag_end]), "Flag End"),
            ]
            return {
                "pattern": "Bull Flag",
                "confidence": round(confidence, 1),
                "key_points": key_points,
                "description": (
                    f"Pole +{pole_return*100:.1f}% over {pole_len} bars, "
                    f"flag consolidation {flag_drift*100:.1f}%"
                ),
            }
    return None


def _detect_bear_flag(df: pd.DataFrame) -> Optional[PatternResult]:
    """Detect a Bear Flag pattern.

    Rules
    -----
    - Pole: >10 % drop in 5–15 bars
    - Flag: subsequent 5–20 bars with slight upward drift (<5 %)
    """
    close = df["close"]
    n = len(close)
    if n < 25:
        return None

    for pole_end in range(n - 5, 14, -1):
        for pole_len in range(5, 16):
            pole_start = pole_end - pole_len
            if pole_start < 0:
                break
            pole_return = (
                float(close.iloc[pole_end]) - float(close.iloc[pole_start])
            ) / float(close.iloc[pole_start])
            if pole_return > -0.10:
                continue

            flag_end = min(pole_end + 20, n - 1)
            if flag_end <= pole_end + 4:
                continue
            flag_prices = close.iloc[pole_end: flag_end + 1].values
            flag_drift = (flag_prices[-1] - flag_prices[0]) / flag_prices[0]
            if not (-0.01 <= flag_drift <= 0.05):
                continue

            confidence = min(
                100.0,
                60.0 + min(abs(pole_return) * 100, 20.0) + min(flag_drift * 200, 20.0),
            )
            key_points: List[KeyPoint] = [
                (close.index[pole_start], float(close.iloc[pole_start]), "Pole Start"),
                (close.index[pole_end], float(close.iloc[pole_end]), "Pole Bottom"),
                (close.index[flag_end], float(close.iloc[flag_end]), "Flag End"),
            ]
            return {
                "pattern": "Bear Flag",
                "confidence": round(confidence, 1),
                "key_points": key_points,
                "description": (
                    f"Pole {pole_return*100:.1f}% over {pole_len} bars, "
                    f"flag drift +{flag_drift*100:.1f}%"
                ),
            }
    return None


def _detect_cup_and_handle(df: pd.DataFrame) -> Optional[PatternResult]:
    """Detect a Cup and Handle pattern.

    Rules
    -----
    - U-shaped recovery: price declines then recovers to near the prior high
    - Handle: small pullback of 3–8 % after the cup
    """
    close = df["close"]
    n = len(close)
    if n < 40:
        return None

    # Look in the last 80 bars (or all available)
    window = min(n, 80)
    sub = close.iloc[n - window:]
    peaks_idx, troughs_idx = _find_peaks_troughs(sub)

    if len(peaks_idx) < 2 or len(troughs_idx) < 1:
        return None

    # Cup: first peak → trough → second peak
    for i in range(len(peaks_idx) - 1, 0, -1):
        p2_idx = int(peaks_idx[i])
        p1_candidates = peaks_idx[peaks_idx < p2_idx]
        if len(p1_candidates) == 0:
            continue
        p1_idx = int(p1_candidates[-1])
        # Minimum cup width
        if (p2_idx - p1_idx) < 15:
            continue
        p1_p = float(sub.iloc[p1_idx])
        p2_p = float(sub.iloc[p2_idx])
        # Cup rim should be at similar levels
        if abs(p1_p - p2_p) / max(p1_p, p2_p) > 0.05:
            continue
        # Trough inside the cup
        cup_troughs = troughs_idx[(troughs_idx > p1_idx) & (troughs_idx < p2_idx)]
        if len(cup_troughs) == 0:
            continue
        cup_bottom_idx = int(cup_troughs[np.argmin(sub.iloc[cup_troughs].values)])
        cup_bottom_p = float(sub.iloc[cup_bottom_idx])
        cup_depth = (max(p1_p, p2_p) - cup_bottom_p) / max(p1_p, p2_p)
        if cup_depth < 0.10 or cup_depth > 0.50:
            continue

        # Handle: small pullback after the cup
        handle_end = min(p2_idx + 15, len(sub) - 1)
        if handle_end <= p2_idx + 2:
            continue
        handle_low = float(sub.iloc[p2_idx: handle_end + 1].min())
        handle_drift = (handle_low - p2_p) / p2_p
        if not (-0.08 <= handle_drift <= -0.03):
            continue

        depth_score = min(cup_depth * 200, 20.0)
        symmetry = 1.0 - abs(p1_p - p2_p) / max(p1_p, p2_p)
        confidence = min(100.0, 60.0 + symmetry * 10.0 + depth_score)
        # Map back to original df index
        offset = n - window
        key_points: List[KeyPoint] = [
            (close.index[offset + p1_idx], p1_p, "Cup Left"),
            (close.index[offset + cup_bottom_idx], cup_bottom_p, "Cup Bottom"),
            (close.index[offset + p2_idx], p2_p, "Cup Right"),
            (close.index[offset + handle_end], float(sub.iloc[handle_end]), "Handle"),
        ]
        return {
            "pattern": "Cup and Handle",
            "confidence": round(confidence, 1),
            "key_points": key_points,
            "description": (
                f"Cup depth {cup_depth*100:.1f}%, handle {handle_drift*100:.1f}%"
            ),
        }
    return None


def _detect_triangle(df: pd.DataFrame, ascending: bool) -> Optional[PatternResult]:
    """Detect Ascending or Descending Triangle patterns.

    Ascending triangle: flat resistance, rising support (higher lows).
    Descending triangle: flat support, falling resistance (lower highs).
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    n = len(close)
    if n < 20:
        return None

    window = min(n, 60)
    sub_high = high.iloc[n - window:].values
    sub_low = low.iloc[n - window:].values
    x = np.arange(window, dtype=float)

    resist_slope = _linreg_slope(sub_high)
    support_slope = _linreg_slope(sub_low)

    pattern_name = "Ascending Triangle" if ascending else "Descending Triangle"

    if ascending:
        # Flat resistance (near-zero slope), rising support (positive slope)
        if not (-0.05 < resist_slope < 0.05) or support_slope <= 0:
            return None
        convergence = support_slope / (abs(resist_slope) + 1e-6)
        if convergence < 0.5:
            return None
    else:
        # Flat support (near-zero slope), falling resistance (negative slope)
        if not (-0.05 < support_slope < 0.05) or resist_slope >= 0:
            return None
        convergence = abs(resist_slope) / (abs(support_slope) + 1e-6)
        if convergence < 0.5:
            return None

    confidence = min(100.0, 60.0 + min(abs(convergence) * 10.0, 40.0))
    offset = n - window
    resist_line_start = float(np.polyval(np.polyfit(x, sub_high, 1), 0))
    resist_line_end = float(np.polyval(np.polyfit(x, sub_high, 1), window - 1))
    support_line_start = float(np.polyval(np.polyfit(x, sub_low, 1), 0))
    support_line_end = float(np.polyval(np.polyfit(x, sub_low, 1), window - 1))

    key_points: List[KeyPoint] = [
        (close.index[offset], resist_line_start, "Resistance Start"),
        (close.index[-1], resist_line_end, "Resistance End"),
        (close.index[offset], support_line_start, "Support Start"),
        (close.index[-1], support_line_end, "Support End"),
    ]
    desc = (
        f"Resistance slope: {resist_slope:.4f}, "
        f"Support slope: {support_slope:.4f}"
    )
    return {
        "pattern": pattern_name,
        "confidence": round(confidence, 1),
        "key_points": key_points,
        "description": desc,
    }


def _detect_wedge(df: pd.DataFrame, rising: bool) -> Optional[PatternResult]:
    """Detect Rising or Falling Wedge patterns.

    Rising wedge: both trendlines slope up and converge (bearish).
    Falling wedge: both trendlines slope down and converge (bullish).
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    n = len(close)
    if n < 20:
        return None

    window = min(n, 60)
    sub_high = high.iloc[n - window:].values
    sub_low = low.iloc[n - window:].values
    x = np.arange(window, dtype=float)

    resist_slope = _linreg_slope(sub_high)
    support_slope = _linreg_slope(sub_low)

    pattern_name = "Rising Wedge" if rising else "Falling Wedge"

    if rising:
        # Both slopes positive, resistance slope < support slope (converging)
        if resist_slope <= 0 or support_slope <= 0:
            return None
        if support_slope <= resist_slope:
            return None
    else:
        # Both slopes negative, support slope < resistance slope (converging)
        if resist_slope >= 0 or support_slope >= 0:
            return None
        if resist_slope <= support_slope:
            return None

    spread_start = abs(
        float(np.polyval(np.polyfit(x, sub_high, 1), 0))
        - float(np.polyval(np.polyfit(x, sub_low, 1), 0))
    )
    spread_end = abs(
        float(np.polyval(np.polyfit(x, sub_high, 1), window - 1))
        - float(np.polyval(np.polyfit(x, sub_low, 1), window - 1))
    )
    if spread_end >= spread_start or spread_start == 0:
        return None

    convergence = 1.0 - spread_end / spread_start
    confidence = min(100.0, 60.0 + convergence * 40.0)

    offset = n - window
    resist_start_p = float(np.polyval(np.polyfit(x, sub_high, 1), 0))
    resist_end_p = float(np.polyval(np.polyfit(x, sub_high, 1), window - 1))
    support_start_p = float(np.polyval(np.polyfit(x, sub_low, 1), 0))
    support_end_p = float(np.polyval(np.polyfit(x, sub_low, 1), window - 1))

    key_points: List[KeyPoint] = [
        (close.index[offset], resist_start_p, "Resistance Start"),
        (close.index[-1], resist_end_p, "Resistance End"),
        (close.index[offset], support_start_p, "Support Start"),
        (close.index[-1], support_end_p, "Support End"),
    ]
    return {
        "pattern": pattern_name,
        "confidence": round(confidence, 1),
        "key_points": key_points,
        "description": (
            f"Convergence {convergence*100:.1f}%, "
            f"slopes R:{resist_slope:.4f} S:{support_slope:.4f}"
        ),
    }


# ---------------------------------------------------------------------------
# Registry of available patterns
# ---------------------------------------------------------------------------

_PATTERN_REGISTRY: Dict[str, Callable[["pd.DataFrame"], Optional[PatternResult]]] = {
    "double bottom": _detect_double_bottom,
    "double top": _detect_double_top,
    "head and shoulders": _detect_head_and_shoulders,
    "inverse head and shoulders": _detect_inverse_head_and_shoulders,
    "bull flag": _detect_bull_flag,
    "bear flag": _detect_bear_flag,
    "cup and handle": _detect_cup_and_handle,
    "ascending triangle": lambda df: _detect_triangle(df, ascending=True),
    "descending triangle": lambda df: _detect_triangle(df, ascending=False),
    "rising wedge": lambda df: _detect_wedge(df, rising=True),
    "falling wedge": lambda df: _detect_wedge(df, rising=False),
}


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

    Args:
        pattern_name: Case-insensitive pattern name (e.g. ``"double bottom"``).
        df:           Daily OHLCV DataFrame with ``open``, ``high``, ``low``,
                      ``close``, ``volume`` columns.

    Returns:
        A result dict with ``pattern``, ``confidence``, ``key_points``, and
        ``description`` keys, or ``None`` if the pattern is not found or the
        confidence is below 60 %.
    """
    key = pattern_name.strip().lower()
    detector = _PATTERN_REGISTRY.get(key)
    if detector is None:
        logger.warning("Unknown pattern '%s'. Available: %s", pattern_name, available_patterns())
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
    return result


def scan_for_pattern(
    pattern_name: str,
    tickers: List[str],
    fetcher,
    top_n: int = 5,
) -> List[Dict]:
    """Scan multiple tickers for a pattern, return top *top_n* matches.

    Args:
        pattern_name: Pattern to search for.
        tickers:      List of ticker symbols to scan.
        fetcher:      A :class:`~momentum_radar.data.data_fetcher.BaseDataFetcher`
                      instance.
        top_n:        Maximum number of results to return.

    Returns:
        List of result dicts sorted by confidence (highest first), each
        extended with a ``"ticker"`` key.
    """
    results: List[Dict] = []
    total = len(tickers)
    for idx, ticker in enumerate(tickers, 1):
        logger.info(
            "Scanning %s for '%s' (%d/%d)…", ticker, pattern_name, idx, total
        )
        try:
            df = fetcher.get_daily_bars(ticker, period=f"{PATTERN_LOOKBACK}d")
            if df is None or df.empty:
                continue
            match = detect_pattern(pattern_name, df)
            if match:
                match["ticker"] = ticker
                match["df"] = df
                results.append(match)
        except Exception as exc:
            logger.debug("Error scanning %s: %s", ticker, exc)

    results.sort(key=lambda r: r.get("confidence", 0), reverse=True)
    return results[:top_n]
