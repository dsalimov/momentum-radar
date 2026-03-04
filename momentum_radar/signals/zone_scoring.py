"""
signals/zone_scoring.py – Institutional Supply & Demand zone strength scoring.

Scoring formula (0–100):

+---------------------------------------+--------+
| Component                             | Points |
+=======================================+========+
| Impulse strength  (0–30)              |  0–30  |
| Volume expansion  (0–20)              |  0–20  |
| Clean structure break (0–20)          |  0–20  |
| Time spent in base  (0–10)            |  0–10  |
| Higher-timeframe alignment (0–20)     |  0–20  |
+---------------------------------------+--------+

Zones with ``strength_score < ZONE_MIN_SCORE`` are discarded.

Usage::

    from momentum_radar.signals.zone_scoring import score_zone, ZONE_MIN_SCORE

    score = score_zone(
        impulse_ratio=2.5,
        volume_ratio=1.8,
        base_candle_count=4,
        bos_confirmed=True,
        timeframe="daily",
    )
    if score >= ZONE_MIN_SCORE:
        # keep the zone
        ...
"""

from typing import Dict

# Minimum score for a zone to be considered institutional-grade
ZONE_MIN_SCORE: float = 70.0

# Timeframe alignment bonus (0-20 pts)
_TF_ALIGNMENT: Dict[str, float] = {
    "weekly":  20.0,
    "daily":   16.0,
    "4h":      10.0,
    "1h":       6.0,
    "15m":      3.0,
    "10m":      2.0,
    "5m":       1.0,
    "2m":       0.0,
    "1m":       0.0,
}


def score_zone(
    impulse_ratio: float,
    volume_ratio: float,
    base_candle_count: int,
    bos_confirmed: bool,
    timeframe: str,
) -> float:
    """Compute an institutional zone strength score (0–100).

    Args:
        impulse_ratio:      Impulse size in ATR multiples (e.g. 2.5 means 2.5× ATR).
        volume_ratio:       Volume ratio vs. average during impulse (e.g. 1.8 means 1.8× avg).
        base_candle_count:  Number of candles that formed the base (2–6 recommended).
        bos_confirmed:      Whether a Break-of-Structure was confirmed after the impulse.
        timeframe:          Timeframe label (e.g. ``"daily"``, ``"5m"``).

    Returns:
        Zone strength score capped at 100.0.
    """
    # Impulse strength: saturates at 3× ATR → 30 pts
    impulse_pts = min(impulse_ratio / 3.0 * 30.0, 30.0)
    impulse_pts = max(impulse_pts, 0.0)

    # Volume expansion: 1× avg = 0 pts, 3× avg = 20 pts
    vol_pts = min((volume_ratio - 1.0) / 2.0 * 20.0, 20.0)
    vol_pts = max(vol_pts, 0.0)

    # Clean structure break: binary 20 pts
    bos_pts = 20.0 if bos_confirmed else 0.0

    # Time spent in base: 2–6 candles ideal → 10 pts, <2 or >6 → fewer points
    if 2 <= base_candle_count <= 6:
        base_pts = 10.0
    elif base_candle_count == 1:
        base_pts = 4.0
    elif base_candle_count <= 8:
        base_pts = 6.0
    else:
        base_pts = 2.0

    # Higher-timeframe alignment bonus
    tf_pts = _TF_ALIGNMENT.get(timeframe.lower(), 0.0)

    total = impulse_pts + vol_pts + bos_pts + base_pts + tf_pts
    return min(round(total, 1), 100.0)


def is_displacement(
    candle_range: float,
    body_ratio: float,
    candle_volume: float,
    avg_range: float,
    avg_volume: float,
    range_mult: float = 1.5,
    body_threshold: float = 0.70,
    volume_mult: float = 1.5,
) -> bool:
    """Return ``True`` if a candle qualifies as a displacement / impulse candle.

    A displacement requires:

    * Range > ``range_mult`` × average range
    * Body ratio > ``body_threshold`` (strong body close)
    * Volume > ``volume_mult`` × average volume

    Args:
        candle_range:    High – Low of the candidate candle.
        body_ratio:      |close – open| / (high – low), range 0–1.
        candle_volume:   Volume of the candidate candle.
        avg_range:       Average candle range over the lookback window.
        avg_volume:      Average volume over the lookback window.
        range_mult:      Minimum range multiple of average (default 1.5).
        body_threshold:  Minimum body/range ratio (default 0.70).
        volume_mult:     Minimum volume multiple of average (default 1.5).

    Returns:
        ``True`` when all three criteria are met.
    """
    if avg_range <= 0 or avg_volume <= 0:
        return False
    return (
        candle_range > range_mult * avg_range
        and body_ratio > body_threshold
        and candle_volume > volume_mult * avg_volume
    )


def is_base(candle_ranges: list, avg_range: float, max_range_mult: float = 1.2) -> bool:
    """Return ``True`` if a sequence of candles forms a valid base / consolidation.

    A base must have:
    * At least 2 and at most 6 candles
    * All individual candle ranges below ``max_range_mult`` × ``avg_range``

    Args:
        candle_ranges:  List of high-low ranges for base candles.
        avg_range:      Average candle range over a longer lookback window.
        max_range_mult: Maximum allowed range multiple (default 1.2).

    Returns:
        ``True`` when the sequence qualifies as a tight base.
    """
    if not candle_ranges or avg_range <= 0:
        return False
    if not (2 <= len(candle_ranges) <= 6):
        return False
    return max(candle_ranges) < max_range_mult * avg_range
