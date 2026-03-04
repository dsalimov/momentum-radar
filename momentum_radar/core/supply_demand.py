"""
core/supply_demand.py – Supply & demand zone facade for strategy engines.

Thin wrapper around :mod:`momentum_radar.signals.supply_demand` that exposes
a simplified interface for querying active institutional zones.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

from momentum_radar.signals.supply_demand import SupplyDemandZone, detect_zones

logger = logging.getLogger(__name__)

# Default minimum zone score to include in results
_DEFAULT_MIN_SCORE: float = 60.0


def get_demand_zones(
    ticker: str,
    daily: Optional[pd.DataFrame],
    min_score: float = _DEFAULT_MIN_SCORE,
    timeframe: str = "daily",
) -> List[SupplyDemandZone]:
    """Return demand zones with strength above *min_score*.

    Args:
        ticker:    Stock symbol.
        daily:     Daily OHLCV DataFrame.
        min_score: Minimum zone strength score to include (0–100).
        timeframe: Timeframe label passed to the zone detector.

    Returns:
        List of :class:`~momentum_radar.signals.supply_demand.SupplyDemandZone`.
    """
    zones = detect_zones(ticker, daily, timeframe=timeframe)
    return [z for z in zones if z.zone_type == "demand" and z.strength_score >= min_score]


def get_supply_zones(
    ticker: str,
    daily: Optional[pd.DataFrame],
    min_score: float = _DEFAULT_MIN_SCORE,
    timeframe: str = "daily",
) -> List[SupplyDemandZone]:
    """Return supply zones with strength above *min_score*.

    Args:
        ticker:    Stock symbol.
        daily:     Daily OHLCV DataFrame.
        min_score: Minimum zone strength score to include (0–100).
        timeframe: Timeframe label passed to the zone detector.

    Returns:
        List of :class:`~momentum_radar.signals.supply_demand.SupplyDemandZone`.
    """
    zones = detect_zones(ticker, daily, timeframe=timeframe)
    return [z for z in zones if z.zone_type == "supply" and z.strength_score >= min_score]


def price_in_zone(
    price: float,
    zone: SupplyDemandZone,
    buffer_pct: float = 0.005,
) -> bool:
    """Return True if *price* lies within a zone's boundaries (± buffer).

    Args:
        price:      Current price.
        zone:       Supply or demand zone.
        buffer_pct: Fractional tolerance applied to zone boundaries.

    Returns:
        True when price is inside the zone.
    """
    buffer = zone.zone_high * buffer_pct
    return zone.zone_low - buffer <= price <= zone.zone_high + buffer


def nearest_zone(
    price: float,
    zones: List[SupplyDemandZone],
) -> Optional[SupplyDemandZone]:
    """Return the zone whose midpoint is closest to *price*.

    Args:
        price: Current price.
        zones: List of zones to search.

    Returns:
        The nearest :class:`SupplyDemandZone`, or ``None`` if *zones* is empty.
    """
    if not zones:
        return None
    return min(zones, key=lambda z: abs((z.zone_high + z.zone_low) / 2 - price))
