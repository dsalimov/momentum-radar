"""
signals/zone_manager.py – Active supply & demand zone registry.

Manages the lifecycle of :class:`~momentum_radar.signals.supply_demand.SupplyDemandZone`
objects in memory:

* Adding / updating zones detected from fresh price data
* Invalidating zones when price closes strongly through them
* Filtering zones below the minimum strength threshold
* Returning the best active zone for a ticker at a given price

Usage::

    from momentum_radar.signals.zone_manager import ZoneManager

    manager = ZoneManager()
    manager.update_zones("AAPL", detected_zones)
    active = manager.get_active_zone("AAPL", current_price=150.0, atr=2.5)
    if active:
        print(active.zone_type, active.strength_score)
"""

import logging
from typing import Dict, List, Optional

from momentum_radar.signals.supply_demand import SupplyDemandZone

logger = logging.getLogger(__name__)

# Default minimum zone strength to keep a zone in the registry
_DEFAULT_MIN_SCORE: float = 70.0
# Price must be within this ATR fraction of the zone boundary to be "near" it
_PROXIMITY_MULT: float = 0.50


class ZoneManager:
    """In-memory registry of active supply and demand zones per ticker.

    Attributes:
        min_score: Zones below this strength score are discarded on insert.
    """

    def __init__(self, min_score: float = _DEFAULT_MIN_SCORE) -> None:
        self.min_score = min_score
        # _zones: ticker → list of SupplyDemandZone (sorted by score desc)
        self._zones: Dict[str, List[SupplyDemandZone]] = {}

    # ------------------------------------------------------------------
    # Zone management
    # ------------------------------------------------------------------

    def update_zones(self, ticker: str, zones: List[SupplyDemandZone]) -> None:
        """Replace all stored zones for *ticker* with the freshly detected list.

        Zones below :attr:`min_score` or with ``status == "broken"`` are
        dropped automatically.

        Args:
            ticker: Stock symbol.
            zones:  Newly detected zones (from
                    :func:`~momentum_radar.signals.supply_demand.detect_zones`).
        """
        filtered = [
            z for z in zones
            if z.strength_score >= self.min_score and z.status != "broken"
        ]
        filtered.sort(key=lambda z: z.strength_score, reverse=True)
        self._zones[ticker] = filtered
        logger.debug(
            "ZoneManager: %s → %d zones stored (min_score=%.0f)",
            ticker, len(filtered), self.min_score,
        )

    def invalidate_zone(self, ticker: str, zone: SupplyDemandZone) -> None:
        """Mark *zone* as broken and remove it from the active registry.

        Args:
            ticker: Stock symbol.
            zone:   Zone to invalidate.
        """
        zone.status = "broken"
        if ticker in self._zones:
            self._zones[ticker] = [
                z for z in self._zones[ticker] if z is not zone
            ]
            logger.debug(
                "ZoneManager: invalidated %s %s zone [%.2f–%.2f]",
                ticker, zone.zone_type, zone.zone_low, zone.zone_high,
            )

    def invalidate_broken(self, ticker: str) -> int:
        """Remove all zones with ``status == "broken"`` for *ticker*.

        Args:
            ticker: Stock symbol.

        Returns:
            Number of zones removed.
        """
        before = len(self._zones.get(ticker, []))
        self._zones[ticker] = [
            z for z in self._zones.get(ticker, [])
            if z.status != "broken"
        ]
        removed = before - len(self._zones[ticker])
        if removed:
            logger.debug("ZoneManager: removed %d broken zone(s) for %s", removed, ticker)
        return removed

    def clear(self, ticker: str) -> None:
        """Remove all zones for *ticker* from the registry.

        Args:
            ticker: Stock symbol.
        """
        self._zones.pop(ticker, None)

    # ------------------------------------------------------------------
    # Zone queries
    # ------------------------------------------------------------------

    def get_zones(self, ticker: str) -> List[SupplyDemandZone]:
        """Return all active zones for *ticker*, sorted by strength desc.

        Args:
            ticker: Stock symbol.

        Returns:
            Sorted list of :class:`SupplyDemandZone` objects.
        """
        return list(self._zones.get(ticker, []))

    def get_active_zone(
        self,
        ticker: str,
        current_price: float,
        atr: float,
    ) -> Optional[SupplyDemandZone]:
        """Return the strongest zone that *current_price* is inside or near.

        "Near" is defined as within ``_PROXIMITY_MULT × ATR`` of the zone boundary.

        Args:
            ticker:        Stock symbol.
            current_price: Latest closing price.
            atr:           Current ATR (Average True Range).

        Returns:
            The highest-scored matching :class:`SupplyDemandZone`, or ``None``.
        """
        zones = self._zones.get(ticker, [])
        proximity_buffer = _PROXIMITY_MULT * atr if atr > 0 else 0.0

        for zone in zones:  # already sorted by score desc
            low_bound = zone.zone_low - proximity_buffer
            high_bound = zone.zone_high + proximity_buffer
            if low_bound <= current_price <= high_bound:
                logger.debug(
                    "ZoneManager: %s price %.2f near %s zone [%.2f–%.2f]",
                    ticker, current_price, zone.zone_type,
                    zone.zone_low, zone.zone_high,
                )
                return zone
        return None

    def ticker_count(self) -> int:
        """Return the number of tickers tracked by this manager."""
        return len(self._zones)

    def zone_count(self, ticker: Optional[str] = None) -> int:
        """Return the total number of active zones.

        Args:
            ticker: If given, count only zones for that ticker.

        Returns:
            Zone count.
        """
        if ticker is not None:
            return len(self._zones.get(ticker, []))
        return sum(len(v) for v in self._zones.values())
