"""
storage/zone_store.py – Persistence layer for supply & demand zones.

Provides CRUD helpers for the ``supply_demand_zones`` table so that
zone state (touch count, status, strength score) survives across scan
cycles.

Usage::

    from momentum_radar.storage.zone_store import upsert_zone, load_zones

    # After detecting zones for AAPL:
    for zone in detected_zones:
        upsert_zone(zone)

    # Before scanning AAPL to get last-known zones:
    cached = load_zones("AAPL", timeframe="daily")
"""

import logging
from datetime import datetime
from typing import List, Optional

from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.orm import Session

from momentum_radar.signals.supply_demand import SupplyDemandZone
from momentum_radar.storage.database import Base, _ENGINE, _SessionLocal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------

class SupplyDemandZoneRecord(Base):
    """ORM model for the ``supply_demand_zones`` table."""

    __tablename__ = "supply_demand_zones"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(16), nullable=False, index=True)
    timeframe = Column(String(8), nullable=False)
    zone_type = Column(String(8), nullable=False)    # "demand" | "supply"
    zone_high = Column(Float, nullable=False)
    zone_low = Column(Float, nullable=False)
    strength_score = Column(Float, nullable=False, default=0.0)
    touch_count = Column(Integer, nullable=False, default=0)
    status = Column(String(16), nullable=False, default="fresh")
    impulse_magnitude = Column(Float, nullable=True)
    volume_expansion = Column(Float, nullable=True)
    creation_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_test_timestamp = Column(DateTime, nullable=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_table() -> bool:
    """Create the ``supply_demand_zones`` table if it does not exist.

    Returns:
        ``True`` if the engine is available, ``False`` otherwise.
    """
    if _ENGINE is None:
        logger.debug("Database engine not initialised; skipping zone table creation.")
        return False
    try:
        SupplyDemandZoneRecord.__table__.create(_ENGINE, checkfirst=True)
    except Exception as exc:
        logger.debug("Zone table creation skipped: %s", exc)
    return True


def upsert_zone(zone: SupplyDemandZone) -> None:
    """Insert or update a zone record.

    Matching is performed on (ticker, timeframe, zone_type, zone_low, zone_high)
    rounded to 4 decimal places.  If a matching record exists its
    ``touch_count``, ``status``, ``strength_score``, and
    ``last_test_timestamp`` are refreshed; otherwise a new row is inserted.

    Args:
        zone: :class:`~momentum_radar.signals.supply_demand.SupplyDemandZone`
              to persist.
    """
    if not _ensure_table() or _SessionLocal is None:
        return

    try:
        with Session(_ENGINE) as session:
            # Find an existing record with matching boundaries (±0.5 %)
            records = (
                session.query(SupplyDemandZoneRecord)
                .filter(
                    SupplyDemandZoneRecord.ticker == zone.ticker,
                    SupplyDemandZoneRecord.timeframe == zone.timeframe,
                    SupplyDemandZoneRecord.zone_type == zone.zone_type,
                )
                .all()
            )
            matched: Optional[SupplyDemandZoneRecord] = None
            for rec in records:
                if (
                    abs(rec.zone_low - zone.zone_low) / (zone.zone_low or 1) < 0.005
                    and abs(rec.zone_high - zone.zone_high) / (zone.zone_high or 1) < 0.005
                ):
                    matched = rec
                    break

            if matched is None:
                rec = SupplyDemandZoneRecord(
                    ticker=zone.ticker,
                    timeframe=zone.timeframe,
                    zone_type=zone.zone_type,
                    zone_high=round(zone.zone_high, 4),
                    zone_low=round(zone.zone_low, 4),
                    strength_score=zone.strength_score,
                    touch_count=zone.touch_count,
                    status=zone.status,
                    impulse_magnitude=zone.impulse_magnitude,
                    volume_expansion=zone.volume_expansion,
                    creation_timestamp=datetime.utcnow(),
                    last_test_timestamp=datetime.utcnow() if zone.touch_count > 0 else None,
                )
                session.add(rec)
            else:
                matched.strength_score = zone.strength_score
                matched.touch_count = zone.touch_count
                matched.status = zone.status
                if zone.touch_count > 0:
                    matched.last_test_timestamp = datetime.utcnow()

            session.commit()
            logger.debug(
                "Zone upserted: %s %s %s [%.2f–%.2f] score=%.0f",
                zone.ticker,
                zone.timeframe,
                zone.zone_type,
                zone.zone_low,
                zone.zone_high,
                zone.strength_score,
            )
    except Exception as exc:
        logger.error("Failed to upsert zone for %s: %s", zone.ticker, exc)


def load_zones(
    ticker: str,
    timeframe: Optional[str] = None,
    min_score: float = 50.0,
    exclude_broken: bool = True,
) -> List[SupplyDemandZone]:
    """Load persisted zones for *ticker* from the database.

    Args:
        ticker:         Stock symbol.
        timeframe:      Optional timeframe filter (e.g. ``"daily"``).
        min_score:      Minimum strength score to return.
        exclude_broken: Skip zones marked as ``"broken"``.

    Returns:
        List of :class:`~momentum_radar.signals.supply_demand.SupplyDemandZone`
        objects, sorted by strength score descending.
    """
    if not _ensure_table() or _SessionLocal is None:
        return []

    try:
        with Session(_ENGINE) as session:
            query = session.query(SupplyDemandZoneRecord).filter(
                SupplyDemandZoneRecord.ticker == ticker,
                SupplyDemandZoneRecord.strength_score >= min_score,
            )
            if timeframe:
                query = query.filter(SupplyDemandZoneRecord.timeframe == timeframe)
            if exclude_broken:
                query = query.filter(SupplyDemandZoneRecord.status != "broken")

            records = query.order_by(SupplyDemandZoneRecord.strength_score.desc()).all()
            return [
                SupplyDemandZone(
                    ticker=rec.ticker,
                    timeframe=rec.timeframe,
                    zone_type=rec.zone_type,
                    zone_high=rec.zone_high,
                    zone_low=rec.zone_low,
                    strength_score=rec.strength_score,
                    touch_count=rec.touch_count,
                    status=rec.status,
                    impulse_magnitude=rec.impulse_magnitude or 0.0,
                    volume_expansion=rec.volume_expansion or 0.0,
                )
                for rec in records
            ]
    except Exception as exc:
        logger.error("Failed to load zones for %s: %s", ticker, exc)
        return []
