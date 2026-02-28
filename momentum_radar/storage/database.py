"""
database.py – SQLite persistence layer using SQLAlchemy.

The ``alerts`` table stores every alert that reaches the minimum score
threshold.  A call to :func:`init_db` is required before any inserts.
"""

import logging
from datetime import datetime
from typing import List, Optional

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

_ENGINE = None
_SessionLocal = None


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""


class Alert(Base):
    """ORM model for the ``alerts`` table."""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    ticker = Column(String(16), nullable=False)
    price = Column(Float, nullable=True)
    score = Column(Integer, nullable=False)
    alert_level = Column(String(32), nullable=False)
    modules_triggered = Column(Text, nullable=True)  # comma-separated
    rvol = Column(Float, nullable=True)
    atr_ratio = Column(Float, nullable=True)
    short_interest = Column(Float, nullable=True)
    float_shares = Column(Float, nullable=True)
    market_condition = Column(String(32), nullable=True)
    pct_change = Column(Float, nullable=True)


def init_db(db_url: str = "sqlite:///momentum_radar.db") -> None:
    """Initialise the database engine and create tables.

    This function is idempotent and safe to call multiple times.

    Args:
        db_url: SQLAlchemy database URL.
    """
    global _ENGINE, _SessionLocal
    _ENGINE = create_engine(db_url, echo=False)
    Base.metadata.create_all(_ENGINE)
    _SessionLocal = sessionmaker(bind=_ENGINE)
    logger.info("Database initialised: %s", db_url)


def save_alert(
    ticker: str,
    price: Optional[float],
    score: int,
    alert_level: str,
    modules_triggered: List[str],
    rvol: Optional[float] = None,
    atr_ratio: Optional[float] = None,
    short_interest: Optional[float] = None,
    float_shares: Optional[float] = None,
    market_condition: Optional[str] = None,
    pct_change: Optional[float] = None,
    timestamp: Optional[datetime] = None,
) -> None:
    """Persist an alert record to the database.

    Args:
        ticker: Stock symbol.
        price: Current price at time of alert.
        score: Total signal score.
        alert_level: Alert level string (e.g. ``"high_priority"``).
        modules_triggered: List of signal names that fired.
        rvol: Relative volume.
        atr_ratio: Day range / ATR ratio.
        short_interest: Short interest as decimal.
        float_shares: Float share count.
        market_condition: Descriptive market context string.
        pct_change: Percentage price change today.
        timestamp: Override the default ``datetime.utcnow()``.
    """
    if _SessionLocal is None:
        logger.error("Database not initialised. Call init_db() first.")
        return

    record = Alert(
        timestamp=timestamp or datetime.utcnow(),
        ticker=ticker,
        price=price,
        score=score,
        alert_level=alert_level,
        modules_triggered=",".join(modules_triggered),
        rvol=rvol,
        atr_ratio=atr_ratio,
        short_interest=short_interest,
        float_shares=float_shares,
        market_condition=market_condition,
        pct_change=pct_change,
    )
    try:
        with Session(_ENGINE) as session:
            session.add(record)
            session.commit()
            logger.debug("Alert saved to DB: %s (score=%d)", ticker, score)
    except Exception as exc:
        logger.error("Failed to save alert for %s: %s", ticker, exc)
