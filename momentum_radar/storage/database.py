"""
database.py – SQLite persistence layer using SQLAlchemy.

The ``alerts`` table stores every alert that reaches the minimum score
threshold.  A call to :func:`init_db` is required before any inserts.

Additional tables
-----------------
- ``alert_preferences`` – per-chat user preference for automated alerts on/off
- ``squeeze_alert_records`` – tracks last alert time + score per ticker for
  spam-filter logic (no same ticker within 6 h unless score rises by ≥ 10)
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, create_engine
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

class AlertPreference(Base):
    """Per-chat alert on/off preference."""

    __tablename__ = "alert_preferences"

    chat_id = Column(String(64), primary_key=True, nullable=False)
    alerts_enabled = Column(Boolean, nullable=False, default=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class SqueezeAlertRecord(Base):
    """Tracks the last time a squeeze alert was sent for a ticker.

    Used to prevent alert spam: the same ticker is suppressed for 6 hours
    unless its squeeze score increases by 10 or more.
    """

    __tablename__ = "squeeze_alert_records"

    ticker = Column(String(16), primary_key=True, nullable=False)
    last_alerted_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_score = Column(Integer, nullable=False, default=0)

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


# ---------------------------------------------------------------------------
# Alert preference helpers
# ---------------------------------------------------------------------------

def set_alert_preference(chat_id: str, enabled: bool) -> None:
    """Store or update the alert preference for *chat_id*.

    Args:
        chat_id: Telegram chat identifier.
        enabled: ``True`` to enable automated alerts, ``False`` to disable.
    """
    if _SessionLocal is None:
        logger.error("Database not initialised. Call init_db() first.")
        return
    try:
        with Session(_ENGINE) as session:
            pref = session.get(AlertPreference, chat_id)
            if pref is None:
                pref = AlertPreference(
                    chat_id=chat_id,
                    alerts_enabled=enabled,
                    updated_at=datetime.utcnow(),
                )
                session.add(pref)
            else:
                pref.alerts_enabled = enabled
                pref.updated_at = datetime.utcnow()
            session.commit()
            logger.debug("Alert preference set for %s: enabled=%s", chat_id, enabled)
    except Exception as exc:
        logger.error("Failed to set alert preference for %s: %s", chat_id, exc)


def get_alert_preference(chat_id: str) -> bool:
    """Return the alert preference for *chat_id* (default: ``True``).

    Args:
        chat_id: Telegram chat identifier.

    Returns:
        ``True`` if automated alerts are enabled, ``False`` otherwise.
    """
    if _SessionLocal is None:
        return True
    try:
        with Session(_ENGINE) as session:
            pref = session.get(AlertPreference, chat_id)
            return pref.alerts_enabled if pref is not None else True
    except Exception as exc:
        logger.error("Failed to read alert preference for %s: %s", chat_id, exc)
        return True


# ---------------------------------------------------------------------------
# Squeeze alert spam-filter helpers
# ---------------------------------------------------------------------------

_SQUEEZE_COOLDOWN_HOURS: int = 6
_SQUEEZE_SCORE_BUMP: int = 10


def should_send_squeeze_alert(ticker: str, current_score: int) -> bool:
    """Decide whether a squeeze alert for *ticker* should be sent.

    Rules:
    - Allow if the ticker has never been alerted before.
    - Allow if the last alert was more than 6 hours ago.
    - Allow if the score has increased by 10 or more since the last alert.
    - Suppress otherwise.

    Args:
        ticker:        Stock symbol.
        current_score: Current squeeze score (0–100).

    Returns:
        ``True`` if the alert should be sent.
    """
    if _SessionLocal is None:
        return True
    try:
        with Session(_ENGINE) as session:
            rec = session.get(SqueezeAlertRecord, ticker)
            if rec is None:
                return True
            elapsed = datetime.utcnow() - rec.last_alerted_at
            if elapsed >= timedelta(hours=_SQUEEZE_COOLDOWN_HOURS):
                return True
            if current_score >= rec.last_score + _SQUEEZE_SCORE_BUMP:
                return True
            return False
    except Exception as exc:
        logger.error("should_send_squeeze_alert error for %s: %s", ticker, exc)
        return True


def record_squeeze_alert(ticker: str, score: int) -> None:
    """Record that a squeeze alert was sent for *ticker* with *score*.

    Args:
        ticker: Stock symbol.
        score:  Squeeze score at time of alert.
    """
    if _SessionLocal is None:
        return
    try:
        with Session(_ENGINE) as session:
            rec = session.get(SqueezeAlertRecord, ticker)
            if rec is None:
                rec = SqueezeAlertRecord(
                    ticker=ticker,
                    last_alerted_at=datetime.utcnow(),
                    last_score=score,
                )
                session.add(rec)
            else:
                rec.last_alerted_at = datetime.utcnow()
                rec.last_score = score
            session.commit()
            logger.debug("Squeeze alert recorded for %s (score=%d)", ticker, score)
    except Exception as exc:
        logger.error("Failed to record squeeze alert for %s: %s", ticker, exc)