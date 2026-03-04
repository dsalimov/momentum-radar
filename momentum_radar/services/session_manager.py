"""
services/session_manager.py – Trading session and timeframe management.

Maps the current time of day to the appropriate trading session and recommends
the primary analysis timeframe.

Session schedule (US Eastern time)
-----------------------------------
* **Pre-market**   04:00–09:30 → preparation only, no active signals
* **Open**         09:30–10:00 → 2-minute charts, momentum scalp
* **Morning**      10:00–11:00 → 5-minute charts, structure plays
* **Midday**       11:00–13:00 → 10-minute charts, avoid chop
* **Afternoon**    13:00–16:00 → 10-minute charts, trend continuation
* **After-hours**  16:00–20:00 → review only

Usage::

    from momentum_radar.services.session_manager import (
        get_current_session,
        get_session_timeframe,
        is_market_open,
    )

    session = get_current_session()
    tf = get_session_timeframe(session)
    print(session, tf)   # e.g. "morning" "5m"
"""

import logging
from datetime import datetime, time
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

# Session boundaries (US Eastern time, 24-hr)
_SESSION_BOUNDARIES = [
    (time(4, 0),  time(9, 30),  "premarket"),
    (time(9, 30), time(10, 0),  "open"),
    (time(10, 0), time(11, 0),  "morning"),
    (time(11, 0), time(13, 0),  "midday"),
    (time(13, 0), time(16, 0),  "afternoon"),
    (time(16, 0), time(20, 0),  "afterhours"),
]

# Recommended primary timeframe per session
_SESSION_TIMEFRAMES = {
    "premarket":  "15m",   # preparation / context only
    "open":       "2m",    # high-speed scalp window
    "morning":    "5m",    # structure + momentum plays
    "midday":     "10m",   # filter out chop
    "afternoon":  "10m",   # trend continuation
    "afterhours": "daily", # no active signals
    "closed":     "daily",
}

# Should signals fire in this session?
_SESSION_ACTIVE = {
    "premarket":  False,
    "open":       True,
    "morning":    True,
    "midday":     True,    # with extra filtering
    "afternoon":  True,
    "afterhours": False,
    "closed":     False,
}


def get_current_session(dt: Optional[datetime] = None) -> str:
    """Return the name of the current US Eastern trading session.

    Args:
        dt: Optional datetime to evaluate (defaults to now in ET).

    Returns:
        Session name: ``"premarket"``, ``"open"``, ``"morning"``,
        ``"midday"``, ``"afternoon"``, ``"afterhours"``, or ``"closed"``.
    """
    if dt is None:
        dt = datetime.now(_ET)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=_ET)
    else:
        dt = dt.astimezone(_ET)

    t = dt.time()
    # Weekends
    if dt.weekday() >= 5:
        return "closed"

    for start, end, name in _SESSION_BOUNDARIES:
        if start <= t < end:
            return name

    # Before 04:00 or after 20:00
    return "closed"


def get_session_timeframe(session: Optional[str] = None) -> str:
    """Return the recommended analysis timeframe for *session*.

    Args:
        session: Session name (from :func:`get_current_session`).
                 If ``None``, the current session is determined automatically.

    Returns:
        Timeframe string, e.g. ``"2m"``, ``"5m"``, ``"10m"``, ``"daily"``.
    """
    if session is None:
        session = get_current_session()
    return _SESSION_TIMEFRAMES.get(session, "5m")


def is_market_open(dt: Optional[datetime] = None) -> bool:
    """Return ``True`` when the US regular session (09:30–16:00 ET) is open.

    Args:
        dt: Optional datetime to evaluate.

    Returns:
        ``True`` during regular trading hours on weekdays.
    """
    if dt is None:
        dt = datetime.now(_ET)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=_ET)
    else:
        dt = dt.astimezone(_ET)

    if dt.weekday() >= 5:
        return False
    t = dt.time()
    return time(9, 30) <= t < time(16, 0)


def should_send_signals(session: Optional[str] = None) -> bool:
    """Return ``True`` when the current session supports live signal generation.

    Args:
        session: Session name.  If ``None``, detected automatically.

    Returns:
        ``True`` during ``"open"``, ``"morning"``, ``"midday"``, and
        ``"afternoon"`` sessions.
    """
    if session is None:
        session = get_current_session()
    return _SESSION_ACTIVE.get(session, False)


def get_session_info(dt: Optional[datetime] = None) -> dict:
    """Return a full session context dict.

    Args:
        dt: Optional datetime to evaluate.

    Returns:
        Dict with keys ``"session"``, ``"timeframe"``, ``"is_market_open"``,
        ``"signals_active"``.
    """
    session = get_current_session(dt)
    return {
        "session": session,
        "timeframe": get_session_timeframe(session),
        "is_market_open": is_market_open(dt),
        "signals_active": should_send_signals(session),
    }
