"""
market_hours.py – Market session and context checks.

Provides helpers to determine whether the scanner should be active and
whether the overall market context warrants a score penalty.
"""

import logging
from datetime import datetime, time
from typing import Optional, Tuple

import pytz

from momentum_radar.config import config

logger = logging.getLogger(__name__)

EST = pytz.timezone("America/New_York")


def _parse_time(time_str: str) -> time:
    """Parse an HH:MM string into a :class:`datetime.time` object.

    Args:
        time_str: Time string in ``"HH:MM"`` format.

    Returns:
        Corresponding :class:`datetime.time`.
    """
    h, m = time_str.split(":")
    return time(int(h), int(m))


def is_market_open(now: Optional[datetime] = None) -> bool:
    """Return ``True`` if the scanner should be running right now.

    The scanner is active between ``MARKET_OPEN`` and ``MARKET_CLOSE`` (EST).

    Args:
        now: Override the current time (useful for testing).  Must be
            timezone-aware or will be treated as EST.

    Returns:
        ``True`` if within the active scanning window.
    """
    cfg = config.market_hours
    if now is None:
        now = datetime.now(tz=EST)
    elif now.tzinfo is None:
        now = EST.localize(now)
    else:
        now = now.astimezone(EST)

    current_time = now.time()
    open_time = _parse_time(cfg.market_open)
    close_time = _parse_time(cfg.market_close)
    return open_time <= current_time <= close_time


def is_lunch_lull(now: Optional[datetime] = None) -> bool:
    """Return ``True`` if the current time falls in the lunch-lull window.

    During the lunch lull (default 12:00–13:30 EST) alert frequency is
    naturally reduced by the caller.

    Args:
        now: Override the current time.

    Returns:
        ``True`` if within the lunch-lull window.
    """
    cfg = config.market_hours
    if now is None:
        now = datetime.now(tz=EST)
    elif now.tzinfo is None:
        now = EST.localize(now)
    else:
        now = now.astimezone(EST)

    current_time = now.time()
    lunch_start = _parse_time(cfg.lunch_start)
    lunch_end = _parse_time(cfg.lunch_end)
    return lunch_start <= current_time <= lunch_end


def get_market_score_penalty(
    spy_pct_change: Optional[float],
    qqq_pct_change: Optional[float],
) -> Tuple[int, str]:
    """Determine a score penalty based on broad market context.

    If both SPY and QQQ are moving less than
    :attr:`~momentum_radar.config.MarketHoursConfig.flat_market_threshold`
    (default 0.3%), all signal scores are reduced by 1.

    Args:
        spy_pct_change: SPY percentage change (absolute decimal, e.g. 0.002).
        qqq_pct_change: QQQ percentage change (absolute decimal, e.g. 0.005).

    Returns:
        Tuple of ``(penalty: int, condition_description: str)``.
    """
    threshold = config.market_hours.flat_market_threshold
    if spy_pct_change is None or qqq_pct_change is None:
        return 0, "unknown"
    if abs(spy_pct_change) < threshold and abs(qqq_pct_change) < threshold:
        return 1, "flat_market"
    if spy_pct_change > 0 and qqq_pct_change > 0:
        return 0, "bullish"
    if spy_pct_change < 0 and qqq_pct_change < 0:
        return 0, "bearish"
    return 0, "mixed"
