"""
core/fake_breakout_filter.py – Fake breakout filter facade for strategy engines.

Delegates to :func:`momentum_radar.services.fake_breakout.is_fake_breakout`
and provides a simplified boolean helper for quick strategy-level filtering.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from momentum_radar.services.fake_breakout import is_fake_breakout

logger = logging.getLogger(__name__)


def check_fake_breakout(
    bars: Optional[pd.DataFrame],
    level: float,
    direction: str = "above",
) -> bool:
    """Return True if the breakout of *level* appears to be fake.

    Args:
        bars:      OHLCV DataFrame (intraday or daily).
        level:     Price level that was broken.
        direction: ``"above"`` for a resistance break, ``"below"`` for a support break.

    Returns:
        True if the breakout is flagged as fake (signal should be rejected).
    """
    if bars is None or bars.empty or level <= 0:
        return False
    result = is_fake_breakout(bars, level=level, direction=direction)
    return bool(result.get("is_fake", False))


def passes_fake_breakout_filter(
    bars: Optional[pd.DataFrame],
    level: float,
    direction: str = "above",
) -> bool:
    """Return True if the breakout passes the fake-breakout filter (i.e., is genuine).

    Convenience inverse of :func:`check_fake_breakout`.

    Args:
        bars:      OHLCV DataFrame.
        level:     Price level that was broken.
        direction: ``"above"`` or ``"below"``.

    Returns:
        True if the breakout appears genuine.
    """
    return not check_fake_breakout(bars, level=level, direction=direction)
