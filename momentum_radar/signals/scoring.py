"""
scoring.py – Signal aggregation, scoring, and alert-level classification.

Each registered signal module is called in turn and its score is summed.  The
final total score is mapped to an :class:`AlertLevel` enum value.

Signal modules are registered via the module-level :func:`register_signal`
decorator so that the core scanner never needs to be modified when new modules
are added.
"""

import logging
from enum import Enum
from typing import Callable, Dict, List, Optional

import pandas as pd

from momentum_radar.config import config
from momentum_radar.signals.base import SignalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alert levels
# ---------------------------------------------------------------------------

class AlertLevel(Enum):
    """Alert priority levels based on total signal score."""

    IGNORE = "ignore"
    WATCHLIST = "watchlist"
    HIGH_PRIORITY = "high_priority"
    STRONG_MOMENTUM = "strong_momentum"


# ---------------------------------------------------------------------------
# Signal registry
# ---------------------------------------------------------------------------

SignalFn = Callable[..., SignalResult]
_SIGNAL_REGISTRY: Dict[str, SignalFn] = {}


def register_signal(name: str) -> Callable[[SignalFn], SignalFn]:
    """Class/function decorator that registers a signal callable.

    Usage::

        @register_signal("my_signal")
        def my_signal(bars, daily, fundamentals, **kwargs) -> SignalResult:
            ...

    Args:
        name: Unique name for the signal.

    Returns:
        The original function (unchanged).
    """
    def decorator(fn: SignalFn) -> SignalFn:
        _SIGNAL_REGISTRY[name] = fn
        logger.debug("Registered signal: %s", name)
        return fn
    return decorator


def get_registry() -> Dict[str, SignalFn]:
    """Return a copy of the current signal registry."""
    return dict(_SIGNAL_REGISTRY)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def score_to_alert_level(score: int) -> AlertLevel:
    """Map a numeric score to an :class:`AlertLevel`.

    Thresholds are read from :attr:`~momentum_radar.config.config.scores`.

    Args:
        score: Total signal score.

    Returns:
        Corresponding :class:`AlertLevel`.
    """
    thresholds = config.scores
    if score >= thresholds.strong_momentum:
        return AlertLevel.STRONG_MOMENTUM
    if score >= thresholds.high_priority:
        return AlertLevel.HIGH_PRIORITY
    if score >= thresholds.watchlist:
        return AlertLevel.WATCHLIST
    return AlertLevel.IGNORE


def compute_score(
    ticker: str,
    bars: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
    fundamentals: Optional[Dict],
    options: Optional[Dict],
    market_score_penalty: int = 0,
) -> Dict:
    """Run all registered signal modules and aggregate their scores.

    Args:
        ticker: Stock symbol.
        bars: Intraday OHLCV DataFrame.
        daily: Daily OHLCV DataFrame.
        fundamentals: Fundamental data dict (float, short interest, etc.).
        options: Options activity dict.
        market_score_penalty: Score reduction applied during flat market.

    Returns:
        Dict with keys ``score``, ``alert_level``, ``triggered_modules``,
        ``module_details``.
    """
    total_score = 0
    triggered: List[str] = []
    details: Dict[str, str] = {}

    for name, fn in _SIGNAL_REGISTRY.items():
        try:
            result: SignalResult = fn(
                ticker=ticker,
                bars=bars,
                daily=daily,
                fundamentals=fundamentals,
                options=options,
            )
            if result.triggered:
                total_score += result.score
                triggered.append(name)
                details[name] = result.details
                logger.debug(
                    "%s – signal '%s' fired (score +%d): %s",
                    ticker,
                    name,
                    result.score,
                    result.details,
                )
        except Exception as exc:
            logger.warning("Signal '%s' error for %s: %s", name, ticker, exc)

    total_score = max(0, total_score - market_score_penalty)
    level = score_to_alert_level(total_score)

    return {
        "score": total_score,
        "alert_level": level,
        "triggered_modules": triggered,
        "module_details": details,
    }
