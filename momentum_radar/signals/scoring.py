"""
scoring.py – Signal aggregation, scoring, and alert-level classification.

Each registered signal module is called in turn and its score is summed.  The
final total score is mapped to an :class:`AlertLevel` enum value.

A **weighted 0–100+ score** is also computed using :data:`_MODULE_WEIGHTS`.
An alert is only dispatched when *both* of these gates are passed:

1. ``weighted_score >= config.scores.signal_score_minimum``  (default 75)
2. ``confirmation_count >= config.scores.min_signal_confirmations``  (default 2)

A **chop filter** suppresses signals when the current day's high-low range is
below ``config.signals.chop_range_multiplier × ATR``, preventing signals in
low-volatility, ranging markets.

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
from momentum_radar.utils.indicators import compute_atr

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
# Module weight map (used for the 0-100+ weighted score)
# ---------------------------------------------------------------------------

# Each weight is the maximum point contribution when a module fires at full
# strength (raw score = 2).  A module scoring 1 contributes half the weight.
# weighted_contribution = weight × (raw_score / 2.0)
#
# Tier 1 – primary confirmations (trend + momentum + volume)
# Tier 2 – structural confirmations (price structure, options, S&D)
# Tier 3 – supplementary / context signals
_MODULE_WEIGHTS: Dict[str, int] = {
    "ema_trend":             30,  # Full EMA bullish alignment (price > EMA9 > EMA21 > EMA200)
    "rsi_macd":              25,  # RSI in momentum zone + positive MACD histogram
    "volume_spike":          25,  # Intraday volume spike vs. recent average
    "relative_volume":       20,  # RVOL vs. 30-day average
    "structure_break":       20,  # Break of prev-day high/low or opening range
    "options_flow":          20,  # Unusual call/put activity (smart-money signal)
    "supply_demand_zone":    15,  # Price near an institutional supply/demand zone
    "vwap_proximity":        15,  # Price above VWAP (intraday momentum)
    "third_touch_support":   12,  # Third touch of a key horizontal support level
    "volatility_expansion":  12,  # Day range expansion relative to ATR
    "failed_breakout":       10,  # Liquidity trap / bull-trap / bear-trap
    "volatility_squeeze":     8,  # Bollinger Band squeeze expansion
    "short_interest":         5,  # High short-interest float (squeeze fuel)
}

# Default weight assigned to any module not listed above
_DEFAULT_MODULE_WEIGHT: int = 5


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


def _compute_weighted_score(module_scores: Dict[str, int]) -> int:
    """Compute a weighted score from per-module raw scores.

    Uses :data:`_MODULE_WEIGHTS` to apply importance weights to each module's
    raw score (0–2).  The contribution of each module is::

        contribution = weight × (raw_score / 2.0)

    Only *triggered* modules (raw score > 0) contribute to the weighted total.

    Args:
        module_scores: Mapping of module name → raw score (0–2).

    Returns:
        Integer weighted score.
    """
    total = 0.0
    for name, raw in module_scores.items():
        if raw > 0:
            weight = _MODULE_WEIGHTS.get(name, _DEFAULT_MODULE_WEIGHT)
            total += weight * (raw / 2.0)
    return int(total)


def _is_choppy_market(
    bars: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
) -> bool:
    """Return ``True`` if the market appears to be in a low-volatility chop range.

    Detection: the current day's high-low range is compared to the 14-day ATR.
    If ``range / ATR < config.signals.chop_range_multiplier``, the market lacks
    directional momentum and signals should be suppressed.

    Args:
        bars:  Intraday 1-min OHLCV DataFrame (used for day range when available).
        daily: Daily OHLCV DataFrame (used for ATR calculation).

    Returns:
        ``True`` if the market is considered choppy/ranging.
    """
    if daily is None or daily.empty:
        return False

    atr = compute_atr(daily, period=14)
    if atr is None or atr <= 0:
        return False

    # Derive current day range
    if bars is not None and not bars.empty and "high" in bars.columns and "low" in bars.columns:
        day_range = float(bars["high"].max()) - float(bars["low"].min())
    elif "high" in daily.columns and "low" in daily.columns:
        day_range = float(daily["high"].iloc[-1]) - float(daily["low"].iloc[-1])
    else:
        return False

    if day_range <= 0:
        return False

    ratio = day_range / atr
    threshold = config.signals.chop_range_multiplier
    if ratio < threshold:
        logger.debug("Chop filter: day range/ATR = %.2f < %.2f -> suppressing signals", ratio, threshold)
        return True
    return False


def compute_score(
    ticker: str,
    bars: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
    fundamentals: Optional[Dict],
    options: Optional[Dict],
    market_score_penalty: int = 0,
) -> Dict:
    """Run all registered signal modules and aggregate their scores.

    Two independent gates must be passed before an alert is dispatched:

    1. ``weighted_score >= config.scores.signal_score_minimum`` (default 75)
    2. ``confirmation_count >= config.scores.min_signal_confirmations`` (default 2)

    When the market is in a choppy/ranging state (detected via the volatility
    filter) the weighted score is set to 0 so both gates fail automatically.

    Args:
        ticker: Stock symbol.
        bars: Intraday OHLCV DataFrame.
        daily: Daily OHLCV DataFrame.
        fundamentals: Fundamental data dict (float, short interest, etc.).
        options: Options activity dict.
        market_score_penalty: Score reduction applied during flat market.

    Returns:
        Dict with keys:

        - ``score``              – raw integer score (backward-compatible)
        - ``alert_level``        – :class:`AlertLevel`
        - ``triggered_modules``  – list of module names that fired
        - ``module_details``     – module name → detail string
        - ``weighted_score``     – weighted 0-100+ score for the quality gate
        - ``confirmation_count`` – number of modules that fired
        - ``module_scores``      – module name → raw score (all modules)
        - ``chop_suppressed``    – True when signals were suppressed by the chop filter
    """
    total_score = 0
    triggered: List[str] = []
    details: Dict[str, str] = {}
    module_scores: Dict[str, int] = {}

    for name, fn in _SIGNAL_REGISTRY.items():
        try:
            result: SignalResult = fn(
                ticker=ticker,
                bars=bars,
                daily=daily,
                fundamentals=fundamentals,
                options=options,
            )
            module_scores[name] = result.score
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

    # Chop filter: suppress signals in low-volatility, ranging markets
    chop_suppressed = _is_choppy_market(bars, daily)

    # Weighted quality score
    weighted_score = 0 if chop_suppressed else _compute_weighted_score(module_scores)
    confirmation_count = len(triggered)

    if chop_suppressed:
        logger.info(
            "%s – chop filter active: weighted_score suppressed (raw=%d)",
            ticker,
            _compute_weighted_score(module_scores),
        )
    else:
        logger.debug(
            "%s – weighted_score=%d, confirmations=%d/%d",
            ticker,
            weighted_score,
            confirmation_count,
            config.scores.min_signal_confirmations,
        )

    return {
        "score": total_score,
        "alert_level": level,
        "triggered_modules": triggered,
        "module_details": details,
        "weighted_score": weighted_score,
        "confirmation_count": confirmation_count,
        "module_scores": module_scores,
        "chop_suppressed": chop_suppressed,
    }
