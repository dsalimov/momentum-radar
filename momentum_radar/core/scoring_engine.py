"""
core/scoring_engine.py – Per-strategy scoring models.

Each strategy has its own scoring model with different weights reflecting
that strategy's confirmation hierarchy.

All models return an integer score in the range 0–100.

Strategy weight maps
--------------------
- **Scalp**          – prioritises momentum speed and volume burst
- **Intraday**       – weights structure quality and S&D zone alignment
- **Swing**          – prioritises HTF zones and displacement magnitude
- **Chart Pattern**  – weights pattern clarity and volume behaviour
- **Unusual Volume** – focuses exclusively on volume ratio and level quality
"""

from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-strategy weight maps (values sum to 100)
# ---------------------------------------------------------------------------

_SCALP_WEIGHTS: Dict[str, int] = {
    "momentum":        30,  # RSI/MACD alignment
    "volume_spike":    25,  # Intraday volume burst
    "structure_break": 20,  # Break of prior swing high/low
    "htf_bias":        15,  # Daily-trend alignment
    "fake_breakout":   10,  # Fake-breakout filter passed
}

_INTRADAY_WEIGHTS: Dict[str, int] = {
    "trend_alignment": 25,  # Trend aligned on multiple timeframes
    "supply_demand":   25,  # S&D zone retest quality
    "volume_confirm":  20,  # Volume expansion on entry bar
    "structure":       20,  # Clean price structure in direction
    "fake_breakout":   10,  # Fake-breakout filter passed
}

_SWING_WEIGHTS: Dict[str, int] = {
    "htf_zone":        30,  # Higher-timeframe S&D zone
    "displacement":    25,  # Impulse magnitude (ATR multiple)
    "structure_break": 20,  # Major structure break confirmation
    "liquidity_sweep": 15,  # Stop hunt / liquidity sweep
    "volume_confirm":  10,  # Volume on the breakout bar
}

_CHART_PATTERN_WEIGHTS: Dict[str, int] = {
    "pattern_clarity":    30,  # Structural clarity of the pattern
    "volume_contraction": 20,  # Volume contracts during formation
    "volume_expansion":   20,  # Volume expands on breakout
    "follow_through":     20,  # Continuation candle after break
    "fake_breakout":      10,  # Fake-breakout filter passed
}

_UNUSUAL_VOLUME_WEIGHTS: Dict[str, int] = {
    "volume_ratio":   40,  # Current volume vs. 20-day average
    "level_break":    30,  # Resistance / support break quality
    "close_strength": 20,  # Strong-body close (not wick-heavy)
    "continuation":   10,  # Continuation candle follows
}

_STRATEGY_WEIGHTS: Dict[str, Dict[str, int]] = {
    "scalp":           _SCALP_WEIGHTS,
    "intraday":        _INTRADAY_WEIGHTS,
    "swing":           _SWING_WEIGHTS,
    "chart_pattern":   _CHART_PATTERN_WEIGHTS,
    "unusual_volume":  _UNUSUAL_VOLUME_WEIGHTS,
}


def compute_strategy_score(
    strategy: str,
    confirmations: Dict[str, float],
) -> int:
    """Compute a weighted strategy score from confirmation strengths.

    Args:
        strategy:      Strategy key – one of ``"scalp"``, ``"intraday"``,
                       ``"swing"``, ``"chart_pattern"``, ``"unusual_volume"``.
        confirmations: Mapping of confirmation key → normalised strength (0.0–1.0).
                       Missing keys are treated as 0.0.

    Returns:
        Integer score in the range 0–100.
    """
    weights = _STRATEGY_WEIGHTS.get(strategy)
    if not weights:
        logger.warning("Unknown strategy: %s – returning 0", strategy)
        return 0

    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0

    earned = 0.0
    for key, max_pts in weights.items():
        strength = float(confirmations.get(key, 0.0))
        strength = max(0.0, min(1.0, strength))   # clamp to [0, 1]
        earned += max_pts * strength

    score = int(round(earned * 100.0 / total_weight))
    return min(100, max(0, score))


def score_to_grade(score: int) -> str:
    """Map a 0–100 strategy score to a letter grade.

    Args:
        score: Strategy score (0–100).

    Returns:
        ``"A+"`` for score ≥ 90, ``"A"`` for ≥ 80, ``"B"`` for ≥ 70, ``"C"`` otherwise.
    """
    if score >= 90:
        return "A+"
    if score >= 80:
        return "A"
    if score >= 70:
        return "B"
    return "C"


def get_strategy_weights(strategy: str) -> Dict[str, int]:
    """Return a copy of the weight map for the given strategy.

    Args:
        strategy: Strategy key.

    Returns:
        Dict of confirmation key → maximum points.
    """
    return dict(_STRATEGY_WEIGHTS.get(strategy, {}))
