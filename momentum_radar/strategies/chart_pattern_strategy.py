"""
strategies/chart_pattern_strategy.py – Chart Pattern Engine.

Detects
-------
* Bull / Bear Flags
* Ascending / Descending / Symmetrical Triangles
* Double Top / Double Bottom
* Head & Shoulders / Inverse Head & Shoulders
* Cup & Handle

Requirements
------------
* Clear structural pattern (delegated to :mod:`momentum_radar.patterns.detector`)
* Volume contraction during formation
* Volume expansion on breakout
* Break + follow-through
* Fake-breakout filter passed
* Score ≥ 75

Quality gates
-------------
* Score ≥ 75 / 100
* ≥ 3 confirmations
* Fake-breakout filter passed
* R:R ≥ 2.0
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from momentum_radar.core.fake_breakout_filter import passes_fake_breakout_filter
from momentum_radar.core.regime_engine import get_htf_bias, get_regime_display
from momentum_radar.core.risk_engine import compute_trade_params
from momentum_radar.core.scoring_engine import compute_strategy_score, score_to_grade
from momentum_radar.strategies.base import StrategySignal
from momentum_radar.utils.indicators import compute_atr

logger = logging.getLogger(__name__)

_MIN_SCORE: int = 75
_MIN_CONFIRMATIONS: int = 3
_TIMEFRAME: str = "Daily"


def _detect_pattern(daily: Optional[pd.DataFrame]) -> Optional[Dict]:
    """Attempt pattern detection via the existing detector module.

    Returns the raw pattern result dict on success, or ``None`` if no
    pattern is detected or the module is unavailable.
    """
    if daily is None or len(daily) < 20:
        return None
    try:
        from momentum_radar.patterns.detector import detect_pattern

        return detect_pattern(daily)
    except Exception as exc:
        logger.debug("Pattern detector error: %s", exc)
        return None


def _check_volume_contraction(daily: Optional[pd.DataFrame], lookback: int = 10) -> Optional[str]:
    """Volume trend declining over the last *lookback* bars (formation phase)."""
    if daily is None or len(daily) < lookback + 2 or "volume" not in daily.columns:
        return None
    formation_vol  = float(daily["volume"].iloc[-(lookback + 1):-1].mean())
    prior_avg_vol  = float(daily["volume"].iloc[-(lookback + 11):-(lookback + 1)].mean())
    if prior_avg_vol > 0 and formation_vol < prior_avg_vol * 0.85:
        return "Volume Contraction During Formation"
    return None


def _check_volume_expansion(daily: Optional[pd.DataFrame]) -> Optional[str]:
    """Last bar volume higher than the prior 10-bar average (breakout confirmation)."""
    if daily is None or len(daily) < 12 or "volume" not in daily.columns:
        return None
    avg  = float(daily["volume"].iloc[-12:-1].mean())
    last = float(daily["volume"].iloc[-1])
    if avg > 0 and last >= avg * 1.3:
        return f"Volume Expansion on Breakout ({last / avg:.1f}x)"
    return None


def _check_follow_through(daily: Optional[pd.DataFrame]) -> Optional[str]:
    """Second consecutive close above the breakout level."""
    if daily is None or len(daily) < 3 or "close" not in daily.columns:
        return None
    closes = daily["close"]
    if float(closes.iloc[-1]) > float(closes.iloc[-2]) > float(closes.iloc[-3]):
        return "Follow-Through Confirmed"
    return None


def evaluate(
    ticker: str,
    bars: Optional[pd.DataFrame] = None,
    daily: Optional[pd.DataFrame] = None,
    options: Optional[Dict] = None,
    fundamentals: Optional[Dict] = None,
    now: Optional[datetime] = None,
) -> StrategySignal:
    """Evaluate the chart pattern strategy for *ticker*.

    Args:
        ticker:       Stock symbol.
        bars:         Intraday bars (unused by this strategy).
        daily:        Daily OHLCV DataFrame.
        options:      Options activity dict (unused).
        fundamentals: Fundamental data dict (unused).
        now:          Override current datetime (unused).

    Returns:
        :class:`~momentum_radar.strategies.base.StrategySignal`.
    """
    signal = StrategySignal(
        ticker=ticker,
        strategy="chart_pattern",
        direction="BUY",
        timeframe=_TIMEFRAME,
        regime=get_regime_display(daily),
        htf_bias=get_htf_bias(daily),
    )

    confirmations: List[str] = []

    # --- Pattern detection ---
    pattern_result = _detect_pattern(daily)
    pattern_conf_str: Optional[str] = None
    if pattern_result:
        pname = pattern_result.get("pattern", "")
        pconf = float(pattern_result.get("confidence", 0.0))
        if pname and pconf >= 70.0:
            pattern_conf_str = f"{pname} (conf {pconf:.0f}%)"
            confirmations.append(pattern_conf_str)

    vol_contraction = _check_volume_contraction(daily)
    if vol_contraction:
        confirmations.append(vol_contraction)

    vol_expansion = _check_volume_expansion(daily)
    if vol_expansion:
        confirmations.append(vol_expansion)

    follow_through = _check_follow_through(daily)
    if follow_through:
        confirmations.append(follow_through)

    signal.confirmations = confirmations

    # Fake breakout filter
    level = 0.0
    if daily is not None and "high" in daily.columns and len(daily) >= 22:
        level = float(daily["high"].iloc[-22:-1].max())
    signal.fake_breakout_passed = passes_fake_breakout_filter(daily, level=level)

    # Scoring
    strengths = {
        "pattern_clarity":    1.0 if pattern_conf_str else 0.0,
        "volume_contraction": 1.0 if vol_contraction  else 0.0,
        "volume_expansion":   1.0 if vol_expansion    else 0.0,
        "follow_through":     1.0 if follow_through   else 0.0,
        "fake_breakout":      1.0 if signal.fake_breakout_passed else 0.0,
    }
    signal.score = compute_strategy_score("chart_pattern", strengths)
    signal.grade = score_to_grade(signal.score)

    # Trade parameters
    if daily is not None and "close" in daily.columns and len(daily) > 0:
        entry = float(daily["close"].iloc[-1])
        atr   = compute_atr(daily)
        trade = compute_trade_params("chart_pattern", entry=entry, atr=atr)
        signal.entry   = trade.entry
        signal.stop    = trade.stop
        signal.target  = trade.target
        signal.target2 = trade.target2
        signal.rr      = trade.rr

    signal.valid = (
        signal.score >= _MIN_SCORE
        and signal.confirmation_count >= _MIN_CONFIRMATIONS
        and signal.fake_breakout_passed
        and signal.rr >= 2.0
    )

    return signal
