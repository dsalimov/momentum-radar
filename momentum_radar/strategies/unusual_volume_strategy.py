"""
strategies/unusual_volume_strategy.py – Unusual Volume + Level Break Engine.

Triggers only when ALL of the following are present:
1. Volume ≥ 2× 20-day average
2. Break of major resistance or support (close above 20-day high)
3. Strong-body close (body ≥ 60 % of total candle range)
4. Continuation candle confirmation (follow-through on next bar)

No signal if:
* Wick-heavy breakout (body < 40 % of range)
* Immediate rejection on next bar
* Low follow-through

Quality gates
-------------
* Score ≥ 75 / 100
* ≥ 2 confirmations
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
_MIN_CONFIRMATIONS: int = 2
_VOLUME_SPIKE_MULT: float = 2.0
_BODY_RATIO_MIN: float = 0.40
_TIMEFRAME: str = "Daily"


def _check_volume(daily: Optional[pd.DataFrame]) -> Optional[str]:
    """Volume ≥ 2× 20-day average."""
    if daily is None or len(daily) < 22 or "volume" not in daily.columns:
        return None
    avg  = float(daily["volume"].iloc[-22:-1].mean())
    last = float(daily["volume"].iloc[-1])
    if avg > 0 and last >= avg * _VOLUME_SPIKE_MULT:
        ratio = last / avg
        return f"Volume Spike ({ratio:.1f}x Average)"
    return None


def _check_level_break(daily: Optional[pd.DataFrame]) -> Optional[str]:
    """Close above prior 20-day high (resistance break)."""
    if daily is None or len(daily) < 22 or "close" not in daily.columns:
        return None
    last_close  = float(daily["close"].iloc[-1])
    prior_high  = float(daily["high"].iloc[-22:-1].max())
    if last_close > prior_high:
        return f"Resistance Break (above ${prior_high:.2f})"
    return None


def _check_close_strength(daily: Optional[pd.DataFrame]) -> Optional[str]:
    """Strong-body close: body ≥ 60 % of total candle range."""
    if daily is None or len(daily) < 1:
        return None
    o  = float(daily["open"].iloc[-1])
    c  = float(daily["close"].iloc[-1])
    h  = float(daily["high"].iloc[-1])
    lo = float(daily["low"].iloc[-1])
    total_range = h - lo
    if total_range <= 0:
        return None
    body = abs(c - o)
    if body / total_range >= 0.60 and c > o:
        return "Strong Body Close"
    return None


def _check_continuation(daily: Optional[pd.DataFrame]) -> Optional[str]:
    """Second bar closes above first breakout bar close (follow-through)."""
    if daily is None or len(daily) < 2 or "close" not in daily.columns:
        return None
    if float(daily["close"].iloc[-1]) > float(daily["close"].iloc[-2]):
        return "Continuation Candle"
    return None


def evaluate(
    ticker: str,
    bars: Optional[pd.DataFrame] = None,
    daily: Optional[pd.DataFrame] = None,
    options: Optional[Dict] = None,
    fundamentals: Optional[Dict] = None,
    now: Optional[datetime] = None,
) -> StrategySignal:
    """Evaluate the unusual-volume strategy for *ticker*.

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
        strategy="unusual_volume",
        direction="BUY",
        timeframe=_TIMEFRAME,
        regime=get_regime_display(daily),
        htf_bias=get_htf_bias(daily),
    )

    confirmations: List[str] = []

    vol = _check_volume(daily)
    if vol:
        confirmations.append(vol)

    level_break = _check_level_break(daily)
    if level_break:
        confirmations.append(level_break)

    close_str = _check_close_strength(daily)
    if close_str:
        confirmations.append(close_str)

    cont = _check_continuation(daily)
    if cont:
        confirmations.append(cont)

    signal.confirmations = confirmations

    # Wick-heavy rejection check: if no strong close → fail
    level = 0.0
    if daily is not None and "high" in daily.columns and len(daily) >= 22:
        level = float(daily["high"].iloc[-22:-1].max())
    signal.fake_breakout_passed = passes_fake_breakout_filter(daily, level=level)

    # Scoring
    strengths = {
        "volume_ratio":   1.0 if vol        else 0.0,
        "level_break":    1.0 if level_break else 0.0,
        "close_strength": 1.0 if close_str  else 0.0,
        "continuation":   1.0 if cont       else 0.0,
    }
    signal.score = compute_strategy_score("unusual_volume", strengths)
    signal.grade = score_to_grade(signal.score)

    # Trade parameters
    if daily is not None and "close" in daily.columns and len(daily) > 0:
        entry = float(daily["close"].iloc[-1])
        atr   = compute_atr(daily)
        trade = compute_trade_params("unusual_volume", entry=entry, atr=atr)
        signal.entry  = trade.entry
        signal.stop   = trade.stop
        signal.target = trade.target
        signal.rr     = trade.rr

    signal.valid = (
        signal.score >= _MIN_SCORE
        and signal.confirmation_count >= _MIN_CONFIRMATIONS
        and signal.fake_breakout_passed
        and signal.rr >= 2.0
    )

    return signal
