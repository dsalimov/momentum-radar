"""
strategies/intraday_strategy.py – Intraday Engine.

Timeframe
---------
* 5m  (morning session)
* 10m (midday / afternoon)

Entry requirements
------------------
1. Trend alignment   – price above key EMAs on multiple timeframes
2. S&D zone retest   – price entering a scored supply or demand zone
3. Volume confirmation – volume above average on entry bar
4. Clean structure   – no messy chop (BOS confirmed on daily)
5. Fake-breakout filter passed

Quality gates
-------------
* Score ≥ 75 / 100
* ≥ 3 confirmations
* Fake-breakout filter passed
* R:R ≥ 2.5
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from momentum_radar.core.fake_breakout_filter import passes_fake_breakout_filter
from momentum_radar.core.regime_engine import get_htf_bias, get_regime_display
from momentum_radar.core.risk_engine import compute_trade_params
from momentum_radar.core.scoring_engine import compute_strategy_score, score_to_grade
from momentum_radar.core.structure_engine import detect_structure_break
from momentum_radar.core.supply_demand import get_demand_zones, price_in_zone
from momentum_radar.services.session_manager import get_current_session
from momentum_radar.strategies.base import StrategySignal
from momentum_radar.utils.indicators import compute_atr

logger = logging.getLogger(__name__)

_MIN_SCORE: int = 75
_MIN_CONFIRMATIONS: int = 3

_SESSION_TIMEFRAME: Dict[str, str] = {
    "open":      "5m",
    "morning":   "5m",
    "midday":    "10m",
    "afternoon": "10m",
}


def _check_trend(daily: Optional[pd.DataFrame]) -> Optional[str]:
    """Price above EMA21 and EMA50 (multi-timeframe trend)."""
    if daily is None or len(daily) < 50 or "close" not in daily.columns:
        return None
    closes = daily["close"]
    last   = float(closes.iloc[-1])
    ema21  = float(closes.ewm(span=21, adjust=False).mean().iloc[-1])
    ema50  = float(closes.ewm(span=50, adjust=False).mean().iloc[-1])
    if last > ema21 and ema21 > ema50:
        return "HTF Bias Bullish"
    return None


def _check_sd_zone(
    ticker: str,
    bars: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
) -> Optional[str]:
    """Price entering a scored demand zone."""
    if bars is None or daily is None or "close" not in bars.columns:
        return None
    price = float(bars["close"].iloc[-1])
    zones = get_demand_zones(ticker, daily, min_score=60.0)
    for zone in zones:
        if price_in_zone(price, zone):
            return f"Demand Zone Retest (score {zone.strength_score:.0f})"
    return None


def _check_volume(bars: Optional[pd.DataFrame]) -> Optional[str]:
    """Volume above 20-bar average on entry bar."""
    if bars is None or len(bars) < 21 or "volume" not in bars.columns:
        return None
    avg  = float(bars["volume"].iloc[-21:-1].mean())
    last = float(bars["volume"].iloc[-1])
    if avg > 0 and last >= avg * 1.2:
        return f"Volume Expansion ({last / avg:.1f}x)"
    return None


def _check_structure(daily: Optional[pd.DataFrame]) -> Optional[str]:
    """Bullish BOS on daily bars."""
    result = detect_structure_break(daily)
    if result.confirmed and result.direction == "bullish":
        return "Break of Structure"
    return None


def evaluate(
    ticker: str,
    bars: Optional[pd.DataFrame] = None,
    daily: Optional[pd.DataFrame] = None,
    options: Optional[Dict] = None,
    fundamentals: Optional[Dict] = None,
    now: Optional[datetime] = None,
) -> StrategySignal:
    """Evaluate the intraday strategy for *ticker*.

    Args:
        ticker:       Stock symbol.
        bars:         Intraday OHLCV DataFrame (5m or 10m bars).
        daily:        Daily OHLCV DataFrame.
        options:      Options activity dict (unused).
        fundamentals: Fundamental data dict (unused).
        now:          Override current datetime (for testing).

    Returns:
        :class:`~momentum_radar.strategies.base.StrategySignal`.
    """
    session   = get_current_session(now)
    timeframe = _SESSION_TIMEFRAME.get(session, "5m")

    signal = StrategySignal(
        ticker=ticker,
        strategy="intraday",
        direction="BUY",
        timeframe=timeframe,
        regime=get_regime_display(daily),
        htf_bias=get_htf_bias(daily),
        session=session,
    )

    confirmations: List[str] = []

    trend  = _check_trend(daily)
    if trend:
        confirmations.append(trend)

    sd_zone = _check_sd_zone(ticker, bars, daily)
    if sd_zone:
        confirmations.append(sd_zone)

    vol = _check_volume(bars)
    if vol:
        confirmations.append(vol)

    struct = _check_structure(daily)
    if struct:
        confirmations.append(struct)

    signal.confirmations = confirmations

    # Fake breakout filter – use most recent close as reference level
    level = 0.0
    if bars is not None and "close" in bars.columns and len(bars) > 0:
        level = float(bars["close"].iloc[-1])
    signal.fake_breakout_passed = passes_fake_breakout_filter(bars, level=level)

    # Scoring
    strengths = {
        "trend_alignment": 1.0 if trend   else 0.0,
        "supply_demand":   1.0 if sd_zone else 0.0,
        "volume_confirm":  1.0 if vol     else 0.0,
        "structure":       1.0 if struct  else 0.0,
        "fake_breakout":   1.0 if signal.fake_breakout_passed else 0.0,
    }
    signal.score = compute_strategy_score("intraday", strengths)
    signal.grade = score_to_grade(signal.score)

    # Trade parameters
    if bars is not None and "close" in bars.columns and len(bars) > 0:
        entry = float(bars["close"].iloc[-1])
        atr   = compute_atr(daily) if daily is not None else None
        trade = compute_trade_params("intraday", entry=entry, atr=atr)
        signal.entry  = trade.entry
        signal.stop   = trade.stop
        signal.target = trade.target
        signal.rr     = trade.rr

    signal.valid = (
        signal.score >= _MIN_SCORE
        and signal.confirmation_count >= _MIN_CONFIRMATIONS
        and signal.fake_breakout_passed
        and signal.rr >= 2.5
    )

    return signal
