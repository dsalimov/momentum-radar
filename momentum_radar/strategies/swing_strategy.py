"""
strategies/swing_strategy.py – Swing Trade Engine.

Timeframe
---------
* 1H / 4H / Daily

Entry requirements
------------------
1. Higher-timeframe S&D zone   – price in a scored daily/weekly zone
2. Major structure break        – close above prior 20-day swing high
3. Displacement (impulse)       – large-body candle ≥ 2× ATR
4. Liquidity sweep confirmation – recent wick below swing low then reversal
5. Volume on breakout bar       – volume above 30-day average

Quality gates
-------------
* Score ≥ 75 / 100
* ≥ 3 confirmations
* R:R ≥ 3.0
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
from momentum_radar.core.structure_engine import detect_structure_break
from momentum_radar.core.supply_demand import get_demand_zones, price_in_zone
from momentum_radar.strategies.base import StrategySignal
from momentum_radar.utils.indicators import compute_atr

logger = logging.getLogger(__name__)

_MIN_SCORE: int = 75
_MIN_CONFIRMATIONS: int = 3
_TIMEFRAME: str = "1H"


def _check_htf_zone(
    ticker: str,
    daily: Optional[pd.DataFrame],
) -> Optional[str]:
    """Price entering a scored daily demand zone."""
    if daily is None or "close" not in daily.columns:
        return None
    price = float(daily["close"].iloc[-1])
    zones = get_demand_zones(ticker, daily, min_score=70.0, timeframe="daily")
    for zone in zones:
        if price_in_zone(price, zone, buffer_pct=0.01):
            return f"HTF Demand Zone (score {zone.strength_score:.0f})"
    return None


def _check_structure(daily: Optional[pd.DataFrame]) -> Optional[str]:
    """Major structure break on the daily timeframe."""
    result = detect_structure_break(daily, lookback=20)
    if result.confirmed and result.direction == "bullish":
        return f"Major BOS (+{result.break_pct:.1f}%)"
    return None


def _check_displacement(daily: Optional[pd.DataFrame]) -> Optional[str]:
    """Last daily candle body ≥ 2× ATR (impulsive move)."""
    if daily is None or len(daily) < 15 or "open" not in daily.columns:
        return None
    atr = compute_atr(daily)
    if not atr or atr <= 0:
        return None
    last_body = abs(float(daily["close"].iloc[-1]) - float(daily["open"].iloc[-1]))
    if last_body >= 2.0 * atr:
        return f"Displacement ({last_body / atr:.1f}× ATR)"
    return None


def _check_liquidity_sweep(daily: Optional[pd.DataFrame]) -> Optional[str]:
    """Recent wick below swing low with strong close above it."""
    if daily is None or len(daily) < 10:
        return None
    lows   = daily["low"]
    closes = daily["close"]
    prior_low = float(lows.iloc[-11:-1].min())
    last_low  = float(lows.iloc[-1])
    last_close = float(closes.iloc[-1])
    if last_low < prior_low and last_close > prior_low:
        return "Liquidity Sweep Confirmed"
    return None


def _check_volume(daily: Optional[pd.DataFrame]) -> Optional[str]:
    """Breakout bar volume above 30-day average."""
    if daily is None or len(daily) < 32 or "volume" not in daily.columns:
        return None
    avg  = float(daily["volume"].iloc[-32:-1].mean())
    last = float(daily["volume"].iloc[-1])
    if avg > 0 and last >= avg * 1.2:
        return f"Volume Expansion ({last / avg:.1f}x)"
    return None


def evaluate(
    ticker: str,
    bars: Optional[pd.DataFrame] = None,
    daily: Optional[pd.DataFrame] = None,
    options: Optional[Dict] = None,
    fundamentals: Optional[Dict] = None,
    now: Optional[datetime] = None,
) -> StrategySignal:
    """Evaluate the swing strategy for *ticker*.

    Args:
        ticker:       Stock symbol.
        bars:         Intraday bars (unused; kept for interface consistency).
        daily:        Daily OHLCV DataFrame (primary data source).
        options:      Options activity dict (unused).
        fundamentals: Fundamental data dict (unused).
        now:          Override current datetime (unused).

    Returns:
        :class:`~momentum_radar.strategies.base.StrategySignal`.
    """
    signal = StrategySignal(
        ticker=ticker,
        strategy="swing",
        direction="BUY",
        timeframe=_TIMEFRAME,
        regime=get_regime_display(daily),
        htf_bias=get_htf_bias(daily),
    )

    confirmations: List[str] = []

    htf_zone = _check_htf_zone(ticker, daily)
    if htf_zone:
        confirmations.append(htf_zone)

    struct = _check_structure(daily)
    if struct:
        confirmations.append(struct)

    disp = _check_displacement(daily)
    if disp:
        confirmations.append(disp)

    liq = _check_liquidity_sweep(daily)
    if liq:
        confirmations.append(liq)

    vol = _check_volume(daily)
    if vol:
        confirmations.append(vol)

    signal.confirmations = confirmations

    # Fake breakout filter on daily bars
    level = 0.0
    if daily is not None and "high" in daily.columns and len(daily) >= 22:
        level = float(daily["high"].iloc[-22:-1].max())
    signal.fake_breakout_passed = passes_fake_breakout_filter(daily, level=level)

    # Scoring
    strengths = {
        "htf_zone":        1.0 if htf_zone else 0.0,
        "displacement":    1.0 if disp     else 0.0,
        "structure_break": 1.0 if struct   else 0.0,
        "liquidity_sweep": 1.0 if liq      else 0.0,
        "volume_confirm":  1.0 if vol      else 0.0,
    }
    signal.score = compute_strategy_score("swing", strengths)
    signal.grade = score_to_grade(signal.score)

    # Trade parameters
    if daily is not None and "close" in daily.columns and len(daily) > 0:
        entry = float(daily["close"].iloc[-1])
        atr   = compute_atr(daily)
        trade = compute_trade_params("swing", entry=entry, atr=atr)
        signal.entry   = trade.entry
        signal.stop    = trade.stop
        signal.target  = trade.target
        signal.target2 = trade.target2
        signal.rr      = trade.rr

    signal.valid = (
        signal.score >= _MIN_SCORE
        and signal.confirmation_count >= _MIN_CONFIRMATIONS
        and signal.rr >= 3.0
    )

    return signal
