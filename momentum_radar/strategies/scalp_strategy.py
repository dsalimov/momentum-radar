"""
strategies/scalp_strategy.py – Scalp Engine.

Timeframe
---------
* 2m  (09:30–10:00, opening momentum window)
* 5m  (after 10:00 if momentum continues)

Entry requirements
------------------
1. Strong momentum   – RSI 45–72 and positive MACD histogram
2. Volume spike      – intraday volume ≥ 1.5× 30-bar average
3. Break of structure – close above prior 10-bar high
4. HTF alignment      – daily price > EMA21 > EMA50
5. Fake-breakout filter passed

Quality gates
-------------
* Score ≥ 75 / 100
* ≥ 3 confirmations
* Fake-breakout filter passed
* R:R ≥ 2.0
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
from momentum_radar.services.session_manager import get_current_session
from momentum_radar.strategies.base import StrategySignal
from momentum_radar.utils.indicators import compute_atr

logger = logging.getLogger(__name__)

_MIN_SCORE: int = 75
_MIN_CONFIRMATIONS: int = 3

_SESSION_TIMEFRAME: Dict[str, str] = {
    "open":      "2m",
    "morning":   "5m",
    "midday":    "5m",
    "afternoon": "5m",
}


def _rsi(closes: pd.Series, period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    val = float((100 - 100 / (1 + rs)).iloc[-1])
    return None if math.isnan(val) else val


def _check_momentum(bars: Optional[pd.DataFrame]) -> Optional[str]:
    """RSI 45-72 and positive MACD histogram."""
    if bars is None or len(bars) < 30 or "close" not in bars.columns:
        return None
    closes = bars["close"]
    rsi = _rsi(closes)
    if rsi is None:
        return None
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    hist = float(
        (ema12 - ema26 - (ema12 - ema26).ewm(span=9, adjust=False).mean()).iloc[-1]
    )
    if 45 <= rsi <= 72 and hist > 0:
        return f"Momentum Alignment (RSI {rsi:.0f})"
    return None


def _check_volume(bars: Optional[pd.DataFrame]) -> Optional[str]:
    """Intraday volume ≥ 1.5× 30-bar average."""
    if bars is None or len(bars) < 31 or "volume" not in bars.columns:
        return None
    avg = float(bars["volume"].iloc[-31:-1].mean())
    last = float(bars["volume"].iloc[-1])
    if avg > 0 and last >= avg * 1.5:
        return f"Volume Spike ({last / avg:.1f}x)"
    return None


def _check_structure(bars: Optional[pd.DataFrame]) -> Optional[str]:
    """Close above prior 10-bar high."""
    if bars is None or len(bars) < 12 or "close" not in bars.columns:
        return None
    last_close = float(bars["close"].iloc[-1])
    prior_high = float(bars["high"].iloc[-12:-1].max())
    if last_close > prior_high:
        return "Break of Structure"
    return None


def _check_htf(daily: Optional[pd.DataFrame]) -> Optional[str]:
    """Daily price > EMA21 > EMA50 (HTF bullish alignment)."""
    if daily is None or len(daily) < 50 or "close" not in daily.columns:
        return None
    closes = daily["close"]
    last = float(closes.iloc[-1])
    ema21 = float(closes.ewm(span=21, adjust=False).mean().iloc[-1])
    ema50 = float(closes.ewm(span=50, adjust=False).mean().iloc[-1])
    if last > ema21 > ema50:
        return "HTF Alignment"
    return None


def evaluate(
    ticker: str,
    bars: Optional[pd.DataFrame] = None,
    daily: Optional[pd.DataFrame] = None,
    options: Optional[Dict] = None,
    fundamentals: Optional[Dict] = None,
    now: Optional[datetime] = None,
) -> StrategySignal:
    """Evaluate the scalp strategy for *ticker*.

    Args:
        ticker:       Stock symbol.
        bars:         Intraday OHLCV DataFrame (2m or 5m bars).
        daily:        Daily OHLCV DataFrame.
        options:      Options activity dict (accepted for interface consistency).
        fundamentals: Fundamental data dict (unused by this strategy).
        now:          Override current datetime (for testing).

    Returns:
        :class:`~momentum_radar.strategies.base.StrategySignal`.
    """
    session   = get_current_session(now)
    timeframe = _SESSION_TIMEFRAME.get(session, "5m")

    signal = StrategySignal(
        ticker=ticker,
        strategy="scalp",
        direction="BUY",
        timeframe=timeframe,
        regime=get_regime_display(daily),
        htf_bias=get_htf_bias(daily),
        session=session,
    )

    # --- Confirmation checks ---
    confirmations: List[str] = []

    mom  = _check_momentum(bars)
    if mom:
        confirmations.append(mom)

    vol = _check_volume(bars)
    if vol:
        confirmations.append(vol)

    struct = _check_structure(bars)
    if struct:
        confirmations.append(struct)

    htf = _check_htf(daily)
    if htf:
        confirmations.append(htf)

    signal.confirmations = confirmations

    # --- Fake breakout filter ---
    level = 0.0
    if bars is not None and "high" in bars.columns and len(bars) >= 12:
        level = float(bars["high"].iloc[-12:-1].max())
    elif bars is not None and "close" in bars.columns and len(bars) > 0:
        level = float(bars["close"].iloc[-1])
    signal.fake_breakout_passed = passes_fake_breakout_filter(bars, level=level)

    # --- Scoring ---
    strengths = {
        "momentum":        1.0 if mom    else 0.0,
        "volume_spike":    1.0 if vol    else 0.0,
        "structure_break": 1.0 if struct else 0.0,
        "htf_bias":        1.0 if htf    else 0.0,
        "fake_breakout":   1.0 if signal.fake_breakout_passed else 0.0,
    }
    signal.score = compute_strategy_score("scalp", strengths)
    signal.grade = score_to_grade(signal.score)

    # --- Trade parameters ---
    if bars is not None and "close" in bars.columns and len(bars) > 0:
        entry = float(bars["close"].iloc[-1])
        atr   = compute_atr(daily) if daily is not None else None
        trade = compute_trade_params("scalp", entry=entry, atr=atr)
        signal.entry  = trade.entry
        signal.stop   = trade.stop
        signal.target = trade.target
        signal.rr     = trade.rr

    # --- Quality gate ---
    signal.valid = (
        signal.score >= _MIN_SCORE
        and signal.confirmation_count >= _MIN_CONFIRMATIONS
        and signal.fake_breakout_passed
        and signal.rr >= 2.0
    )

    return signal
