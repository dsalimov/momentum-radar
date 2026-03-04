"""
signals/zone_retest.py – Supply & Demand zone retest confirmation engine.

When price returns to a previously identified zone, this module evaluates
whether the interaction constitutes a *confirmed* retest or a *fake* probe.

Confirmation criteria (minimum 2 required for an alert):

1. **Volume reaction** – above-average volume during the retest bar.
2. **Rejection wick** – large tail pointing away from the zone.
3. **Engulfing candle** – current bar engulfs the previous bar's body.
4. **Momentum shift** – close vs. open direction changes relative to impulse.
5. **Fake breakout filter** – price spiked beyond zone but closed back inside.

Scoring:

A retest is issued as an alert only when:
* ``confirmation_count >= RETEST_MIN_CONFIRMATIONS`` (default 2)
* The zone's ``strength_score >= RETEST_MIN_ZONE_SCORE`` (default 75)

Usage::

    import pandas as pd
    from momentum_radar.signals.zone_retest import evaluate_retest

    result = evaluate_retest(zone, bars, atr)
    if result["confirmed"]:
        print(result["confirmations"])
        print(result["score"])
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

from momentum_radar.signals.supply_demand import SupplyDemandZone

logger = logging.getLogger(__name__)

# Minimum zone strength required before a retest alert is raised
RETEST_MIN_ZONE_SCORE: float = 75.0
# Minimum number of confirmation criteria that must be met
RETEST_MIN_CONFIRMATIONS: int = 2
# Minimum wick-to-range ratio to qualify as a rejection wick
_REJECTION_WICK_RATIO: float = 0.40
# Volume must be at least this multiple of the average to count as reaction
_RETEST_VOLUME_MULT: float = 1.20


def _check_volume_reaction(
    bars: pd.DataFrame,
    avg_volume: float,
) -> bool:
    """Return ``True`` if the last bar has above-average volume (retest volume).

    Args:
        bars:       OHLCV DataFrame; the last row is the current retest bar.
        avg_volume: Rolling average volume over the lookback window.

    Returns:
        ``True`` when volume confirms the retest.
    """
    if bars is None or bars.empty or avg_volume <= 0:
        return False
    last_vol = float(bars["volume"].iloc[-1])
    return last_vol >= _RETEST_VOLUME_MULT * avg_volume


def _check_rejection_wick(
    bars: pd.DataFrame,
    zone_type: str,
) -> bool:
    """Return ``True`` if the last bar shows a rejection wick away from the zone.

    For a **demand** zone (price coming from below):
        Lower wick / (high – low) >= ``_REJECTION_WICK_RATIO``

    For a **supply** zone (price coming from above):
        Upper wick / (high – low) >= ``_REJECTION_WICK_RATIO``

    Args:
        bars:       OHLCV DataFrame.
        zone_type:  ``"demand"`` or ``"supply"``.

    Returns:
        ``True`` when a meaningful rejection wick is present.
    """
    if bars is None or bars.empty:
        return False
    last = bars.iloc[-1]
    candle_range = float(last["high"]) - float(last["low"])
    if candle_range <= 0:
        return False

    if zone_type == "demand":
        lower_wick = float(min(last["open"], last["close"])) - float(last["low"])
        return (lower_wick / candle_range) >= _REJECTION_WICK_RATIO
    else:  # supply
        upper_wick = float(last["high"]) - float(max(last["open"], last["close"]))
        return (upper_wick / candle_range) >= _REJECTION_WICK_RATIO


def _check_engulfing(bars: pd.DataFrame, zone_type: str) -> bool:
    """Return ``True`` if the last bar's body engulfs the previous bar's body.

    For **demand**: current close > current open (bullish) and the body size
    is larger than the previous bar's body.

    For **supply**: current close < current open (bearish) and the body size
    is larger than the previous bar's body.

    Args:
        bars:      OHLCV DataFrame (at least 2 rows).
        zone_type: ``"demand"`` or ``"supply"``.

    Returns:
        ``True`` when an engulfing pattern is detected.
    """
    if bars is None or len(bars) < 2:
        return False

    prev = bars.iloc[-2]
    curr = bars.iloc[-1]
    prev_body = abs(float(prev["close"]) - float(prev["open"]))
    curr_body = abs(float(curr["close"]) - float(curr["open"]))

    if zone_type == "demand":
        bullish = float(curr["close"]) > float(curr["open"])
        return bullish and curr_body > prev_body and float(curr["close"]) > float(prev["open"])
    else:  # supply
        bearish = float(curr["close"]) < float(curr["open"])
        return bearish and curr_body > prev_body and float(curr["close"]) < float(prev["open"])


def _check_momentum_shift(bars: pd.DataFrame, zone_type: str) -> bool:
    """Return ``True`` if the last few bars show a momentum shift in zone direction.

    For **demand**: at least 1 of the last 3 bars closed bullish (close > open)
    **and** the most recent close is the highest close of those 3 bars.

    For **supply**: at least 1 of the last 3 bars closed bearish (close < open)
    **and** the most recent close is the lowest close of those 3 bars.

    Args:
        bars:      OHLCV DataFrame.
        zone_type: ``"demand"`` or ``"supply"``.

    Returns:
        ``True`` when a momentum shift is detected.
    """
    if bars is None or len(bars) < 3:
        return False

    recent = bars.iloc[-3:]
    closes = recent["close"].values
    opens = recent["open"].values
    last_close = closes[-1]

    if zone_type == "demand":
        bullish_count = sum(1 for c, o in zip(closes, opens) if c > o)
        return bullish_count >= 1 and last_close == max(closes)
    else:  # supply
        bearish_count = sum(1 for c, o in zip(closes, opens) if c < o)
        return bearish_count >= 1 and last_close == min(closes)


def _check_fake_breakout_reclaim(
    bars: pd.DataFrame,
    zone: SupplyDemandZone,
) -> bool:
    """Return ``True`` if price spiked beyond the zone boundary then reclaimed it.

    This is the classic "fake breakout → reclaim" setup, which has high
    follow-through probability.

    For **demand**: a recent low breached ``zone_low`` but the latest close
    is back inside the zone.

    For **supply**: a recent high pierced ``zone_high`` but the latest close
    is back inside the zone.

    Args:
        bars:  OHLCV DataFrame.
        zone:  The zone being tested.

    Returns:
        ``True`` when a fake breakout reclaim is detected.
    """
    if bars is None or len(bars) < 2:
        return False

    recent = bars.iloc[-4:] if len(bars) >= 4 else bars.iloc[-2:]
    last_close = float(bars["close"].iloc[-1])

    if zone.zone_type == "demand":
        spiked_below = float(recent["low"].min()) < zone.zone_low
        reclaimed = zone.zone_low <= last_close <= zone.zone_high
        return spiked_below and reclaimed
    else:  # supply
        spiked_above = float(recent["high"].max()) > zone.zone_high
        reclaimed = zone.zone_low <= last_close <= zone.zone_high
        return spiked_above and reclaimed


def evaluate_retest(
    zone: SupplyDemandZone,
    bars: pd.DataFrame,
    atr: float,
) -> Dict:
    """Evaluate whether the current price action constitutes a confirmed retest.

    Args:
        zone:  The zone being retested.
        bars:  Recent OHLCV bars (should include the current retest bar).
        atr:   Current ATR for context.

    Returns:
        Dict with keys:

        - ``confirmed``      – ``True`` when retest is confirmed (≥ min criteria met
                               and zone score ≥ threshold)
        - ``confirmations``  – list of confirmation names that passed
        - ``score``          – retest confidence score (0–100)
        - ``zone_score_ok``  – whether the zone's strength meets the threshold
    """
    confirmations: List[str] = []

    if bars is None or bars.empty:
        return {"confirmed": False, "confirmations": [], "score": 0, "zone_score_ok": False}

    avg_vol = float(bars["volume"].mean()) if "volume" in bars.columns else 0.0

    if _check_volume_reaction(bars, avg_vol):
        confirmations.append("volume_reaction")

    if _check_rejection_wick(bars, zone.zone_type):
        confirmations.append("rejection_wick")

    if _check_engulfing(bars, zone.zone_type):
        confirmations.append("engulfing_candle")

    if _check_momentum_shift(bars, zone.zone_type):
        confirmations.append("momentum_shift")

    if _check_fake_breakout_reclaim(bars, zone):
        confirmations.append("fake_breakout_reclaim")

    count = len(confirmations)
    # Score: proportional to confirmations (0–100)
    score = int(min(count / 5.0 * 100.0, 100.0))

    zone_score_ok = zone.strength_score >= RETEST_MIN_ZONE_SCORE
    confirmed = (
        count >= RETEST_MIN_CONFIRMATIONS
        and zone_score_ok
    )

    if confirmed:
        logger.debug(
            "Retest confirmed for %s %s zone [%.2f–%.2f]: %s",
            zone.ticker, zone.zone_type, zone.zone_low, zone.zone_high,
            confirmations,
        )

    return {
        "confirmed": confirmed,
        "confirmations": confirmations,
        "score": score,
        "zone_score_ok": zone_score_ok,
    }
