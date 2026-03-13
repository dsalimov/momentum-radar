"""
options/golden_sweep_detector.py – Institutional options Golden Sweep detector.

A **Golden Sweep** is an unusually large, aggressive options order that signals
institutional conviction. This module detects sweeps and produces
:class:`SweepAlert` objects that drive the alert pipeline.

Detection criteria
------------------
1. Options volume ≥ ``SWEEP_VOLUME_MULT`` × average → size gate.
2. RVOL ≥ ``SWEEP_MIN_RVOL`` on the underlying → confirms unusual stock activity.
3. Volume spike on the underlying confirms unusual activity.
4. Directional alignment between sweep and intraday price action.
5. Minimum alignment score of 2/4 before generating an alert.

Sweep classification by DTE
----------------------------
- 0–``SWEEP_DAY_TRADE_MAX_DTE`` days  → **Day Trade**
- ``SWEEP_DAY_TRADE_MAX_DTE``+1–``SWEEP_SWING_TRADE_MAX_DTE`` → **Swing Trade**
- ``SWEEP_SWING_TRADE_MAX_DTE``+1+ → **Position Trade**

Flow dict schema
----------------
Primary (explicit sweep) keys::

    sweep_type:       "call" | "put"
    sweep_volume:     int – number of contracts in the sweep
    avg_call_volume:  int – 20-period average call volume
    avg_put_volume:   int – 20-period average put volume
    sweep_strike:     float – option strike price
    sweep_dte:        int – days to expiry
    sweep_expiration: str – expiry label (e.g. "Weekly")
    sweep_premium:    float – estimated notional premium in dollars

Fallback (aggregate volume) keys::

    call_volume:     int – total call volume
    put_volume:      int – total put volume
    avg_call_volume: int
    avg_put_volume:  int
    sweep_strike:    float
    sweep_dte:       int
    sweep_premium:   float

Usage::

    from momentum_radar.options.golden_sweep_detector import (
        detect_golden_sweep, SweepAlert
    )

    alert = detect_golden_sweep(ticker, flow_dict, intraday_bars, daily_bars)
    if alert is not None:
        print(alert.direction, alert.trade_type, alert.entry)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from momentum_radar.utils.indicators import compute_atr, compute_rvol, compute_vwap

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Minimum ratio of sweep volume to average volume to trigger a signal.
SWEEP_VOLUME_MULT: float = 3.0

#: Minimum RVOL on underlying to confirm unusual stock activity.
SWEEP_MIN_RVOL: float = 1.5

#: Maximum DTE for a Day Trade classification.
SWEEP_DAY_TRADE_MAX_DTE: int = 7

#: Maximum DTE for a Swing Trade classification.
SWEEP_SWING_TRADE_MAX_DTE: int = 21

#: Cooldown period in seconds between repeat alerts for the same ticker.
SWEEP_COOLDOWN_SECONDS: int = 900

# ---------------------------------------------------------------------------
# Private thresholds
# ---------------------------------------------------------------------------

#: Minimum volume spike multiplier (last bar vs recent average) for alignment.
_MIN_VOLUME_SPIKE: float = 3.0

#: VWAP proximity tolerance as a fraction of price.
_VWAP_PROXIMITY_PCT: float = 0.01

#: Minimum alignment score (0–4) required to produce an alert.
_MIN_ALIGNMENT_SCORE: int = 2

#: Minimum risk:reward to accept a generated trade plan.
_MIN_RISK_REWARD: float = 1.5

#: Fallback ATR when no daily data is available (fraction of price).
_ATR_FALLBACK_PCT: float = 0.005

#: Entry price buffer above/below current price (fraction).
_ENTRY_BUFFER_PCT: float = 0.001


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class SweepAlert:
    """A fully-qualified Golden Sweep trade alert.

    Attributes:
        ticker:           Stock symbol.
        direction:        ``"bullish"`` (calls) or ``"bearish"`` (puts).
        contract_type:    ``"call"`` or ``"put"``.
        strike:           Option strike price.
        expiration:       Expiry label (e.g. ``"Weekly"`` or ISO date string).
        dte:              Calendar days to expiry.
        contract_volume:  Sweep volume in number of contracts.
        estimated_flow:   Estimated notional premium in dollars.
        underlying_price: Underlying stock price at signal time.
        entry:            Suggested stock entry price.
        stop:             Stop-loss price on the underlying.
        target:           Profit-target price on the underlying.
        rvol:             Relative volume of the underlying.
        volume_spike:     Last-bar volume as a multiple of the recent average.
        zone_alignment:   Description of nearest supply/demand zone alignment.
        confidence:       ``"High"`` or ``"Medium"``.
        trade_type:       ``"Day Trade"``, ``"Swing Trade"``, or ``"Position Trade"``.
        timestamp:        When the alert was generated.
        details:          Human-readable detection summary.
    """

    ticker: str
    direction: str           # "bullish" | "bearish"
    contract_type: str       # "call" | "put"
    strike: float
    expiration: str
    dte: int
    contract_volume: int
    estimated_flow: float
    underlying_price: float
    entry: float
    stop: float
    target: float
    rvol: float
    volume_spike: float
    zone_alignment: str
    confidence: str          # "High" | "Medium"
    trade_type: str          # "Day Trade" | "Swing Trade" | "Position Trade"
    timestamp: datetime = field(default_factory=datetime.now)
    details: str = ""

    @property
    def risk_reward(self) -> float:
        """Risk:reward ratio (reward / risk).  Returns 0.0 when risk is zero."""
        risk = abs(self.entry - self.stop)
        reward = abs(self.target - self.entry)
        if risk <= 0:
            return 0.0
        return round(reward / risk, 1)


# ---------------------------------------------------------------------------
# Helper functions (also exported for direct testing)
# ---------------------------------------------------------------------------

def _classify_dte(dte: int) -> str:
    """Return a trade-type label based on days to expiry.

    Args:
        dte: Calendar days to expiry (≥ 0).

    Returns:
        ``"Day Trade"``, ``"Swing Trade"``, or ``"Position Trade"``.
    """
    if dte <= SWEEP_DAY_TRADE_MAX_DTE:
        return "Day Trade"
    if dte <= SWEEP_SWING_TRADE_MAX_DTE:
        return "Swing Trade"
    return "Position Trade"


def _assess_alignment(
    bars: Optional[pd.DataFrame],
    direction: str,
    rvol: float,
    volume_spike: float,
    vwap: Optional[float],
    current_price: float,
) -> Tuple[int, str]:
    """Score the alignment between the sweep direction and underlying behaviour.

    Scoring (0–4 points):

    +1 RVOL ≥ ``SWEEP_MIN_RVOL``
    +1 Volume spike ≥ ``_MIN_VOLUME_SPIKE``
    +1 Price within ``_VWAP_PROXIMITY_PCT`` of VWAP
    +1 Intraday EMA5/EMA20 momentum matches sweep direction

    Args:
        bars:          Intraday OHLCV DataFrame.
        direction:     ``"bullish"`` or ``"bearish"``.
        rvol:          Relative volume of the underlying.
        volume_spike:  Last-bar volume spike multiplier.
        vwap:          Current VWAP value, or ``None`` if unavailable.
        current_price: Current underlying price.

    Returns:
        Tuple of ``(score, description_string)``.
    """
    score = 0
    reasons: List[str] = []

    if rvol >= SWEEP_MIN_RVOL:
        score += 1
        reasons.append(f"RVOL {rvol:.1f}x")

    if volume_spike >= _MIN_VOLUME_SPIKE:
        score += 1
        reasons.append(f"vol spike {volume_spike:.1f}x")

    if vwap is not None and current_price > 0:
        proximity = abs(current_price - vwap) / current_price
        if proximity <= _VWAP_PROXIMITY_PCT:
            score += 1
            reasons.append("near VWAP")

    if bars is not None and len(bars) >= 10 and "close" in bars.columns:
        closes = bars["close"]
        ema_fast = float(closes.ewm(span=5, adjust=False).mean().iloc[-1])
        ema_slow = float(closes.ewm(span=20, adjust=False).mean().iloc[-1])
        if direction == "bullish" and ema_fast > ema_slow * 1.001:
            score += 1
            reasons.append("bullish momentum")
        elif direction == "bearish" and ema_fast < ema_slow * 0.999:
            score += 1
            reasons.append("bearish momentum")

    description = " | ".join(reasons) if reasons else "no alignment"
    return score, description


def _compute_entry_stop_target(
    direction: str,
    current_price: float,
    daily: Optional[pd.DataFrame],
) -> Tuple[float, float, float]:
    """Compute ATR-based entry, stop-loss, and profit target.

    The entry is placed just above (bullish) or below (bearish) current price
    to avoid a stale fill.  The stop is one ATR away from entry; the target is
    two risk-units beyond entry (2× risk).

    Args:
        direction:     ``"bullish"`` or ``"bearish"``.
        current_price: Current underlying price.
        daily:         Daily OHLCV DataFrame used to compute ATR.

    Returns:
        Tuple of ``(entry, stop, target)``.
    """
    atr = compute_atr(daily) if daily is not None else None
    if atr is None or atr <= 0:
        atr = current_price * _ATR_FALLBACK_PCT

    if direction == "bullish":
        entry = round(current_price * (1.0 + _ENTRY_BUFFER_PCT), 2)
        stop = round(entry - atr, 2)
        risk = entry - stop
        target = round(entry + 2.0 * risk, 2)
    else:
        entry = round(current_price * (1.0 - _ENTRY_BUFFER_PCT), 2)
        stop = round(entry + atr, 2)
        risk = stop - entry
        target = round(entry - 2.0 * risk, 2)

    return entry, stop, target


def _nearest_zone(
    bars: pd.DataFrame,
    current_price: float,
    direction: str,
    lookback: int = 30,
    tolerance_pct: float = 0.015,
) -> str:
    """Return the nearest supply/demand zone description or ``"N/A"``."""
    if bars is None or bars.empty:
        return "N/A"

    df = bars.tail(lookback)
    price_series = df["low"] if direction == "bullish" else df["high"]
    threshold = current_price * tolerance_pct
    nearby = price_series[abs(price_series - current_price) <= threshold]
    if len(nearby) < 2:
        return "N/A"

    zone_low = round(float(nearby.min()), 2)
    zone_high = round(float(nearby.max()), 2)
    label = "Demand Zone" if direction == "bullish" else "Supply Zone"
    return f"{label} {zone_low:.2f}–{zone_high:.2f}"


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------

def detect_golden_sweep(
    ticker: str,
    flow: Optional[Dict],
    bars: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
) -> Optional[SweepAlert]:
    """Detect an institutional Golden Sweep and return a :class:`SweepAlert`.

    The function supports two flow dict schemas:

    1. **Explicit sweep** – ``flow`` contains ``sweep_type`` + ``sweep_volume``.
    2. **Aggregate fallback** – ``flow`` contains ``call_volume`` / ``put_volume``
       totals; the dominant side is used if its ratio ≥ ``SWEEP_VOLUME_MULT``.

    If no price data (*bars*) is available the function returns ``None`` because
    entry, stop and target cannot be computed.  The signal is also suppressed
    when the alignment score is below ``_MIN_ALIGNMENT_SCORE`` or the computed
    risk:reward ratio is below ``_MIN_RISK_REWARD``.

    Args:
        ticker: Stock symbol.
        flow:   Options flow dict (see module docstring for key schemas).
        bars:   Intraday OHLCV DataFrame (1-min preferred).
        daily:  Daily OHLCV DataFrame (30+ days for ATR/RVOL).

    Returns:
        :class:`SweepAlert` if all gates pass, otherwise ``None``.
    """
    if flow is None:
        return None

    # ------------------------------------------------------------------
    # Step 1 – Determine direction, contract type, and volumes
    # ------------------------------------------------------------------
    direction: Optional[str] = None
    contract_type: Optional[str] = None
    sweep_volume: int = 0
    avg_vol: int = 1  # denominator guard

    if "sweep_type" in flow and "sweep_volume" in flow:
        sweep_type_raw = str(flow.get("sweep_type", "")).lower()
        vol = int(flow.get("sweep_volume", 0))
        avg_call = int(flow.get("avg_call_volume", 1) or 1)
        avg_put = int(flow.get("avg_put_volume", 1) or 1)

        if sweep_type_raw == "call":
            direction = "bullish"
            contract_type = "call"
            avg_vol = avg_call
            sweep_volume = vol
        elif sweep_type_raw == "put":
            direction = "bearish"
            contract_type = "put"
            avg_vol = avg_put
            sweep_volume = vol
        else:
            logger.debug("%s – Unknown sweep_type '%s', skipping", ticker, sweep_type_raw)
            return None
    else:
        # Fallback: derive direction from aggregate call/put volume
        call_vol = int(flow.get("call_volume", 0) or 0)
        put_vol = int(flow.get("put_volume", 0) or 0)
        avg_call = int(flow.get("avg_call_volume", 1) or 1)
        avg_put = int(flow.get("avg_put_volume", 1) or 1)

        call_ratio = call_vol / avg_call if avg_call > 0 else 0.0
        put_ratio = put_vol / avg_put if avg_put > 0 else 0.0

        if call_ratio < SWEEP_VOLUME_MULT and put_ratio < SWEEP_VOLUME_MULT:
            logger.debug(
                "%s – Aggregate flow below threshold (call %.1fx, put %.1fx)",
                ticker, call_ratio, put_ratio,
            )
            return None

        if call_ratio >= put_ratio:
            direction = "bullish"
            contract_type = "call"
            sweep_volume = call_vol
            avg_vol = avg_call
        else:
            direction = "bearish"
            contract_type = "put"
            sweep_volume = put_vol
            avg_vol = avg_put

    # ------------------------------------------------------------------
    # Step 2 – Volume threshold gate
    # ------------------------------------------------------------------
    if avg_vol <= 0 or sweep_volume < SWEEP_VOLUME_MULT * avg_vol:
        logger.debug(
            "%s – Sweep volume %d below %.1fx threshold (avg %d)",
            ticker, sweep_volume, SWEEP_VOLUME_MULT, avg_vol,
        )
        return None

    # ------------------------------------------------------------------
    # Step 3 – Current price (requires intraday bars)
    # ------------------------------------------------------------------
    if bars is None or bars.empty or "close" not in bars.columns:
        logger.debug("%s – No intraday bars; cannot determine current price", ticker)
        return None

    current_price = float(bars["close"].iloc[-1])
    if current_price <= 0:
        return None

    # ------------------------------------------------------------------
    # Step 4 – RVOL and volume spike
    # ------------------------------------------------------------------
    rvol = 0.0
    if daily is not None and not daily.empty:
        rvol_val = compute_rvol(bars, daily)
        if rvol_val is not None:
            rvol = float(rvol_val)

    vol_spike = 0.0
    if len(bars) >= 2 and "volume" in bars.columns:
        lookback_end = max(1, len(bars) - 1)
        lookback_start = max(0, lookback_end - 20)
        recent_avg = float(bars["volume"].iloc[lookback_start:lookback_end].mean())
        if recent_avg > 0:
            vol_spike = round(float(bars["volume"].iloc[-1]) / recent_avg, 2)

    # ------------------------------------------------------------------
    # Step 5 – VWAP
    # ------------------------------------------------------------------
    vwap: Optional[float] = None
    try:
        vwap = compute_vwap(bars)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Step 6 – Alignment score gate
    # ------------------------------------------------------------------
    alignment_score, alignment_desc = _assess_alignment(
        bars, direction, rvol, vol_spike, vwap, current_price,
    )

    if alignment_score < _MIN_ALIGNMENT_SCORE:
        logger.debug(
            "%s – Alignment score %d/%d below minimum; skipping sweep alert",
            ticker, alignment_score, 4,
        )
        return None

    # ------------------------------------------------------------------
    # Step 7 – DTE classification
    # ------------------------------------------------------------------
    dte = int(flow.get("sweep_dte", 0) or 0)
    trade_type = _classify_dte(dte)

    # ------------------------------------------------------------------
    # Step 8 – Entry / stop / target
    # ------------------------------------------------------------------
    entry, stop, target = _compute_entry_stop_target(direction, current_price, daily)

    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk <= 0 or reward / risk < _MIN_RISK_REWARD:
        logger.debug(
            "%s – R:R %.1f below minimum %.1f; skipping sweep alert",
            ticker, reward / risk if risk > 0 else 0.0, _MIN_RISK_REWARD,
        )
        return None

    # ------------------------------------------------------------------
    # Step 9 – Build SweepAlert
    # ------------------------------------------------------------------
    expiration = flow.get("sweep_expiration") or (
        f"{dte}d" if dte > 0 else "Unknown"
    )
    strike = float(flow.get("sweep_strike", 0.0) or 0.0)
    estimated_flow = float(flow.get("sweep_premium", 0.0) or 0.0)
    confidence = "High" if rvol >= 2.0 and vol_spike >= 4.0 else "Medium"
    zone_alignment = _nearest_zone(bars, current_price, direction)

    ratio = sweep_volume / avg_vol if avg_vol > 0 else 0.0
    details = (
        f"{contract_type} sweep {ratio:.1f}× avg | "
        f"DTE {dte} | alignment {alignment_score}/4"
    )

    logger.info(
        "Golden Sweep detected: %s %s %s %s entry=%.2f stop=%.2f target=%.2f "
        "RR=%.1f conf=%s",
        ticker, direction, contract_type, trade_type,
        entry, stop, target, reward / risk, confidence,
    )

    return SweepAlert(
        ticker=ticker,
        direction=direction,
        contract_type=contract_type,
        strike=strike,
        expiration=expiration,
        dte=dte,
        contract_volume=sweep_volume,
        estimated_flow=estimated_flow,
        underlying_price=current_price,
        entry=entry,
        stop=stop,
        target=target,
        rvol=round(rvol, 1),
        volume_spike=round(vol_spike, 1),
        zone_alignment=zone_alignment,
        confidence=confidence,
        trade_type=trade_type,
        details=details,
    )
