"""
signals/golden_sweep.py – Golden Sweep / institutional options flow detection.

A "Golden Sweep" is an unusually large options order (sweep or block trade)
that signals institutional directional conviction.  This module detects such
sweeps, classifies their trade-type, checks underlying alignment, and produces
a :class:`SweepAlert` ready for formatting and delivery.

Detection criteria
------------------
* Contract volume ≥ :data:`SWEEP_VOLUME_MULT` × average volume
* (Optional) Estimated notional ≥ :data:`SWEEP_MIN_NOTIONAL`
* DTE classification:

  * 0–7 DTE → Day Trade (weekly options)
  * 8–21 DTE → Swing Trade
  * >21 DTE → Position Trade

Underlying alignment (required for HIGH confidence):
----------------------------------------------------
1. RVOL ≥ 1.5
2. Volume spike ≥ 3× average
3. Momentum in same direction as sweep
4. Price near VWAP (within :data:`SWEEP_VWAP_PROXIMITY_PCT`)

Confidence thresholds
---------------------
* alignment_score ≥ 3 → ``"High"``
* alignment_score ≥ 2 → ``"Medium"``
* alignment_score < 2 → no alert (suppressed)

Alert cooldown: :data:`SWEEP_COOLDOWN_SECONDS` per ticker (15 min).

Usage::

    from momentum_radar.signals.golden_sweep import detect_golden_sweep

    alert = detect_golden_sweep(ticker, options_flow, bars, daily)
    if alert:
        print(alert.confidence, alert.direction, alert.entry)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from momentum_radar.utils.indicators import compute_atr, compute_rvol, compute_vwap

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

#: Minimum contract volume multiple vs. average to qualify as a sweep.
SWEEP_VOLUME_MULT: float = 3.0

#: Minimum estimated notional flow ($) to qualify as institutional (when available).
SWEEP_MIN_NOTIONAL: float = 100_000.0

#: Maximum DTE for day-trade focus (weekly options).
SWEEP_DAY_TRADE_MAX_DTE: int = 7

#: Maximum DTE for swing-trade focus.
SWEEP_SWING_TRADE_MAX_DTE: int = 21

#: Minimum RVOL for underlying alignment check.
SWEEP_MIN_RVOL: float = 1.5

#: Minimum volume spike (× avg) for underlying alignment check.
SWEEP_MIN_VOLUME_SPIKE: float = 3.0

#: Price proximity to VWAP (as a fraction of price) for "near VWAP" check.
SWEEP_VWAP_PROXIMITY_PCT: float = 0.02

#: Cooldown between sweep alerts for the same ticker (seconds).
SWEEP_COOLDOWN_SECONDS: int = 900  # 15 minutes


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_float(d: dict, key: str, default: float = 0.0) -> float:
    """Safely extract a float from *d*, treating ``None`` as *default*."""
    return float(d.get(key) or default)


# ---------------------------------------------------------------------------
# SweepAlert dataclass
# ---------------------------------------------------------------------------

@dataclass
class SweepAlert:
    """A fully-described Golden Sweep alert ready for formatting and delivery.

    Attributes:
        ticker:           Stock symbol.
        direction:        ``"bullish"`` (calls) or ``"bearish"`` (puts).
        contract_type:    ``"call"`` or ``"put"``.
        strike:           Option strike price.
        expiration:       Human-readable expiration label (e.g. ``"Weekly"``).
        dte:              Days to expiration.
        contract_volume:  Number of contracts traded in the sweep.
        estimated_flow:   Approximate notional value in USD.
        underlying_price: Current stock price at detection time.
        entry:            Suggested entry price for the underlying.
        stop:             Suggested stop-loss price.
        target:           Suggested profit-target price.
        rvol:             Relative volume of the underlying.
        volume_spike:     Intraday volume spike multiplier (× avg bar volume).
        zone_alignment:   Description of nearest supply/demand or VWAP level.
        confidence:       ``"High"``, ``"Medium"``, or ``"Low"``.
        trade_type:       ``"Day Trade"``, ``"Swing Trade"``, or ``"Position Trade"``.
        timestamp:        When the sweep was detected.
        details:          Human-readable summary of triggering conditions.
    """

    ticker: str
    direction: str
    contract_type: str
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
    confidence: str
    trade_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: str = ""

    @property
    def risk_reward(self) -> float:
        """Risk:reward ratio (reward / risk).  Returns 0.0 if risk is zero."""
        risk = abs(self.entry - self.stop)
        reward = abs(self.target - self.entry)
        return round(reward / risk, 1) if risk > 0 else 0.0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _classify_dte(dte: int) -> str:
    """Classify trade type based on days-to-expiration."""
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
) -> tuple[int, str]:
    """Score how well the underlying aligns with the sweep direction (0–4).

    Checks:
    1. RVOL ≥ :data:`SWEEP_MIN_RVOL`
    2. Volume spike ≥ :data:`SWEEP_MIN_VOLUME_SPIKE`
    3. Price momentum in the sweep direction
    4. Price near VWAP (within :data:`SWEEP_VWAP_PROXIMITY_PCT`)

    Returns:
        Tuple of (alignment_score, zone_description_string).
    """
    score = 0
    zone_desc = "No key level nearby"

    # 1. RVOL
    if rvol >= SWEEP_MIN_RVOL:
        score += 1

    # 2. Volume spike
    if volume_spike >= SWEEP_MIN_VOLUME_SPIKE:
        score += 1

    # 3. Momentum: compare recent close to close 5 bars ago
    if bars is not None and not bars.empty and "close" in bars.columns and len(bars) >= 5:
        closes = bars["close"].iloc[-5:].tolist()
        if direction == "bullish" and closes[-1] > closes[0]:
            score += 1
        elif direction == "bearish" and closes[-1] < closes[0]:
            score += 1

    # 4. Price near VWAP
    if vwap is not None and vwap > 0 and current_price > 0:
        proximity = abs(current_price - vwap) / vwap
        if proximity <= SWEEP_VWAP_PROXIMITY_PCT:
            score += 1
            zone_desc = f"Near VWAP ${vwap:.2f}"

    return score, zone_desc


def _compute_entry_stop_target(
    direction: str,
    current_price: float,
    daily: Optional[pd.DataFrame],
) -> tuple[float, float, float]:
    """Derive entry, stop, and target from current price and ATR.

    Uses ATR-based offsets; falls back to a 1.5 % default if ATR is unavailable.

    Returns:
        (entry, stop, target) rounded to 2 decimal places.
    """
    atr_pct = 0.015  # 1.5 % default
    if daily is not None and not daily.empty and current_price > 0:
        atr = compute_atr(daily)
        if atr is not None and atr > 0:
            atr_pct = atr / current_price

    if direction == "bullish":
        entry = round(current_price * 1.001, 2)
        stop = round(entry * (1 - atr_pct), 2)
        target = round(entry * (1 + atr_pct * 2), 2)
    else:
        entry = round(current_price * 0.999, 2)
        stop = round(entry * (1 + atr_pct), 2)
        target = round(entry * (1 - atr_pct * 2), 2)

    return entry, stop, target


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_golden_sweep(
    ticker: str,
    options_flow: Optional[Dict],
    bars: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
) -> Optional[SweepAlert]:
    """Detect a Golden Sweep from options flow and underlying price data.

    The *options_flow* dict may contain the following keys (all optional
    except where noted):

    * ``sweep_type``      – ``"call"`` or ``"put"`` (if a large sweep was observed)
    * ``sweep_volume``    – number of contracts in the sweep
    * ``sweep_strike``    – strike price of the sweep contract
    * ``sweep_dte``       – days to expiration
    * ``sweep_expiration``– expiration label (e.g. ``"2024-01-26"``)
    * ``sweep_premium``   – estimated notional value in USD
    * ``call_volume``     – total call volume for the session
    * ``put_volume``      – total put volume for the session
    * ``avg_call_volume`` – 30-day average daily call volume
    * ``avg_put_volume``  – 30-day average daily put volume

    If ``sweep_type`` / ``sweep_volume`` are not provided the function falls
    back to total call/put volume vs. their averages.

    Args:
        ticker:       Stock symbol.
        options_flow: Options flow data dict (see above).
        bars:         Intraday OHLCV DataFrame (1-min preferred).
        daily:        Daily OHLCV DataFrame.

    Returns:
        :class:`SweepAlert` if a qualifying sweep is detected, else ``None``.
    """
    if options_flow is None:
        return None

    # ------------------------------------------------------------------
    # Step 1: Determine sweep direction and volume
    # ------------------------------------------------------------------
    sweep_type = str(options_flow.get("sweep_type", "")).lower()
    sweep_volume = int(options_flow.get("sweep_volume", 0) or 0)

    if sweep_type in ("call", "put") and sweep_volume > 0:
        # Explicit sweep data provided – pick the matching average
        if sweep_type == "call":
            avg_vol = _get_float(options_flow, "avg_call_volume", 1.0) or 1.0
        else:
            avg_vol = _get_float(options_flow, "avg_put_volume", 1.0) or 1.0
    else:
        # Fall back: compare total call/put volume to averages
        call_vol = _get_float(options_flow, "call_volume")
        put_vol = _get_float(options_flow, "put_volume")
        avg_call = _get_float(options_flow, "avg_call_volume", 1.0) or 1.0
        avg_put = _get_float(options_flow, "avg_put_volume", 1.0) or 1.0

        call_ratio = call_vol / avg_call if avg_call > 0 else 0.0
        put_ratio = put_vol / avg_put if avg_put > 0 else 0.0

        if call_ratio < SWEEP_VOLUME_MULT and put_ratio < SWEEP_VOLUME_MULT:
            logger.debug("%s – options volume below sweep threshold; no sweep", ticker)
            return None

        if call_ratio >= put_ratio:
            sweep_type = "call"
            sweep_volume = int(call_vol)
            avg_vol = avg_call
        else:
            sweep_type = "put"
            sweep_volume = int(put_vol)
            avg_vol = avg_put

    # ------------------------------------------------------------------
    # Step 2: Volume ratio gate
    # ------------------------------------------------------------------
    vol_ratio = sweep_volume / avg_vol if avg_vol > 0 else 0.0
    if vol_ratio < SWEEP_VOLUME_MULT:
        logger.debug(
            "%s – sweep volume ratio %.1fx < %.1fx threshold; no sweep",
            ticker, vol_ratio, SWEEP_VOLUME_MULT,
        )
        return None

    # ------------------------------------------------------------------
    # Step 3: Notional gate (soft check – allowed to bypass if vol is very high)
    # ------------------------------------------------------------------
    sweep_premium = _get_float(options_flow, "sweep_premium")
    if sweep_premium > 0 and sweep_premium < SWEEP_MIN_NOTIONAL:
        if vol_ratio < SWEEP_VOLUME_MULT * 2:
            logger.debug(
                "%s – sweep notional $%.0f below threshold; no sweep",
                ticker, sweep_premium,
            )
            return None

    # ------------------------------------------------------------------
    # Step 4: Underlying price
    # ------------------------------------------------------------------
    current_price = 0.0
    if bars is not None and not bars.empty and "close" in bars.columns:
        current_price = float(bars["close"].iloc[-1])
    elif daily is not None and not daily.empty and "close" in daily.columns:
        current_price = float(daily["close"].iloc[-1])

    if current_price <= 0:
        logger.debug("%s – cannot determine current price; no sweep alert", ticker)
        return None

    # ------------------------------------------------------------------
    # Step 5: Compute underlying metrics
    # ------------------------------------------------------------------
    rvol_raw = compute_rvol(bars, daily) if daily is not None else None
    rvol = float(rvol_raw) if rvol_raw is not None else 0.0

    volume_spike = 0.0
    if bars is not None and not bars.empty and "volume" in bars.columns and len(bars) > 1:
        avg_bar_vol = float(bars["volume"].iloc[:-1].mean())
        if avg_bar_vol > 0:
            volume_spike = round(float(bars["volume"].iloc[-1]) / avg_bar_vol, 2)

    vwap = compute_vwap(bars) if bars is not None else None

    # ------------------------------------------------------------------
    # Step 6: Alignment check
    # ------------------------------------------------------------------
    direction = "bullish" if sweep_type == "call" else "bearish"
    alignment_score, zone_desc = _assess_alignment(
        bars, direction, rvol, volume_spike, vwap, current_price
    )

    if alignment_score >= 3:
        confidence = "High"
    elif alignment_score >= 2:
        confidence = "Medium"
    else:
        # Low alignment – suppress to avoid noise
        logger.debug(
            "%s – sweep alignment score %d/4 too low; no alert", ticker, alignment_score
        )
        return None

    # ------------------------------------------------------------------
    # Step 7: DTE and expiration
    # ------------------------------------------------------------------
    dte = int(options_flow.get("sweep_dte", 0) or 0)
    trade_type = _classify_dte(dte)

    raw_expiration = options_flow.get("sweep_expiration", "")
    if raw_expiration:
        expiration = str(raw_expiration)
    elif dte <= SWEEP_DAY_TRADE_MAX_DTE:
        expiration = "Weekly"
    else:
        expiration = f"{dte}d"

    strike = _get_float(options_flow, "sweep_strike", current_price) or current_price

    # ------------------------------------------------------------------
    # Step 8: Entry / stop / target
    # ------------------------------------------------------------------
    entry, stop, target = _compute_entry_stop_target(direction, current_price, daily)

    # ------------------------------------------------------------------
    # Build and return the alert
    # ------------------------------------------------------------------
    return SweepAlert(
        ticker=ticker,
        direction=direction,
        contract_type=sweep_type,
        strike=strike,
        expiration=expiration,
        dte=dte,
        contract_volume=sweep_volume,
        estimated_flow=sweep_premium,
        underlying_price=round(current_price, 2),
        entry=entry,
        stop=stop,
        target=target,
        rvol=round(rvol, 1),
        volume_spike=round(volume_spike, 1),
        zone_alignment=zone_desc,
        confidence=confidence,
        trade_type=trade_type,
        details=(
            f"{sweep_type.title()} sweep {vol_ratio:.1f}× avg | "
            f"DTE {dte} | alignment {alignment_score}/4 | "
            f"RVOL {rvol:.1f} | vol spike {volume_spike:.1f}×"
        ),
    )
