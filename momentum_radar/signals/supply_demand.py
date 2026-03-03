"""
signals/supply_demand.py – Institutional supply & demand zone detection.

Registered signals
------------------
- ``supply_demand_zone`` – fires when the current price enters a scored
  supply or demand zone detected from price structure

Detection algorithm
-------------------
Demand zones form from:
  1. A *base* – 3-to-7 bars of tight range (range < ATR threshold)
  2. Immediately followed by an *impulse* upward move (> ``_IMPULSE_ATR_MULT`` × ATR)
  3. Volume during impulse > ``_IMPULSE_VOL_MULT`` × average volume
  4. The zone spans [base_low … base_high]

Supply zones use the same logic with a downward impulse.

Zone scoring (0–100)
--------------------
- Impulse magnitude  : 0–30 pts
- Volume expansion   : 0–20 pts
- Base tightness     : 0–15 pts
- Zone freshness     : 25 pts (fresh) / 15 (tested once) / 5 (tested ≥2) / 0 (broken)
- Timeframe weight   : 0–10 pts  (Daily > shorter timeframes)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from momentum_radar.signals.base import SignalResult
from momentum_radar.signals.scoring import register_signal
from momentum_radar.utils.indicators import compute_atr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Algorithm constants
# ---------------------------------------------------------------------------

# How many bars form a "base" (consolidation)
_BASE_MIN_BARS: int = 2
_BASE_MAX_BARS: int = 7
# Base range must be tighter than this fraction of ATR
_BASE_ATR_MULT: float = 0.80
# Impulse must be at least this multiple of ATR to qualify
_IMPULSE_ATR_MULT: float = 1.20
# Volume during impulse must exceed this multiple of the prior average
_IMPULSE_VOL_MULT: float = 1.30
# How close price must be to a zone to trigger (fraction of zone height or ATR)
_ZONE_PROXIMITY_MULT: float = 0.50
# Minimum zone height relative to ATR (filters micro-zones)
_ZONE_MIN_HEIGHT_ATR: float = 0.10


# ---------------------------------------------------------------------------
# Zone dataclass
# ---------------------------------------------------------------------------

@dataclass
class SupplyDemandZone:
    """A single supply or demand zone detected from price structure.

    Attributes:
        ticker:              Stock symbol.
        timeframe:           Timeframe the zone was detected on (e.g. ``"daily"``).
        zone_type:           ``"demand"`` or ``"supply"``.
        zone_high:           Upper boundary of the zone.
        zone_low:            Lower boundary of the zone.
        strength_score:      Zone quality score (0–100).
        touch_count:         Number of times price has retested the zone.
        status:              ``"fresh"`` / ``"tested"`` / ``"broken"`` / ``"flipped"``.
        impulse_magnitude:   Size of the originating impulse in ATR multiples.
        volume_expansion:    Volume ratio during impulse vs. average.
        creation_bar_index:  Index of the base start bar in the source DataFrame.
    """

    ticker: str
    timeframe: str
    zone_type: str            # "demand" | "supply"
    zone_high: float
    zone_low: float
    strength_score: float
    touch_count: int = 0
    status: str = "fresh"     # "fresh" | "tested" | "broken" | "flipped"
    impulse_magnitude: float = 0.0
    volume_expansion: float = 0.0
    creation_bar_index: int = 0

    @property
    def mid_price(self) -> float:
        """Midpoint of the zone."""
        return (self.zone_high + self.zone_low) / 2

    @property
    def height(self) -> float:
        """Zone height."""
        return self.zone_high - self.zone_low

    @property
    def strength_label(self) -> str:
        """Human-readable strength category."""
        if self.strength_score >= 80:
            return "Institutional"
        if self.strength_score >= 65:
            return "Strong"
        if self.strength_score >= 50:
            return "Moderate"
        return "Weak"


# ---------------------------------------------------------------------------
# Timeframe weights (higher TF = higher weight)
# ---------------------------------------------------------------------------

_TF_SCORE_BONUS: Dict[str, float] = {
    "weekly": 10.0,
    "daily":  8.0,
    "4h":     6.0,
    "1h":     4.0,
    "15m":    2.0,
    "5m":     0.0,
}


# ---------------------------------------------------------------------------
# Zone detection helpers
# ---------------------------------------------------------------------------

def _detect_zones(
    df: pd.DataFrame,
    atr: float,
    ticker: str,
    timeframe: str,
    avg_volume: float,
) -> List[SupplyDemandZone]:
    """Scan *df* for supply and demand zones.

    Args:
        df:         OHLCV DataFrame (any timeframe).
        atr:        ATR for the series (used as volatility reference).
        ticker:     Stock symbol.
        timeframe:  Label for the timeframe (e.g. ``"daily"``).
        avg_volume: Average volume (baseline for volume expansion check).

    Returns:
        List of detected :class:`SupplyDemandZone` objects, sorted by
        strength score descending.
    """
    if len(df) < _BASE_MAX_BARS + 3 or atr is None or atr <= 0:
        return []

    zones: List[SupplyDemandZone] = []
    n = len(df)

    for i in range(_BASE_MIN_BARS, n - 3):
        for base_len in range(_BASE_MIN_BARS, min(_BASE_MAX_BARS + 1, i + 1)):
            base_start = i - base_len + 1
            base = df.iloc[base_start : i + 1]

            base_range = float(base["high"].max() - base["low"].min())
            if base_range > _BASE_ATR_MULT * atr:
                continue  # Too wide for a base

            # ----- Upward impulse (demand zone candidate) -----
            impulse_up = df.iloc[i + 1 : i + 4]
            if len(impulse_up) >= 1:
                impulse_move_up = float(impulse_up["close"].max()) - float(df.iloc[i]["close"])
                if impulse_move_up >= _IMPULSE_ATR_MULT * atr:
                    imp_vol = float(impulse_up["volume"].mean()) if "volume" in impulse_up.columns else 0.0
                    vol_ratio = imp_vol / avg_volume if avg_volume > 0 else 1.0
                    if vol_ratio >= _IMPULSE_VOL_MULT:
                        zone_high = float(base["high"].max())
                        zone_low = float(base["low"].min())
                        if zone_high - zone_low >= _ZONE_MIN_HEIGHT_ATR * atr:
                            score = _compute_score(
                                impulse_magnitude=impulse_move_up / atr,
                                volume_expansion=vol_ratio,
                                base_range=base_range,
                                atr=atr,
                                touch_count=0,
                                status="fresh",
                                timeframe=timeframe,
                            )
                            zones.append(
                                SupplyDemandZone(
                                    ticker=ticker,
                                    timeframe=timeframe,
                                    zone_type="demand",
                                    zone_high=zone_high,
                                    zone_low=zone_low,
                                    strength_score=round(score, 1),
                                    touch_count=0,
                                    status="fresh",
                                    impulse_magnitude=round(impulse_move_up / atr, 2),
                                    volume_expansion=round(vol_ratio, 2),
                                    creation_bar_index=base_start,
                                )
                            )
                            break  # Found a valid demand base at this position

            # ----- Downward impulse (supply zone candidate) -----
            impulse_dn = df.iloc[i + 1 : i + 4]
            if len(impulse_dn) >= 1:
                impulse_move_dn = float(df.iloc[i]["close"]) - float(impulse_dn["close"].min())
                if impulse_move_dn >= _IMPULSE_ATR_MULT * atr:
                    imp_vol = float(impulse_dn["volume"].mean()) if "volume" in impulse_dn.columns else 0.0
                    vol_ratio = imp_vol / avg_volume if avg_volume > 0 else 1.0
                    if vol_ratio >= _IMPULSE_VOL_MULT:
                        zone_high = float(base["high"].max())
                        zone_low = float(base["low"].min())
                        if zone_high - zone_low >= _ZONE_MIN_HEIGHT_ATR * atr:
                            score = _compute_score(
                                impulse_magnitude=impulse_move_dn / atr,
                                volume_expansion=vol_ratio,
                                base_range=base_range,
                                atr=atr,
                                touch_count=0,
                                status="fresh",
                                timeframe=timeframe,
                            )
                            zones.append(
                                SupplyDemandZone(
                                    ticker=ticker,
                                    timeframe=timeframe,
                                    zone_type="supply",
                                    zone_high=zone_high,
                                    zone_low=zone_low,
                                    strength_score=round(score, 1),
                                    touch_count=0,
                                    status="fresh",
                                    impulse_magnitude=round(impulse_move_dn / atr, 2),
                                    volume_expansion=round(vol_ratio, 2),
                                    creation_bar_index=base_start,
                                )
                            )
                            break

    # Deduplicate overlapping zones of the same type
    zones = _deduplicate_zones(zones)
    # Update touch count and status based on subsequent price action
    _update_zone_lifecycle(zones, df)

    zones.sort(key=lambda z: z.strength_score, reverse=True)
    return zones


def _compute_score(
    impulse_magnitude: float,
    volume_expansion: float,
    base_range: float,
    atr: float,
    touch_count: int,
    status: str,
    timeframe: str,
) -> float:
    """Compute a zone strength score (0–100).

    Breakdown:
    - Impulse magnitude (0–30 pts): larger impulse → stronger zone
    - Volume expansion  (0–20 pts): more volume → more conviction
    - Base tightness    (0–15 pts): tighter base → cleaner origin
    - Zone freshness    (0–25 pts): fewer touches → higher probability
    - Timeframe bonus   (0–10 pts): higher TF → more institutional weight

    Args:
        impulse_magnitude: Impulse size in ATR multiples.
        volume_expansion:  Volume ratio vs. average.
        base_range:        Range of the base in price units.
        atr:               ATR value.
        touch_count:       Number of retests so far.
        status:            Zone status string.
        timeframe:         Timeframe label.

    Returns:
        Strength score (0–100).
    """
    # Impulse magnitude: saturates at 3× ATR
    impulse_pts = min(impulse_magnitude / 3.0 * 30.0, 30.0)

    # Volume expansion: saturates at 3× average
    vol_pts = min((volume_expansion - 1.0) / 2.0 * 20.0, 20.0)
    vol_pts = max(vol_pts, 0.0)

    # Base tightness: tighter = better (base_range / atr; lower is better)
    tightness_ratio = base_range / atr if atr > 0 else 1.0
    tightness_pts = max(0.0, (1.0 - tightness_ratio) * 15.0)

    # Zone freshness
    if status == "broken":
        freshness_pts = 0.0
    elif touch_count == 0:
        freshness_pts = 25.0
    elif touch_count == 1:
        freshness_pts = 15.0
    else:
        freshness_pts = 5.0

    # Timeframe bonus
    tf_pts = _TF_SCORE_BONUS.get(timeframe.lower(), 0.0)

    total = impulse_pts + vol_pts + tightness_pts + freshness_pts + tf_pts
    return min(total, 100.0)


def _deduplicate_zones(zones: List[SupplyDemandZone]) -> List[SupplyDemandZone]:
    """Remove overlapping zones of the same type, keeping the strongest."""
    result: List[SupplyDemandZone] = []
    for zone in sorted(zones, key=lambda z: z.strength_score, reverse=True):
        overlap = any(
            z.zone_type == zone.zone_type
            and z.zone_low <= zone.zone_high
            and zone.zone_low <= z.zone_high
            for z in result
        )
        if not overlap:
            result.append(zone)
    return result


def _update_zone_lifecycle(
    zones: List[SupplyDemandZone],
    df: pd.DataFrame,
) -> None:
    """Update ``touch_count`` and ``status`` for each zone based on *df*.

    A *touch* occurs when the closing price enters the zone.
    A *break* occurs when two consecutive closes are beyond the zone boundary
    with above-average volume.

    Args:
        zones: List of zones to update (modified in-place).
        df:    OHLCV DataFrame used to evaluate subsequent price action.
    """
    if df is None or df.empty:
        return

    avg_vol = float(df["volume"].mean()) if "volume" in df.columns else 0.0

    for zone in zones:
        bar_start = zone.creation_bar_index + 1  # examine bars *after* base
        if bar_start >= len(df):
            continue

        subsequent = df.iloc[bar_start:]
        closes = subsequent["close"].values
        vols = subsequent["volume"].values if "volume" in subsequent.columns else np.ones(len(closes))

        in_zone_count = 0
        consec_breaks = 0

        for j, (c, v) in enumerate(zip(closes, vols)):
            inside = zone.zone_low <= c <= zone.zone_high
            if inside and in_zone_count == 0:
                zone.touch_count += 1
                in_zone_count += 1
            elif not inside:
                in_zone_count = 0

            # Break detection (2 consecutive closes beyond boundary with volume)
            if zone.zone_type == "demand" and c < zone.zone_low and v >= avg_vol * 0.8:
                consec_breaks += 1
            elif zone.zone_type == "supply" and c > zone.zone_high and v >= avg_vol * 0.8:
                consec_breaks += 1
            else:
                consec_breaks = 0

            if consec_breaks >= 2:
                zone.status = "broken"
                break

        if zone.status != "broken":
            if zone.touch_count >= 2:
                zone.status = "tested"
            elif zone.touch_count == 1:
                zone.status = "tested"
            else:
                zone.status = "fresh"

        # Refresh score with updated touch count
        if zone.status != "broken":
            zone.strength_score = round(
                _compute_score(
                    impulse_magnitude=zone.impulse_magnitude,
                    volume_expansion=zone.volume_expansion,
                    base_range=0.0,  # already factored in at creation
                    atr=1.0,
                    touch_count=zone.touch_count,
                    status=zone.status,
                    timeframe=zone.timeframe,
                ),
                1,
            )


def detect_zones(
    ticker: str,
    daily: Optional[pd.DataFrame],
    bars: Optional[pd.DataFrame] = None,
    min_score: float = 50.0,
) -> List[SupplyDemandZone]:
    """Detect supply and demand zones across available timeframes.

    Uses daily bars for Daily/Weekly zones and, if intraday bars are
    provided and long enough, resampled bars for 1H / 15M / 5M zones.

    Args:
        ticker:    Stock symbol.
        daily:     Daily OHLCV DataFrame (≥ 30 bars recommended).
        bars:      Intraday 1-min OHLCV DataFrame (optional).
        min_score: Minimum strength score to keep a zone.

    Returns:
        All zones with ``strength_score ≥ min_score``, sorted by score
        descending.
    """
    all_zones: List[SupplyDemandZone] = []

    # --- Daily zones ---
    if daily is not None and len(daily) >= 15:
        atr = compute_atr(daily) or 0.0
        avg_vol = float(daily["volume"].mean()) if "volume" in daily.columns and not daily.empty else 0.0
        if atr > 0:
            all_zones += _detect_zones(daily, atr, ticker, "daily", avg_vol)

    # --- Weekly zones (aggregate daily to weekly) ---
    if daily is not None and len(daily) >= 30:
        try:
            weekly = daily.resample("W").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna()
            if len(weekly) >= 10:
                atr_w = compute_atr(weekly) or 0.0
                avg_vol_w = float(weekly["volume"].mean()) if "volume" in weekly.columns else 0.0
                if atr_w > 0:
                    all_zones += _detect_zones(weekly, atr_w, ticker, "weekly", avg_vol_w)
        except Exception as exc:
            logger.debug("Weekly aggregation failed for %s: %s", ticker, exc)

    # --- Intraday zones (1H, 15M, 5M) from 1-min bars ---
    if bars is not None and not bars.empty and len(bars) >= 30:
        for tf_label, rule in [("1h", "1h"), ("15m", "15min"), ("5m", "5min")]:
            try:
                tf_df = bars.resample(rule).agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
                ).dropna()
                if len(tf_df) >= 10:
                    atr_tf = compute_atr(tf_df) or 0.0
                    avg_vol_tf = float(tf_df["volume"].mean()) if "volume" in tf_df.columns else 0.0
                    if atr_tf > 0:
                        all_zones += _detect_zones(tf_df, atr_tf, ticker, tf_label, avg_vol_tf)
            except Exception as exc:
                logger.debug("Intraday %s resample failed for %s: %s", tf_label, ticker, exc)

    # Filter by minimum score and deduplicate across timeframes
    filtered = [z for z in all_zones if z.strength_score >= min_score and z.status != "broken"]
    filtered.sort(key=lambda z: z.strength_score, reverse=True)
    return filtered


def get_active_zone(
    ticker: str,
    current_price: float,
    zones: List[SupplyDemandZone],
    atr: float,
) -> Optional[SupplyDemandZone]:
    """Return the strongest active zone that *current_price* is near or inside.

    "Near" means within ``_ZONE_PROXIMITY_MULT × ATR`` of the zone boundary.

    Args:
        ticker:        Stock symbol (for logging).
        current_price: Latest price.
        zones:         Candidate zones list.
        atr:           Current ATR (proximity threshold denominator).

    Returns:
        The best matching :class:`SupplyDemandZone`, or ``None``.
    """
    proximity_buffer = _ZONE_PROXIMITY_MULT * atr if atr > 0 else 0.0

    for zone in zones:  # already sorted by score
        inside = zone.zone_low <= current_price <= zone.zone_high
        near_demand = (
            zone.zone_type == "demand"
            and current_price >= zone.zone_low - proximity_buffer
            and current_price <= zone.zone_high + proximity_buffer
        )
        near_supply = (
            zone.zone_type == "supply"
            and current_price >= zone.zone_low - proximity_buffer
            and current_price <= zone.zone_high + proximity_buffer
        )
        if inside or near_demand or near_supply:
            logger.debug(
                "%s: price %.2f near %s %s zone [%.2f–%.2f] score=%.0f",
                ticker,
                current_price,
                zone.strength_label,
                zone.zone_type,
                zone.zone_low,
                zone.zone_high,
                zone.strength_score,
            )
            return zone

    return None


# ---------------------------------------------------------------------------
# Registered signal
# ---------------------------------------------------------------------------

@register_signal("supply_demand_zone")
def supply_demand_zone(
    ticker: str,
    bars: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
    **kwargs,
) -> SignalResult:
    """Fire when the current price enters or approaches a scored S&D zone.

    Score contribution:
    - +2 for Institutional or Strong zone (score ≥ 65) on daily/weekly
    - +1 for Moderate zone or intraday-only zone

    Args:
        ticker: Stock symbol.
        bars:   Intraday 1-min OHLCV DataFrame.
        daily:  Daily OHLCV DataFrame.

    Returns:
        :class:`~momentum_radar.signals.base.SignalResult`
    """
    if daily is None or daily.empty:
        return SignalResult(triggered=False, score=0, details="No daily data for S&D detection")

    atr = compute_atr(daily)
    if atr is None or atr <= 0:
        return SignalResult(triggered=False, score=0, details="Could not compute ATR")

    # Current price: prefer intraday last close
    if bars is not None and not bars.empty and "close" in bars.columns:
        current_price = float(bars["close"].iloc[-1])
    else:
        current_price = float(daily["close"].iloc[-1])

    zones = detect_zones(ticker, daily, bars, min_score=50.0)
    active_zone = get_active_zone(ticker, current_price, zones, atr)

    if active_zone is None:
        return SignalResult(
            triggered=False,
            score=0,
            details="Price not near any scored supply/demand zone",
        )

    # High-strength zones on daily/weekly timeframes get a full +2
    high_strength = active_zone.strength_score >= 65
    high_tf = active_zone.timeframe in ("daily", "weekly")

    if high_strength and high_tf:
        score = 2
    else:
        score = 1

    touch_label = (
        "fresh"
        if active_zone.touch_count == 0
        else f"tested {active_zone.touch_count}×"
    )

    return SignalResult(
        triggered=True,
        score=score,
        details=(
            f"{active_zone.strength_label} {active_zone.zone_type.title()} Zone "
            f"[${active_zone.zone_low:.2f}–${active_zone.zone_high:.2f}] "
            f"({active_zone.timeframe}, {touch_label}, "
            f"strength {active_zone.strength_score:.0f}/100)"
        ),
    )
