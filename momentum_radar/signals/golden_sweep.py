"""
golden_sweep.py – Institutional options Golden Sweep detector.

A **Golden Sweep** is an unusually large, aggressive options order (sweep or
block trade) characterised by:

* Volume exceeds open interest on the contract (strong directional conviction).
* Minimum contract count threshold (``GOLDEN_SWEEP_MIN_CONTRACTS``).
* Short-dated expiry suggesting tactical, conviction-driven flow.

Sweep classification
--------------------
- **Weekly sweep** (0–7 days to expiry) → Day Trade signal.
- **2-3 Week sweep** (8–21 days to expiry) → Swing Trade signal.
- **Monthly sweep** (22+ days to expiry) → Swing/positional context.

Underlying confirmation
-----------------------
Before a Golden Sweep is flagged as a high-probability setup the bot checks:

1. RVOL ≥ 1.5 on the underlying stock.
2. Volume spike ≥ 3× recent average on the underlying.
3. Sweep direction aligned with intraday momentum (bullish sweep → intraday
   uptrend, bearish sweep → intraday downtrend).
4. Minimum risk:reward ≥ 1.5:1 on the resulting trade.

Usage::

    from momentum_radar.signals.golden_sweep import detect_golden_sweep, GoldenSweepSetup

    setup = detect_golden_sweep(ticker, options_flow_data, bars, daily)
    if setup is not None:
        print(setup.direction, setup.sweep_type, setup.contracts)
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
from datetime import date, datetime
from typing import Dict, List, Optional

import pandas as pd

from momentum_radar.config import config
from momentum_radar.signals.base import SignalResult
from momentum_radar.signals.scoring import register_signal
from momentum_radar.utils.indicators import compute_atr, compute_rvol
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from momentum_radar.utils.indicators import compute_atr, compute_rvol, compute_vwap

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (overridable via config)
# ---------------------------------------------------------------------------

#: Minimum risk:reward ratio for a sweep-based trade setup.
MIN_RISK_REWARD: float = 1.5
#: ATR multiplier used to set the stop-loss distance from entry.
STOP_ATR_MULT: float = 1.0
#: Target is this multiple of risk beyond entry.
TARGET_RR_MULT: float = 2.0


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class GoldenSweepSetup:
    """A fully-qualified Golden Sweep trade setup.

    Attributes:
        ticker:           Stock symbol.
        direction:        ``"Bullish"`` (calls) or ``"Bearish"`` (puts).
        sweep_type:       ``"Weekly"``, ``"2-3 Week"``, or ``"Monthly"``.
        trade_type:       ``"Day Trade"`` or ``"Swing Trade"``.
        contract_type:    ``"Call"`` or ``"Put"``.
        strike:           Option strike price.
        expiry:           Expiry date string (ISO format, e.g. ``"2024-11-15"``).
        contracts:        Sweep volume (number of contracts).
        underlying_price: Underlying stock price at signal time.
        entry:            Suggested stock entry price.
        stop:             Stop-loss price on the underlying.
        target:           Profit-target price on the underlying.
        rvol:             Relative volume of the underlying.
        volume_spike:     Volume spike multiplier on the underlying.
        supply_demand_zone: Optional description of nearest S/D zone.
        confidence:       ``"High"``, ``"Medium"``, or ``"Low"``.
        timestamp:        When the setup was detected.
        details:          Human-readable description.
    """

    ticker: str
    direction: str                    # "Bullish" | "Bearish"
    sweep_type: str                   # "Weekly" | "2-3 Week" | "Monthly"
    trade_type: str                   # "Day Trade" | "Swing Trade"
    contract_type: str                # "Call" | "Put"
    strike: float
    expiry: str
    contracts: int
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
    supply_demand_zone: Optional[str] = None
    confidence: str = "High"
    zone_alignment: str
    confidence: str
    trade_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: str = ""

    @property
    def risk_reward(self) -> float:
        """Risk:reward ratio (reward / risk).  Returns 0 if risk is zero."""
        risk = abs(self.entry - self.stop)
        reward = abs(self.target - self.entry)
        if risk <= 0:
            return 0.0
        return round(reward / risk, 1)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _days_to_expiry(expiry_str: str) -> Optional[int]:
    """Return calendar days from today to *expiry_str* (ISO date string).

    Returns ``None`` if the string cannot be parsed.
    """
    try:
        expiry_date = date.fromisoformat(expiry_str)
        return (expiry_date - date.today()).days
    except (ValueError, TypeError):
        return None


def _classify_sweep_type(days: Optional[int]) -> tuple[str, str]:
    """Return ``(sweep_type, trade_type)`` based on days to expiry.

    Args:
        days: Calendar days to expiry (``None`` → unknown).

    Returns:
        Tuple of sweep type label and trade type label.
    """
    cfg = config.signals
    if days is None:
        return "Unknown", "Swing Trade"
    if days <= cfg.golden_sweep_weekly_days:
        return "Weekly", "Day Trade"
    if days <= cfg.golden_sweep_swing_days:
        return "2-3 Week", "Swing Trade"
    return "Monthly", "Swing Trade"


def _recent_avg_volume(bars: pd.DataFrame, lookback: int = 20) -> float:
    """Mean volume of the last *lookback* bars (excluding current)."""
    if "volume" not in bars.columns or len(bars) < 2:
        return 0.0
    # Exclude the current (last) bar so the average represents "prior" volume.
    return float(bars["volume"].iloc[-lookback - 1:-1].mean())


def _volume_spike_mult(bars: pd.DataFrame, lookback: int = 20) -> float:
    """Current-bar volume as a multiple of the recent average."""
    avg = _recent_avg_volume(bars, lookback)
    if avg <= 0:
        return 0.0
    return round(float(bars["volume"].iloc[-1]) / avg, 2)


def _intraday_trend(bars: pd.DataFrame) -> str:
    """Return ``"up"``, ``"down"``, or ``"flat"`` based on simple EMA slope."""
    if bars is None or len(bars) < 10 or "close" not in bars.columns:
        return "flat"
    closes = bars["close"]
    ema_fast = float(closes.ewm(span=5, adjust=False).mean().iloc[-1])
    ema_slow = float(closes.ewm(span=20, adjust=False).mean().iloc[-1])
    if ema_fast > ema_slow * 1.001:
        return "up"
    if ema_fast < ema_slow * 0.999:
        return "down"
    return "flat"


def _nearest_sd_zone(
    bars: pd.DataFrame,
    current_price: float,
    direction: str,
    lookback: int = 30,
    tolerance_pct: float = 0.015,
) -> Optional[str]:
    """Find the nearest demand (for bullish) or supply (for bearish) zone.

    Uses a simple proxy: clusters of lows (demand) or highs (supply) in the
    recent *lookback* bars that are within *tolerance_pct* of current price.

    Returns a human-readable zone description string, or ``None``.
    """
    if bars is None or bars.empty:
        return None

    df = bars.tail(lookback)
    price_series = df["low"] if direction == "Bullish" else df["high"]
    threshold = current_price * tolerance_pct

    # Collect levels within tolerance of current price
    nearby = price_series[abs(price_series - current_price) <= threshold]
    if len(nearby) < 2:
        return None

    zone_low = round(float(nearby.min()), 2)
    zone_high = round(float(nearby.max()), 2)
    label = "Demand Zone" if direction == "Bullish" else "Supply Zone"
    return f"{label} {zone_low:.2f}–{zone_high:.2f}"


def _select_best_sweep(sweeps: List[Dict], sweep_direction: str) -> Optional[Dict]:
    """Return the highest-quality sweep from a list.

    Selection priority:
    1. Highest volume (most aggressive flow).
    2. Minimum contract threshold.

    Args:
        sweeps:          List of sweep dicts (from ``options_analyzer``).
        sweep_direction: ``"Bullish"`` or ``"Bearish"``.

    Returns:
        Best sweep dict or ``None`` if none qualify.
    """
    min_contracts = config.signals.golden_sweep_min_contracts
    qualified = [s for s in sweeps if int(s.get("volume", 0)) >= min_contracts]
    if not qualified:
        return None
    return max(qualified, key=lambda s: s.get("volume", 0))


# ---------------------------------------------------------------------------
# Main public function
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
    options_flow_data: Optional[Dict],
    bars: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
) -> Optional[GoldenSweepSetup]:
    """Detect an institutional Golden Sweep on *ticker*.

    The function analyses rich options flow data (as returned by
    :func:`~momentum_radar.options.options_analyzer.get_options_flow`) to find
    unusually large, conviction-driven sweeps.

    Detection steps
    ---------------
    1. Extract ``call_sweeps`` / ``put_sweeps`` from *options_flow_data*.
    2. Filter to sweeps with volume ≥ ``golden_sweep_min_contracts``.
    3. Select the highest-volume sweep as the primary signal.
    4. Check underlying RVOL ≥ ``golden_sweep_rvol_min``.
    5. Check volume spike ≥ ``golden_sweep_volume_spike_min``.
    6. Verify intraday trend aligns with sweep direction.
    7. Calculate entry / stop / target using ATR-based risk.
    8. Accept only if risk:reward ≥ 1.5.

    Args:
        ticker:            Stock symbol.
        options_flow_data: Dict from
            :func:`~momentum_radar.options.options_analyzer.get_options_flow`;
            must contain ``call_sweeps`` and ``put_sweeps`` keys.
        bars:              Intraday OHLCV DataFrame (1-min preferred).
        daily:             Daily OHLCV DataFrame (30+ days).

    Returns:
        :class:`GoldenSweepSetup` if a qualifying sweep is found, else ``None``.
    """
    cfg = config.signals

    if not options_flow_data:
        return None

    call_sweeps: List[Dict] = options_flow_data.get("call_sweeps", []) or []
    put_sweeps: List[Dict] = options_flow_data.get("put_sweeps", []) or []

    if not call_sweeps and not put_sweeps:
        return None

    # Evaluate both directions and pick the better one
    best_call = _select_best_sweep(call_sweeps, "Bullish")
    best_put = _select_best_sweep(put_sweeps, "Bearish")

    if best_call is None and best_put is None:
        return None

    # Choose the side with greater volume
    if best_call is None:
        chosen_sweep = best_put
        direction = "Bearish"
        contract_type = "Put"
    elif best_put is None:
        chosen_sweep = best_call
        direction = "Bullish"
        contract_type = "Call"
    else:
        if int(best_call.get("volume", 0)) >= int(best_put.get("volume", 0)):
            chosen_sweep = best_call
            direction = "Bullish"
            contract_type = "Call"
        else:
            chosen_sweep = best_put
            direction = "Bearish"
            contract_type = "Put"

    # Underlying checks
    current_price = float(options_flow_data.get("current_price", 0.0))
    if bars is not None and not bars.empty and "close" in bars.columns:
        current_price = float(bars["close"].iloc[-1])

    if current_price <= 0:
        return None

    rvol = compute_rvol(bars, daily) if (bars is not None and daily is not None) else 0.0
    vol_spike = _volume_spike_mult(bars) if bars is not None else 0.0

    if rvol < cfg.golden_sweep_rvol_min:
        logger.debug(
            "%s – Golden Sweep skipped: RVOL %.1f < %.1f",
            ticker,
            rvol,
            cfg.golden_sweep_rvol_min,
        )
        return None

    if vol_spike < cfg.golden_sweep_volume_spike_min:
        logger.debug(
            "%s – Golden Sweep skipped: vol spike %.1f < %.1f",
            ticker,
            vol_spike,
            cfg.golden_sweep_volume_spike_min,
        )
        return None

    # Trend alignment
    trend = _intraday_trend(bars)
    if direction == "Bullish" and trend == "down":
        logger.debug("%s – Golden Sweep (bullish) skipped: intraday trend is down", ticker)
        return None
    if direction == "Bearish" and trend == "up":
        logger.debug("%s – Golden Sweep (bearish) skipped: intraday trend is up", ticker)
        return None

    # ATR-based entry / stop / target
    atr = compute_atr(daily) if daily is not None and not daily.empty else None
    if atr is None or atr <= 0:
        # Fallback: 0.5 % of current price
        atr = current_price * 0.005

    entry = current_price
    if direction == "Bullish":
        stop = round(entry - STOP_ATR_MULT * atr, 2)
        target = round(entry + TARGET_RR_MULT * abs(entry - stop), 2)
    else:
        stop = round(entry + STOP_ATR_MULT * atr, 2)
        target = round(entry - TARGET_RR_MULT * abs(stop - entry), 2)

    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk <= 0:
        logger.debug("%s – Golden Sweep skipped: zero risk (entry == stop)", ticker)
        return None
    if reward / risk < MIN_RISK_REWARD:
        logger.debug("%s – Golden Sweep skipped: R:R %.1f < %.1f", ticker, reward / risk, MIN_RISK_REWARD)
        return None

    # Sweep classification
    expiry_str = chosen_sweep.get("expiry", "")
    days_left = _days_to_expiry(expiry_str)
    sweep_type, trade_type = _classify_sweep_type(days_left)

    contracts = int(chosen_sweep.get("volume", 0))
    strike = float(chosen_sweep.get("strike", 0.0))

    # Nearest supply/demand zone
    sd_zone = _nearest_sd_zone(bars, current_price, direction)

    confidence = "High" if rvol >= 2.0 and vol_spike >= 4.0 else "Medium"

    details = (
        f"{sweep_type} {contract_type} Sweep; {contracts:,} contracts; "
        f"strike ${strike:.0f}; expiry {expiry_str}; RVOL {rvol:.1f}; "
        f"vol spike {vol_spike:.1f}x"
    )

    logger.info(
        "Golden Sweep detected: %s %s %s → %s (%s) "
        "contracts=%d strike=%.0f entry=%.2f stop=%.2f target=%.2f RR=%.1f",
        ticker,
        direction,
        sweep_type,
        trade_type,
        contract_type,
        contracts,
        strike,
        entry,
        stop,
        target,
        reward / risk,
    )

    return GoldenSweepSetup(
        ticker=ticker,
        direction=direction,
        sweep_type=sweep_type,
        trade_type=trade_type,
        contract_type=contract_type,
        strike=strike,
        expiry=expiry_str,
        contracts=contracts,
        underlying_price=current_price,
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
        volume_spike=round(vol_spike, 1),
        supply_demand_zone=sd_zone,
        confidence=confidence,
        timestamp=datetime.now(),
        details=details,
    )


# ---------------------------------------------------------------------------
# Scoring-registry wrapper
# ---------------------------------------------------------------------------

@register_signal("golden_sweep")
def golden_sweep_signal(
    ticker: str,
    options: Optional[Dict] = None,
    bars: Optional[pd.DataFrame] = None,
    daily: Optional[pd.DataFrame] = None,
    **kwargs,
) -> SignalResult:
    """Scoring-registry signal wrapper for the Golden Sweep detector.

    Returns a :class:`~momentum_radar.signals.base.SignalResult` compatible
    with the scoring engine.  Score is 3 when a high-confidence sweep is
    detected, 2 for medium-confidence.

    Args:
        ticker:  Stock symbol.
        options: Options flow dict (must contain ``call_sweeps``/``put_sweeps``).
        bars:    Intraday OHLCV DataFrame.
        daily:   Daily OHLCV DataFrame.

    Returns:
        :class:`~momentum_radar.signals.base.SignalResult`
    """
    setup = detect_golden_sweep(ticker, options, bars, daily)

    if setup is None:
        return SignalResult(
            triggered=False,
            score=0,
            details="No golden sweep detected",
        )

    score = 3 if setup.confidence == "High" else 2
    return SignalResult(
        triggered=True,
        score=score,
        details=(
            f"{setup.sweep_type} {setup.contract_type} Sweep → {setup.direction} {setup.trade_type} "
            f"({setup.contracts:,} contracts, strike ${setup.strike:.0f})"
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
