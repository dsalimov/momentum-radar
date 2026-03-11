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
    underlying_price: float
    entry: float
    stop: float
    target: float
    rvol: float
    volume_spike: float
    supply_demand_zone: Optional[str] = None
    confidence: str = "High"
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
    return float(bars["volume"].iloc[-lookback - 1: -1].mean())


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
    if risk <= 0 or reward / risk < MIN_RISK_REWARD:
        logger.debug("%s – Golden Sweep skipped: R:R %.1f < %.1f", ticker, reward / risk if risk > 0 else 0, MIN_RISK_REWARD)
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
        ),
    )
