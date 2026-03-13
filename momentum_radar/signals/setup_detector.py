"""
setup_detector.py – Professional trade setup detection.

Implements six high-quality intraday setups modelled on institutional
trading workflows.  Before any setup is evaluated the ticker must pass a
**5-step trader decision model**:

1. Liquidity check   – avg daily volume and RVOL requirements
2. Momentum check    – intraday range vs ATR, directional movement
3. Market structure  – VWAP position, key levels identified
4. Setup detection   – one of six named setups must be present
5. Risk/Reward check – minimum 1.5 R:R required

Setups (in priority order):

1. Liquidity Sweep        – stop-hunt reversal
2. Opening Range Breakout – institutional ORB entry
3. VWAP Reclaim           – long on VWAP cross-above with volume
4. VWAP Breakdown         – short on VWAP cross-below with volume
5. Support Bounce         – long at multi-touch support
6. Momentum Ignition      – aggressive directional expansion

Usage::

    from momentum_radar.signals.setup_detector import detect_setups, TradeSetup

    setups = detect_setups(ticker, bars, daily)
    for setup in setups:
        print(setup.setup_type.value, setup.entry, setup.stop, setup.target)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd

from momentum_radar.utils.indicators import compute_atr, compute_vwap, compute_rvol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

#: Minimum RVOL for a ticker to be tradeable (Step 1 filter).
MIN_RVOL: float = 1.5
#: Minimum average daily volume (Step 1 filter).
MIN_AVG_DAILY_VOLUME: int = 1_000_000
#: Minimum intraday volume spike multiplier vs. recent candle average.
MIN_VOLUME_SPIKE_MULT: float = 1.5
#: Volume spike multiplier required for the stronger setups.
STRONG_VOLUME_SPIKE_MULT: float = 3.0
#: Minimum risk:reward ratio before a setup is reported.
MIN_RISK_REWARD: float = 1.5
#: Buffer applied to stop-loss levels (as % of entry price).
STOP_BUFFER_PCT: float = 0.003
#: Number of recent bars used to detect support/resistance clusters.
SR_LOOKBACK_BARS: int = 30
#: Minimum number of touches to count as a valid support/resistance level.
MIN_SR_TOUCHES: int = 2
#: Price proximity threshold for counting support/resistance touches (% of price).
SR_PROXIMITY_PCT: float = 0.005
#: Number of consecutive directional candles needed for momentum ignition.
MOMENTUM_IGNITION_CANDLES: int = 3


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class SetupType(Enum):
    """Named setup categories in descending priority."""

    GOLDEN_SWEEP = "Golden Sweep"
    LIQUIDITY_SWEEP = "Liquidity Sweep"
    OPENING_RANGE_BREAKOUT = "Opening Range Breakout"
    VWAP_RECLAIM = "VWAP Reclaim"
    VWAP_BREAKDOWN = "VWAP Breakdown"
    SUPPORT_BOUNCE = "Support Bounce"
    MOMENTUM_IGNITION = "Momentum Ignition"
    CHART_PATTERN_BREAKOUT = "Chart Pattern Breakout"


class SetupDirection(Enum):
    """Trade direction."""

    LONG = "Long"
    SHORT = "Short"


class StrategyType(Enum):
    """High-level trading strategy classification for user-facing alerts.

    Every :class:`TradeSetup` is automatically classified into one of these
    three categories so that the alert header clearly communicates the expected
    time horizon and trading style.
    """

    SCALP_TRADE = "SCALP TRADE"
    DAY_TRADE = "DAY TRADE"
    SWING_TRADE = "SWING TRADE"


#: Mapping from setup type to the appropriate strategy classification.
_SETUP_STRATEGY: Dict[SetupType, StrategyType] = {
    SetupType.LIQUIDITY_SWEEP:        StrategyType.SCALP_TRADE,
    SetupType.MOMENTUM_IGNITION:      StrategyType.SCALP_TRADE,
    SetupType.VWAP_RECLAIM:           StrategyType.DAY_TRADE,
    SetupType.VWAP_BREAKDOWN:         StrategyType.DAY_TRADE,
    SetupType.OPENING_RANGE_BREAKOUT: StrategyType.DAY_TRADE,
    SetupType.GOLDEN_SWEEP:           StrategyType.DAY_TRADE,
    SetupType.SUPPORT_BOUNCE:         StrategyType.DAY_TRADE,
    SetupType.CHART_PATTERN_BREAKOUT: StrategyType.SWING_TRADE,
}


@dataclass
class TradeSetup:
    """A single, fully-defined trade setup with entry, stop and target.

    Attributes:
        ticker:        Stock symbol.
        setup_type:    One of the :class:`SetupType` values.
        direction:     ``LONG`` or ``SHORT``.
        entry:         Suggested entry price.
        stop:          Stop-loss price.
        target:        Profit-target price.
        rvol:          Relative volume at signal time.
        volume_spike:  Current-bar volume vs. recent average (multiplier).
        confidence:    Qualitative confidence grade (``"High"`` / ``"Medium"``).
        timestamp:     When the setup was detected.
        details:       Human-readable description of triggering conditions.
        strategy_type: Auto-computed trading strategy classification.
    """

    ticker: str
    setup_type: SetupType
    direction: SetupDirection
    entry: float
    stop: float
    target: float
    rvol: float
    volume_spike: float
    confidence: str
    timestamp: datetime
    details: str = ""
    target2: float = 0.0
    strategy_type: Optional[StrategyType] = field(default=None)

    def __post_init__(self) -> None:
        if self.strategy_type is None:
            self.strategy_type = _SETUP_STRATEGY.get(
                self.setup_type, StrategyType.DAY_TRADE
            )

    @property
    def risk_reward(self) -> float:
        """Risk:reward ratio (reward / risk).  Returns 0 if risk is zero."""
        risk = abs(self.entry - self.stop)
        reward = abs(self.target - self.entry)
        if risk <= 0:
            return 0.0
        return round(reward / risk, 1)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _recent_avg_volume(bars: pd.DataFrame, lookback: int = 20) -> float:
    """Compute the mean volume of the last *lookback* bars (excluding current)."""
    if "volume" not in bars.columns or len(bars) < 2:
        return 0.0
    return float(bars["volume"].iloc[-lookback - 1 : -1].mean())


def _volume_spike_mult(bars: pd.DataFrame, lookback: int = 20) -> float:
    """Current-bar volume as a multiple of the recent average volume."""
    avg = _recent_avg_volume(bars, lookback)
    if avg <= 0:
        return 0.0
    return round(float(bars["volume"].iloc[-1]) / avg, 2)


def _find_support_levels(bars: pd.DataFrame, lookback: int = SR_LOOKBACK_BARS) -> List[float]:
    """Return a list of intraday support price levels.

    A support level is defined as a local low that was visited by price
    (within ``SR_PROXIMITY_PCT``) at least ``MIN_SR_TOUCHES`` times.

    Args:
        bars:    Intraday OHLCV DataFrame.
        lookback: Number of bars to consider.

    Returns:
        List of support price levels (sorted ascending).
    """
    df = bars.tail(lookback)
    if df.empty or "low" not in df.columns:
        return []

    candidate_lows = df["low"].values
    supports: List[float] = []
    used: List[bool] = [False] * len(candidate_lows)

    for i, low in enumerate(candidate_lows):
        if used[i]:
            continue
        threshold = low * SR_PROXIMITY_PCT
        touches = sum(
            1 for j, other in enumerate(candidate_lows)
            if not used[j] and abs(other - low) <= threshold
        )
        if touches >= MIN_SR_TOUCHES:
            level = float(
                sum(
                    other for j, other in enumerate(candidate_lows)
                    if abs(other - low) <= threshold
                )
                / touches
            )
            supports.append(level)
            for j, other in enumerate(candidate_lows):
                if abs(other - low) <= threshold:
                    used[j] = True

    return sorted(set(round(s, 2) for s in supports))


def _find_resistance_levels(bars: pd.DataFrame, lookback: int = SR_LOOKBACK_BARS) -> List[float]:
    """Return a list of intraday resistance price levels.

    Mirrors :func:`_find_support_levels` using bar highs.

    Args:
        bars:    Intraday OHLCV DataFrame.
        lookback: Number of bars to consider.

    Returns:
        List of resistance price levels (sorted ascending).
    """
    df = bars.tail(lookback)
    if df.empty or "high" not in df.columns:
        return []

    candidate_highs = df["high"].values
    resistances: List[float] = []
    used: List[bool] = [False] * len(candidate_highs)

    for i, high in enumerate(candidate_highs):
        if used[i]:
            continue
        threshold = high * SR_PROXIMITY_PCT
        touches = sum(
            1 for j, other in enumerate(candidate_highs)
            if not used[j] and abs(other - high) <= threshold
        )
        if touches >= MIN_SR_TOUCHES:
            level = float(
                sum(
                    other for j, other in enumerate(candidate_highs)
                    if abs(other - high) <= threshold
                )
                / touches
            )
            resistances.append(level)
            for j, other in enumerate(candidate_highs):
                if abs(other - high) <= threshold:
                    used[j] = True

    return sorted(set(round(r, 2) for r in resistances))


# ---------------------------------------------------------------------------
# Step 1 – Liquidity filter
# ---------------------------------------------------------------------------

def _passes_liquidity_check(bars: pd.DataFrame, daily: pd.DataFrame) -> bool:
    """Return True if the ticker meets minimum liquidity requirements.

    Checks:
    * Average daily volume > :data:`MIN_AVG_DAILY_VOLUME`
    * RVOL > :data:`MIN_RVOL`

    Args:
        bars:  Intraday OHLCV DataFrame.
        daily: Daily OHLCV DataFrame.

    Returns:
        ``True`` if liquidity requirements are met.
    """
    if daily is None or daily.empty:
        return False

    # Average daily volume check
    if "volume" in daily.columns:
        avg_vol = float(daily["volume"].tail(30).mean())
        if avg_vol < MIN_AVG_DAILY_VOLUME:
            return False

    # RVOL check
    rvol = compute_rvol(bars, daily)
    if rvol is None or rvol < MIN_RVOL:
        return False

    return True


# ---------------------------------------------------------------------------
# Step 2 – Momentum filter
# ---------------------------------------------------------------------------

def _passes_momentum_check(bars: pd.DataFrame, daily: pd.DataFrame) -> bool:
    """Return True if the stock shows meaningful intraday momentum.

    Checks:
    * Intraday range > 1 ATR
    * Clear directional movement (not sideways chop)

    Args:
        bars:  Intraday OHLCV DataFrame.
        daily: Daily OHLCV DataFrame.

    Returns:
        ``True`` if momentum requirements are met.
    """
    if bars is None or bars.empty:
        return False

    atr = compute_atr(daily) if daily is not None else None
    if atr is None or atr <= 0:
        return True  # Cannot check — allow through

    if "high" not in bars.columns or "low" not in bars.columns:
        return False

    day_range = float(bars["high"].max()) - float(bars["low"].min())
    if day_range < atr:
        return False

    # Require at least some directional movement (close vs. open span)
    if "close" in bars.columns and "open" in bars.columns and len(bars) >= 5:
        first_open = float(bars["open"].iloc[0])
        last_close = float(bars["close"].iloc[-1])
        move_pct = abs(last_close - first_open) / first_open if first_open > 0 else 0.0
        if move_pct < 0.002:  # less than 0.2% net move = choppy
            return False

    return True


# ---------------------------------------------------------------------------
# Individual setup detectors
# ---------------------------------------------------------------------------

def _detect_vwap_reclaim(
    ticker: str,
    bars: pd.DataFrame,
    daily: pd.DataFrame,
    rvol: float,
    vwap: float,
) -> Optional[TradeSetup]:
    """Detect a VWAP Reclaim (long) setup.

    Conditions:
    * Previous candle closed below VWAP
    * Current candle closes above VWAP
    * Volume spike present (:data:`MIN_VOLUME_SPIKE_MULT`)
    * RR >= :data:`MIN_RISK_REWARD`

    Args:
        ticker: Stock symbol.
        bars:   Intraday OHLCV DataFrame.
        daily:  Daily OHLCV DataFrame.
        rvol:   Pre-computed RVOL.
        vwap:   Current VWAP value.

    Returns:
        :class:`TradeSetup` or ``None``.
    """
    if len(bars) < 2:
        return None

    prev_close = float(bars["close"].iloc[-2])
    curr_close = float(bars["close"].iloc[-1])

    if not (prev_close < vwap < curr_close):
        return None

    vol_mult = _volume_spike_mult(bars)
    if vol_mult < MIN_VOLUME_SPIKE_MULT:
        return None

    entry = curr_close
    stop = round(vwap * (1 - STOP_BUFFER_PCT), 2)

    # Target: nearest resistance above entry
    resistances = [r for r in _find_resistance_levels(bars) if r > entry]
    if resistances:
        target = min(resistances)
    else:
        atr = compute_atr(daily) or (entry * 0.01)
        target = round(entry + 1.5 * atr, 2)

    if abs(entry - stop) <= 0 or abs(target - entry) / abs(entry - stop) < MIN_RISK_REWARD:
        return None

    return TradeSetup(
        ticker=ticker,
        setup_type=SetupType.VWAP_RECLAIM,
        direction=SetupDirection.LONG,
        entry=round(entry, 2),
        stop=round(stop, 2),
        target=round(target, 2),
        rvol=round(rvol, 1),
        volume_spike=vol_mult,
        confidence="High" if vol_mult >= STRONG_VOLUME_SPIKE_MULT else "Medium",
        timestamp=datetime.now(),
        details=f"Prev close {prev_close:.2f} < VWAP {vwap:.2f}; current close {curr_close:.2f} above VWAP; vol {vol_mult:.1f}x avg",
    )


def _detect_vwap_breakdown(
    ticker: str,
    bars: pd.DataFrame,
    daily: pd.DataFrame,
    rvol: float,
    vwap: float,
) -> Optional[TradeSetup]:
    """Detect a VWAP Breakdown (short) setup.

    Conditions:
    * Previous candle closed above VWAP
    * Current candle closes below VWAP
    * Volume spike present (:data:`MIN_VOLUME_SPIKE_MULT`)
    * RR >= :data:`MIN_RISK_REWARD`

    Args:
        ticker: Stock symbol.
        bars:   Intraday OHLCV DataFrame.
        daily:  Daily OHLCV DataFrame.
        rvol:   Pre-computed RVOL.
        vwap:   Current VWAP value.

    Returns:
        :class:`TradeSetup` or ``None``.
    """
    if len(bars) < 2:
        return None

    prev_close = float(bars["close"].iloc[-2])
    curr_close = float(bars["close"].iloc[-1])

    if not (prev_close > vwap > curr_close):
        return None

    vol_mult = _volume_spike_mult(bars)
    if vol_mult < MIN_VOLUME_SPIKE_MULT:
        return None

    entry = curr_close
    stop = round(vwap * (1 + STOP_BUFFER_PCT), 2)

    # Target: nearest support below entry
    supports = [s for s in _find_support_levels(bars) if s < entry]
    if supports:
        target = max(supports)
    else:
        atr = compute_atr(daily) or (entry * 0.01)
        target = round(entry - 1.5 * atr, 2)

    if abs(entry - stop) <= 0 or abs(target - entry) / abs(entry - stop) < MIN_RISK_REWARD:
        return None

    return TradeSetup(
        ticker=ticker,
        setup_type=SetupType.VWAP_BREAKDOWN,
        direction=SetupDirection.SHORT,
        entry=round(entry, 2),
        stop=round(stop, 2),
        target=round(target, 2),
        rvol=round(rvol, 1),
        volume_spike=vol_mult,
        confidence="High" if vol_mult >= STRONG_VOLUME_SPIKE_MULT else "Medium",
        timestamp=datetime.now(),
        details=f"Prev close {prev_close:.2f} > VWAP {vwap:.2f}; current close {curr_close:.2f} below VWAP; vol {vol_mult:.1f}x avg",
    )


def _detect_support_bounce(
    ticker: str,
    bars: pd.DataFrame,
    daily: pd.DataFrame,
    rvol: float,
) -> Optional[TradeSetup]:
    """Detect a Support Bounce (long) setup.

    Conditions:
    * Established support level with ``MIN_SR_TOUCHES`` touches
    * Previous candle low touched the support
    * Current candle closes above the support level
    * Volume increasing
    * RR >= :data:`MIN_RISK_REWARD`

    Args:
        ticker: Stock symbol.
        bars:   Intraday OHLCV DataFrame.
        daily:  Daily OHLCV DataFrame.
        rvol:   Pre-computed RVOL.

    Returns:
        :class:`TradeSetup` or ``None``.
    """
    if len(bars) < 3:
        return None

    supports = _find_support_levels(bars)
    if not supports:
        return None

    curr_close = float(bars["close"].iloc[-1])
    prev_low = float(bars["low"].iloc[-2])

    # Find the nearest support that was just tested
    for support in supports:
        proximity = support * SR_PROXIMITY_PCT * 2
        if abs(prev_low - support) <= proximity and curr_close > support:
            # Volume should be increasing on the bounce candle
            vol_mult = _volume_spike_mult(bars)
            if vol_mult < MIN_VOLUME_SPIKE_MULT:
                continue

            entry = curr_close
            stop = round(support * (1 - STOP_BUFFER_PCT), 2)

            # Target: nearest resistance above entry
            resistances = [r for r in _find_resistance_levels(bars) if r > entry]
            if resistances:
                target = min(resistances)
            else:
                atr = compute_atr(daily) or (entry * 0.01)
                target = round(entry + 2.0 * atr, 2)

            if abs(entry - stop) <= 0 or abs(target - entry) / abs(entry - stop) < MIN_RISK_REWARD:
                continue

            return TradeSetup(
                ticker=ticker,
                setup_type=SetupType.SUPPORT_BOUNCE,
                direction=SetupDirection.LONG,
                entry=round(entry, 2),
                stop=round(stop, 2),
                target=round(target, 2),
                rvol=round(rvol, 1),
                volume_spike=vol_mult,
                confidence="High" if vol_mult >= STRONG_VOLUME_SPIKE_MULT else "Medium",
                timestamp=datetime.now(),
                details=f"Support at {support:.2f} tested; bounce close {curr_close:.2f}; vol {vol_mult:.1f}x avg",
            )

    return None


def _detect_liquidity_sweep(
    ticker: str,
    bars: pd.DataFrame,
    daily: pd.DataFrame,
    rvol: float,
) -> Optional[TradeSetup]:
    """Detect a Liquidity Sweep (stop-hunt reversal) setup.

    Conditions (bullish — sweep below support then reverse):
    * Previous candle closes below recent support (the sweep)
    * Current candle closes above the swept support level
    * Volume spike during the sweep candle
    * RR >= :data:`MIN_RISK_REWARD`

    Conditions (bearish — sweep above resistance then reverse):
    * Previous candle closes above recent resistance (the sweep)
    * Current candle closes below the swept resistance level
    * Volume spike during the sweep candle

    Args:
        ticker: Stock symbol.
        bars:   Intraday OHLCV DataFrame.
        daily:  Daily OHLCV DataFrame.
        rvol:   Pre-computed RVOL.

    Returns:
        :class:`TradeSetup` or ``None``.
    """
    if len(bars) < 3:
        return None

    prev_close = float(bars["close"].iloc[-2])
    curr_close = float(bars["close"].iloc[-1])
    prev_low = float(bars["low"].iloc[-2])
    prev_high = float(bars["high"].iloc[-2])

    vol_mult = _volume_spike_mult(bars)
    if vol_mult < MIN_VOLUME_SPIKE_MULT:
        return None

    # --- Bullish sweep: price dipped below support then reclaimed it ---
    supports = _find_support_levels(bars)
    for support in supports:
        # Sweep condition: wick broke below but close reclaimed
        if prev_low < support and curr_close > support:
            entry = curr_close
            stop = round(prev_low * (1 - STOP_BUFFER_PCT), 2)

            resistances = [r for r in _find_resistance_levels(bars) if r > entry]
            if resistances:
                target = min(resistances)
            else:
                atr = compute_atr(daily) or (entry * 0.01)
                target = round(entry + 2.0 * atr, 2)

            if abs(entry - stop) <= 0 or abs(target - entry) / abs(entry - stop) < MIN_RISK_REWARD:
                continue

            return TradeSetup(
                ticker=ticker,
                setup_type=SetupType.LIQUIDITY_SWEEP,
                direction=SetupDirection.LONG,
                entry=round(entry, 2),
                stop=round(stop, 2),
                target=round(target, 2),
                rvol=round(rvol, 1),
                volume_spike=vol_mult,
                confidence="High",
                timestamp=datetime.now(),
                details=f"Support sweep at {support:.2f}; wick low {prev_low:.2f}; reclaim close {curr_close:.2f}; vol {vol_mult:.1f}x avg",
            )

    # --- Bearish sweep: price spiked above resistance then failed ---
    resistances = _find_resistance_levels(bars)
    for resistance in resistances:
        if prev_high > resistance and curr_close < resistance:
            entry = curr_close
            stop = round(prev_high * (1 + STOP_BUFFER_PCT), 2)

            local_supports = [s for s in supports if s < entry]
            if local_supports:
                target = max(local_supports)
            else:
                atr = compute_atr(daily) or (entry * 0.01)
                target = round(entry - 2.0 * atr, 2)

            if abs(entry - stop) <= 0 or abs(target - entry) / abs(entry - stop) < MIN_RISK_REWARD:
                continue

            return TradeSetup(
                ticker=ticker,
                setup_type=SetupType.LIQUIDITY_SWEEP,
                direction=SetupDirection.SHORT,
                entry=round(entry, 2),
                stop=round(stop, 2),
                target=round(target, 2),
                rvol=round(rvol, 1),
                volume_spike=vol_mult,
                confidence="High",
                timestamp=datetime.now(),
                details=f"Resistance sweep at {resistance:.2f}; wick high {prev_high:.2f}; rejection close {curr_close:.2f}; vol {vol_mult:.1f}x avg",
            )

    return None


def _detect_orb(
    ticker: str,
    bars: pd.DataFrame,
    daily: pd.DataFrame,
    rvol: float,
) -> Optional[TradeSetup]:
    """Detect an Opening Range Breakout (ORB) setup.

    Uses the first 15-minute bar high/low as the opening range.  A breakout
    is confirmed when the current bar closes beyond the range with a volume
    spike and the time is after 09:45.

    Args:
        ticker: Stock symbol.
        bars:   1-minute (or active timeframe) intraday OHLCV DataFrame with
                a DatetimeIndex.
        daily:  Daily OHLCV DataFrame.
        rvol:   Pre-computed RVOL.

    Returns:
        :class:`TradeSetup` or ``None``.
    """
    if bars is None or bars.empty:
        return None

    try:
        first_15 = bars.between_time("09:30", "09:44")
    except Exception:
        return None

    if first_15.empty:
        return None

    or_high = float(first_15["high"].max())
    or_low = float(first_15["low"].min())
    or_range = or_high - or_low

    curr_close = float(bars["close"].iloc[-1])
    vol_mult = _volume_spike_mult(bars)

    if vol_mult < MIN_VOLUME_SPIKE_MULT:
        return None

    # Long ORB breakout
    if curr_close > or_high:
        entry = curr_close
        stop = round(or_high * (1 - STOP_BUFFER_PCT), 2)
        target = round(entry + or_range * 1.5, 2)

        if abs(entry - stop) <= 0 or abs(target - entry) / abs(entry - stop) < MIN_RISK_REWARD:
            return None

        return TradeSetup(
            ticker=ticker,
            setup_type=SetupType.OPENING_RANGE_BREAKOUT,
            direction=SetupDirection.LONG,
            entry=round(entry, 2),
            stop=round(stop, 2),
            target=round(target, 2),
            rvol=round(rvol, 1),
            volume_spike=vol_mult,
            confidence="High" if vol_mult >= STRONG_VOLUME_SPIKE_MULT else "Medium",
            timestamp=datetime.now(),
            details=f"ORB high {or_high:.2f} broken; close {curr_close:.2f}; range {or_range:.2f}; vol {vol_mult:.1f}x avg",
        )

    # Short ORB breakdown
    if curr_close < or_low:
        entry = curr_close
        stop = round(or_low * (1 + STOP_BUFFER_PCT), 2)
        target = round(entry - or_range * 1.5, 2)

        if abs(entry - stop) <= 0 or abs(target - entry) / abs(entry - stop) < MIN_RISK_REWARD:
            return None

        return TradeSetup(
            ticker=ticker,
            setup_type=SetupType.OPENING_RANGE_BREAKOUT,
            direction=SetupDirection.SHORT,
            entry=round(entry, 2),
            stop=round(stop, 2),
            target=round(target, 2),
            rvol=round(rvol, 1),
            volume_spike=vol_mult,
            confidence="High" if vol_mult >= STRONG_VOLUME_SPIKE_MULT else "Medium",
            timestamp=datetime.now(),
            details=f"ORB low {or_low:.2f} broken; close {curr_close:.2f}; range {or_range:.2f}; vol {vol_mult:.1f}x avg",
        )

    return None


def _detect_momentum_ignition(
    ticker: str,
    bars: pd.DataFrame,
    daily: pd.DataFrame,
    rvol: float,
    vwap: float,
) -> Optional[TradeSetup]:
    """Detect a Momentum Ignition (explosive directional expansion) setup.

    Conditions:
    * At least :data:`MOMENTUM_IGNITION_CANDLES` consecutive candles in the
      same direction
    * Volume expanding across the run (each bar ≥ previous)
    * Price moving away from VWAP (not consolidating at VWAP)
    * RVOL >= 2.0

    Args:
        ticker: Stock symbol.
        bars:   Intraday OHLCV DataFrame.
        daily:  Daily OHLCV DataFrame.
        rvol:   Pre-computed RVOL.
        vwap:   Current VWAP value.

    Returns:
        :class:`TradeSetup` or ``None``.
    """
    n = MOMENTUM_IGNITION_CANDLES
    if len(bars) < n + 1:
        return None

    if rvol < 2.0:
        return None

    recent = bars.tail(n)
    closes = recent["close"].values.astype(float)
    volumes = recent["volume"].values.astype(float) if "volume" in recent.columns else None

    # Check consecutive direction
    bullish = all(closes[i] > closes[i - 1] for i in range(1, n))
    bearish = all(closes[i] < closes[i - 1] for i in range(1, n))

    if not bullish and not bearish:
        return None

    # Require expanding volume
    if volumes is not None and len(volumes) >= n:
        vol_expanding = all(volumes[i] >= volumes[i - 1] * 0.9 for i in range(1, n))
        if not vol_expanding:
            return None

    curr_close = closes[-1]
    vwap_distance_pct = abs(curr_close - vwap) / vwap if vwap > 0 else 0.0
    if vwap_distance_pct < 0.002:  # price hugging VWAP – not really expanding
        return None

    vol_mult = _volume_spike_mult(bars)

    if bullish:
        entry = curr_close
        momentum_base = float(recent["low"].min())
        stop = round(momentum_base * (1 - STOP_BUFFER_PCT), 2)

        resistances = [r for r in _find_resistance_levels(bars) if r > entry]
        if resistances:
            target = min(resistances)
        else:
            atr = compute_atr(daily) or (entry * 0.01)
            target = round(entry + 2.0 * atr, 2)

        if abs(entry - stop) <= 0 or abs(target - entry) / abs(entry - stop) < MIN_RISK_REWARD:
            return None

        return TradeSetup(
            ticker=ticker,
            setup_type=SetupType.MOMENTUM_IGNITION,
            direction=SetupDirection.LONG,
            entry=round(entry, 2),
            stop=round(stop, 2),
            target=round(target, 2),
            rvol=round(rvol, 1),
            volume_spike=vol_mult,
            confidence="High" if rvol >= 3.0 else "Medium",
            timestamp=datetime.now(),
            details=f"{n} consecutive bullish candles; RVOL {rvol:.1f}; close {curr_close:.2f} vs VWAP {vwap:.2f}",
        )

    if bearish:
        entry = curr_close
        momentum_base = float(recent["high"].max())
        stop = round(momentum_base * (1 + STOP_BUFFER_PCT), 2)

        supports = [s for s in _find_support_levels(bars) if s < entry]
        if supports:
            target = max(supports)
        else:
            atr = compute_atr(daily) or (entry * 0.01)
            target = round(entry - 2.0 * atr, 2)

        if abs(entry - stop) <= 0 or abs(target - entry) / abs(entry - stop) < MIN_RISK_REWARD:
            return None

        return TradeSetup(
            ticker=ticker,
            setup_type=SetupType.MOMENTUM_IGNITION,
            direction=SetupDirection.SHORT,
            entry=round(entry, 2),
            stop=round(stop, 2),
            target=round(target, 2),
            rvol=round(rvol, 1),
            volume_spike=vol_mult,
            confidence="High" if rvol >= 3.0 else "Medium",
            timestamp=datetime.now(),
            details=f"{n} consecutive bearish candles; RVOL {rvol:.1f}; close {curr_close:.2f} vs VWAP {vwap:.2f}",
        )

    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# Signal detectors ordered by priority (highest first)
_SETUP_PRIORITY = [
    ("liquidity_sweep", _detect_liquidity_sweep),
    ("orb", _detect_orb),
    ("vwap_reclaim", _detect_vwap_reclaim),
    ("vwap_breakdown", _detect_vwap_breakdown),
    ("support_bounce", _detect_support_bounce),
    ("momentum_ignition", _detect_momentum_ignition),
]


def detect_setups(
    ticker: str,
    bars: pd.DataFrame,
    daily: Optional[pd.DataFrame] = None,
) -> List[TradeSetup]:
    """Run all setup detectors and return qualifying trade setups.

    The ticker must first pass the 5-step trader decision model:

    1. Liquidity  – RVOL >= 1.5, avg daily volume >= 1 M
    2. Momentum   – intraday range >= 1 ATR, directional movement present
    3. Structure  – VWAP computed successfully
    4. Setups     – each of the six detectors is evaluated
    5. Risk/Reward – each candidate must achieve ≥ 1.5 R:R

    Setups are returned in priority order (highest-quality first).
    Only one setup per :class:`SetupType` is returned.

    Args:
        ticker: Stock symbol.
        bars:   Intraday OHLCV DataFrame (1-min or active timeframe).
        daily:  Daily OHLCV DataFrame (optional; improves ATR accuracy).

    Returns:
        List of :class:`TradeSetup` objects (may be empty).
    """
    if bars is None or bars.empty:
        return []

    # Step 1 – Liquidity
    if daily is not None and not _passes_liquidity_check(bars, daily):
        logger.debug("%s – failed liquidity check; skipping setup detection", ticker)
        return []

    # Step 2 – Momentum
    if daily is not None and not _passes_momentum_check(bars, daily):
        logger.debug("%s – failed momentum check; skipping setup detection", ticker)
        return []

    # Step 3 – Market structure (compute shared indicators)
    vwap = compute_vwap(bars)
    rvol = compute_rvol(bars, daily) if daily is not None else None

    if vwap is None or vwap <= 0:
        logger.debug("%s – VWAP unavailable; skipping setup detection", ticker)
        return []

    rvol_val = rvol if rvol is not None else 0.0

    # Step 4 & 5 – Setup detection with RR filter (applied inside each detector)
    setups: List[TradeSetup] = []
    seen_types: set = set()

    for name, detector_fn in _SETUP_PRIORITY:
        try:
            if name in ("vwap_reclaim", "vwap_breakdown"):
                result = detector_fn(ticker, bars, daily, rvol_val, vwap)
            elif name == "momentum_ignition":
                result = detector_fn(ticker, bars, daily, rvol_val, vwap)
            else:
                result = detector_fn(ticker, bars, daily, rvol_val)

            if result is not None and result.setup_type not in seen_types:
                setups.append(result)
                seen_types.add(result.setup_type)
                logger.debug(
                    "%s – setup detected: %s (%s) entry=%.2f stop=%.2f target=%.2f RR=%.1f",
                    ticker,
                    result.setup_type.value,
                    result.direction.value,
                    result.entry,
                    result.stop,
                    result.target,
                    result.risk_reward,
                )
        except Exception as exc:
            logger.warning("%s – setup detector '%s' error: %s", ticker, name, exc)

    return setups
