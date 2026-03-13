"""
trade_formatter.py – Professional trade setup alert formatting.

Produces clean, human-readable trade alert strings suitable for console
output, Telegram messages, and Discord embeds.

Alert format example (standard trade setup)::

    🚨 DAY TRADE

    Ticker: MAR
    Setup: VWAP Breakdown
    Direction: Short

    Entry:  325.80
    Stop:   327.20
    Target: 321.50

    Risk/Reward: 1:2.1
    Volume Spike: 4.2x
    RVOL: 1.9

    Time: 12:31 EST
    Confidence: High

Alert format example (Golden Sweep)::

    🚨 DAY TRADE

    Ticker: TSLA
    Setup: Weekly Call Sweep → Bullish Day Trade
    Underlying Price: 740.50
    Strike(s): 750 Call(s)
    Expiration: Weekly
    Estimated Flow: 12,000 contracts

    Entry:  742.00
    Stop:   737.00
    Target: 755.00

    RVOL: 2.3
    Volume Spike: 4.0x avg
    Supply/Demand Alignment: Demand Zone 740.00–742.00

    Time: 12:45 PM EST
    Confidence: High

Usage::

    from momentum_radar.alerts.trade_formatter import (
        format_trade_setup,
        format_golden_sweep_alert,
    )

    msg = format_trade_setup(setup)
    golden_msg = format_golden_sweep_alert(golden_setup)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from momentum_radar.signals.setup_detector import TradeSetup, SetupType

logger = logging.getLogger(__name__)

# Emoji per setup type – highest priority gets most prominent emoji
_SETUP_EMOJI = {
    SetupType.GOLDEN_SWEEP:           "💎",
    SetupType.LIQUIDITY_SWEEP:        "⚡",
    SetupType.OPENING_RANGE_BREAKOUT: "🔔",
    SetupType.VWAP_RECLAIM:           "📈",
    SetupType.VWAP_BREAKDOWN:         "📉",
    SetupType.SUPPORT_BOUNCE:         "🔄",
    SetupType.MOMENTUM_IGNITION:      "🚀",
    SetupType.GOLDEN_SWEEP:           "⚡",
    SetupType.CHART_PATTERN_BREAKOUT: "📐",
}

_CONFIDENCE_EMOJI = {
    "High":   "🔥",
    "Medium": "✅",
    "Low":    "👀",
}


def format_trade_setup(
    setup: TradeSetup,
    timestamp: Optional[datetime] = None,
) -> str:
    """Build a clean, professional trade setup alert string.

    Args:
        setup:     :class:`~momentum_radar.signals.setup_detector.TradeSetup`
                   object produced by the setup detector.
        timestamp: Override for the alert timestamp (defaults to
                   ``setup.timestamp`` or ``datetime.now()``).

    Returns:
        Multi-line formatted alert string.
    """
    ts = timestamp or setup.timestamp or datetime.now()
    setup_emoji = _SETUP_EMOJI.get(setup.setup_type, "🚨")
    conf_emoji = _CONFIDENCE_EMOJI.get(setup.confidence, "✅")
    rr = setup.risk_reward
    strategy_label = setup.strategy_type.value

    lines = [
        f"🚨 {strategy_label}",
        "",
        f"Ticker:    {setup.ticker}",
        f"Setup:     {setup_emoji} {setup.setup_type.value}",
        f"Direction: {setup.direction.value}",
        "",
        f"Entry:     {setup.entry:.2f}",
        f"Stop:      {setup.stop:.2f}",
        f"Target:    {setup.target:.2f}",
        "",
        f"Risk/Reward:  1:{rr:.1f}",
        f"Volume Spike: {setup.volume_spike:.1f}x",
        f"RVOL:         {setup.rvol:.1f}",
        "",
        f"Time:       {ts.strftime('%I:%M %p EST')}",
        f"Confidence: {conf_emoji} {setup.confidence}",
    ]

    if setup.details:
        lines += ["", f"Details: {setup.details}"]

    return "\n".join(lines)


def format_trade_setup_list(
    setups: list,
    timestamp: Optional[datetime] = None,
) -> str:
    """Format multiple setups into a single consolidated alert message.

    Args:
        setups:    List of :class:`~momentum_radar.signals.setup_detector.TradeSetup`
                   objects.
        timestamp: Shared timestamp for all setups.

    Returns:
        Multi-line formatted string, or empty string if *setups* is empty.
    """
    if not setups:
        return ""
    parts = [format_trade_setup(s, timestamp=timestamp) for s in setups]
    separator = "\n" + "─" * 40 + "\n"
    return separator.join(parts)


# ---------------------------------------------------------------------------
# Golden Sweep alert formatter
# ---------------------------------------------------------------------------

def format_golden_sweep_alert(
    setup: "GoldenSweepSetup",
    timestamp: Optional[datetime] = None,
) -> str:
    """Build a professional Golden Sweep alert string.

    The header uses the setup's trade type to classify the signal as a
    ``DAY TRADE`` or ``SWING TRADE`` per the strategy classification standard::

        🚨 DAY TRADE

        Ticker: TSLA
        Setup: Weekly Call Sweep → Bullish Day Trade
        Underlying Price: 740.50
        Strike(s): 750 Call(s)
        Expiration: Weekly
        Estimated Flow: 12,000 contracts

        Entry:  742.00
        Stop:   737.00
        Target: 755.00

        RVOL: 2.3
        Volume Spike: 4.0x avg
        Supply/Demand Alignment: Demand Zone 740.00–742.00

        Time: 12:45 PM EST
        Confidence: High

    Args:
        setup:     :class:`~momentum_radar.signals.golden_sweep.GoldenSweepSetup`
                   produced by :func:`~momentum_radar.signals.golden_sweep.detect_golden_sweep`.
        timestamp: Override for the alert timestamp (defaults to ``setup.timestamp``).

    Returns:
        Multi-line formatted alert string.
    """
    # Lazy import to avoid circular dependencies at module level
    from momentum_radar.signals.golden_sweep import GoldenSweepSetup  # noqa: F401

    ts = timestamp or setup.timestamp or datetime.now()
    conf_emoji = _CONFIDENCE_EMOJI.get(setup.confidence, "✅")
    rr = setup.risk_reward
    sd_line = setup.supply_demand_zone or "N/A"

    # Derive strategy label from trade_type
    _TRADE_TYPE_TO_STRATEGY = {
        "Day Trade": "DAY TRADE",
        "Swing Trade": "SWING TRADE",
        "Position Trade": "SWING TRADE",
    }
    trade_type = getattr(setup, "trade_type", "Day Trade")
    strategy_label = _TRADE_TYPE_TO_STRATEGY.get(trade_type, "DAY TRADE")

    lines = [
        f"🚨 {strategy_label}",
        "",
        f"Ticker:            {setup.ticker}",
        f"Setup:             {setup.sweep_type} {setup.contract_type} Sweep → {setup.direction} {setup.trade_type}",
        f"Underlying Price:  {setup.underlying_price:.2f}",
        f"Strike(s):         {setup.strike:.0f} {setup.contract_type}(s)",
        f"Expiration:        {setup.sweep_type}",
        f"Estimated Flow:    {setup.contracts:,} contracts",
        "",
        f"Entry:             {setup.entry:.2f}",
        f"Stop:              {setup.stop:.2f}",
        f"Target:            {setup.target:.2f}",
        "",
        f"Risk/Reward:       1:{rr:.1f}",
        f"RVOL:              {setup.rvol:.1f}",
        f"Volume Spike:      {setup.volume_spike:.1f}x avg",
        f"Supply/Demand Alignment: {sd_line}",
        "",
        f"Time:              {ts.strftime('%I:%M %p EST')}",
        f"Confidence:        {conf_emoji} {setup.confidence}",
    ]

    if setup.details:
        lines += ["", f"Details: {setup.details}"]

    return "\n".join(lines)
