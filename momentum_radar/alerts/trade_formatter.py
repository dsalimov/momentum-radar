"""
trade_formatter.py – Professional trade setup alert formatting.

Produces clean, human-readable trade alert strings suitable for console
output, Telegram messages, and Discord embeds.

Alert format example::

    🚨 TRADE SETUP

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

Usage::

    from momentum_radar.alerts.trade_formatter import format_trade_setup

    msg = format_trade_setup(setup)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from momentum_radar.signals.setup_detector import TradeSetup, SetupType

logger = logging.getLogger(__name__)

# Emoji per setup type – highest priority gets most prominent emoji
_SETUP_EMOJI = {
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

    lines = [
        f"🚨 TRADE SETUP",
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
