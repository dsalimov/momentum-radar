"""
alerts/golden_sweep_formatter.py – Professional alert formatters.

Produces the two canonical institutional alert formats defined in the
project blueprint:

1. **Golden Sweep Alert** – for large options-flow sweeps::

       🚨 GOLDEN SWEEP ALERT
       Ticker: TSLA
       Setup: Weekly Call Sweep → Bullish Day Trade
       ...

2. **Autonomous Trade Alert** – for chart-pattern-driven setups::

       🚨 AUTONOMOUS TRADE ALERT
       Ticker: NVDA
       Setup: Ascending Triangle Breakout → Bullish Swing Trade
       ...

Both functions return a plain string suitable for Telegram messages and
Discord text payloads.

Usage::

    from momentum_radar.alerts.golden_sweep_formatter import (
        format_golden_sweep_alert,
        format_chart_pattern_alert,
    )

    msg = format_golden_sweep_alert(sweep_alert)
    msg = format_chart_pattern_alert(setup, pattern_name="Ascending Triangle")
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from momentum_radar.signals.golden_sweep import SweepAlert
from momentum_radar.signals.setup_detector import SetupDirection, TradeSetup

#: RVOL threshold used to classify a pattern alert as a Day Trade vs. Swing Trade.
_PATTERN_DAY_TRADE_RVOL: float = 2.5


# ---------------------------------------------------------------------------
# Golden Sweep Alert
# ---------------------------------------------------------------------------

def format_golden_sweep_alert(
    sweep: SweepAlert,
    timestamp: Optional[datetime] = None,
) -> str:
    """Format a :class:`~momentum_radar.signals.golden_sweep.SweepAlert`
    into the canonical blueprint alert string.

    Example output::

        🚨 GOLDEN SWEEP ALERT
        Ticker: TSLA
        Setup: Weekly Call Sweep → Bullish Day Trade
        Underlying Price: 740.50
        Strike(s): 750 Call(s)
        Expiration: Weekly
        Estimated Flow: 12,000 contracts
        Entry: 742.00
        Stop: 737.00
        Target: 755.00–760.00
        RVOL: 2.3
        Volume Spike: 4.0x avg
        Supply/Demand Alignment: Demand Zone 740–742
        Time: 12:45 PM EST
        Confidence: High

    Args:
        sweep:     :class:`SweepAlert` produced by
                   :func:`~momentum_radar.signals.golden_sweep.detect_golden_sweep`.
        timestamp: Override for the alert time (defaults to ``sweep.timestamp``).

    Returns:
        Formatted multi-line string.
    """
    ts = timestamp or sweep.timestamp or datetime.now()
    time_str = ts.strftime("%I:%M %p EST")

    bias_label = "Bullish" if sweep.direction == "bullish" else "Bearish"
    contract_label = "Call" if sweep.contract_type == "call" else "Put"
    setup_label = f"{sweep.expiration} {contract_label} Sweep → {bias_label} {sweep.trade_type}"

    flow_label = (
        f"~${sweep.estimated_flow:,.0f} notional"
        if sweep.estimated_flow >= 1_000
        else f"{sweep.contract_volume:,} contracts"
    )

    lines = [
        "🚨 GOLDEN SWEEP ALERT",
        f"Ticker: {sweep.ticker}",
        f"Setup: {setup_label}",
        f"Underlying Price: {sweep.underlying_price:.2f}",
        f"Strike(s): {sweep.strike:.0f} {contract_label}(s)",
        f"Expiration: {sweep.expiration}",
        f"Estimated Flow: {flow_label}",
        f"Entry: {sweep.entry:.2f}",
        f"Stop: {sweep.stop:.2f}",
        f"Target: {sweep.target:.2f}",
        f"RVOL: {sweep.rvol}",
        f"Volume Spike: {sweep.volume_spike:.1f}x avg",
        f"Supply/Demand Alignment: {sweep.zone_alignment}",
        f"Time: {time_str}",
        f"Confidence: {sweep.confidence}",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Autonomous Trade Alert (Chart Pattern)
# ---------------------------------------------------------------------------

def format_chart_pattern_alert(
    setup: TradeSetup,
    pattern_name: str,
    sweep_info: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> str:
    """Format a chart-pattern trade setup into the canonical blueprint alert string.

    Example output::

        🚨 AUTONOMOUS TRADE ALERT
        Ticker: NVDA
        Setup: Ascending Triangle Breakout → Bullish Swing Trade
        Pattern Detected: Ascending Triangle
        Entry: 905.00
        Stop: 898.00
        Target: 920.00–925.00
        RVOL: 2.1
        Volume Spike: 3.8x avg
        Supply/Demand Alignment: Demand Zone 902–905
        Golden Sweep: Weekly Call Sweep 5,000 contracts
        Time: 01:15 PM EST
        Confidence: High

    Args:
        setup:        :class:`~momentum_radar.signals.setup_detector.TradeSetup`
                      from the setup detector.
        pattern_name: Human-readable pattern name, e.g. ``"Ascending Triangle"``.
        sweep_info:   Optional sweep annotation string
                      (e.g. ``"Weekly Call Sweep 5,000 contracts"``).
                      Pass ``None`` to omit the Golden Sweep line.
        timestamp:    Override for the alert time.

    Returns:
        Formatted multi-line string.
    """
    ts = timestamp or setup.timestamp or datetime.now()
    time_str = ts.strftime("%I:%M %p EST")

    direction_label = (
        "Bullish" if setup.direction == SetupDirection.LONG else "Bearish"
    )

    # Derive trade type from RVOL and pattern
    trade_type_label = "Swing Trade" if setup.rvol < _PATTERN_DAY_TRADE_RVOL else "Day Trade"
    setup_label = f"{pattern_name} Breakout → {direction_label} {trade_type_label}"

    lines = [
        "🚨 AUTONOMOUS TRADE ALERT",
        f"Ticker: {setup.ticker}",
        f"Setup: {setup_label}",
        f"Pattern Detected: {pattern_name}",
        f"Entry: {setup.entry:.2f}",
        f"Stop: {setup.stop:.2f}",
        f"Target: {setup.target:.2f}",
        f"RVOL: {setup.rvol}",
        f"Volume Spike: {setup.volume_spike:.1f}x avg",
        f"Supply/Demand Alignment: {setup.details or 'Not specified'}",
    ]

    if sweep_info:
        lines.append(f"Golden Sweep: {sweep_info}")

    lines += [
        f"Time: {time_str}",
        f"Confidence: {setup.confidence}",
    ]

    return "\n".join(lines)
