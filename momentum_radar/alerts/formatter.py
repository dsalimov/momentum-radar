"""
formatter.py – Alert message formatting for console and Telegram.

Two formatters are provided:

- :func:`format_alert`          – original score-based alert (backward-compatible)
- :func:`format_advanced_alert` – structured institutional-grade alert with
  entry / stop / target / R:R / confidence % / win rate / market regime,
  matching the spec::

      TICKER: XYZ
      SETUP: Resistance Breakout
      CONFIDENCE: 82%
      RISK: Medium
      ENTRY: 12.45
      STOP: 11.90
      TARGET: 14.20
      R:R: 2.3
      VOLUME: 2.4x Avg
      OPTIONS FLOW: Bullish
      MARKET REGIME: Risk-On
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from momentum_radar.signals.scoring import AlertLevel

logger = logging.getLogger(__name__)

_LEVEL_EMOJI: Dict[AlertLevel, str] = {
    AlertLevel.IGNORE: "ℹ️",
    AlertLevel.WATCHLIST: "👀",
    AlertLevel.HIGH_PRIORITY: "🚨",
    AlertLevel.STRONG_MOMENTUM: "🔥",
}

_LEVEL_LABEL: Dict[AlertLevel, str] = {
    AlertLevel.IGNORE: "IGNORE",
    AlertLevel.WATCHLIST: "WATCHLIST",
    AlertLevel.HIGH_PRIORITY: "HIGH PRIORITY SIGNAL",
    AlertLevel.STRONG_MOMENTUM: "STRONG MOMENTUM SIGNAL",
}


def format_alert(
    ticker: str,
    price: float,
    pct_change: float,
    rvol: float,
    score: int,
    alert_level: AlertLevel,
    triggered_modules: List[str],
    module_details: Dict[str, str],
    short_interest: Optional[float] = None,
    float_shares: Optional[float] = None,
    atr_ratio: Optional[float] = None,
    timestamp: Optional[datetime] = None,
) -> str:
    """Build a formatted alert string.

    Args:
        ticker: Stock symbol.
        price: Current price.
        pct_change: Percentage change today (e.g. 6.4 for +6.4%).
        rvol: Relative volume ratio.
        score: Total signal score.
        alert_level: :class:`~momentum_radar.signals.scoring.AlertLevel`.
        triggered_modules: List of signal names that fired.
        module_details: Mapping of signal name to detail string.
        short_interest: Short interest as a decimal (e.g. 0.15 for 15%).
        float_shares: Float shares count.
        atr_ratio: Current day range divided by ATR.
        timestamp: Alert timestamp (defaults to ``datetime.now()``).

    Returns:
        Formatted alert string.
    """
    ts = timestamp or datetime.now()
    emoji = _LEVEL_EMOJI.get(alert_level, "📢")
    label = _LEVEL_LABEL.get(alert_level, str(alert_level.value).upper())
    pct_sign = "+" if pct_change >= 0 else ""

    lines: List[str] = [
        f"{emoji} {label}",
        "",
        f"Ticker: {ticker}",
        f"Price: {price:.2f}",
        f"% Change: {pct_sign}{pct_change:.1f}%",
        f"RVOL: {rvol:.1f}",
        f"Score: {score}",
        "",
        "Triggers:",
    ]

    for module in triggered_modules:
        detail = module_details.get(module, "")
        display_name = module.replace("_", " ").title()
        if detail:
            lines.append(f"  - {display_name}: {detail}")
        else:
            lines.append(f"  - {display_name}")

    if not triggered_modules:
        lines.append("  (none)")

    lines.append("")

    if atr_ratio is not None:
        lines.append(f"Range vs ATR: {atr_ratio:.1f}x")
    if float_shares is not None:
        lines.append(f"Float: {float_shares / 1e6:.0f}M")
    if short_interest is not None:
        lines.append(f"Short Interest: {short_interest:.1%}")
    lines.append(f"Time: {ts.strftime('%I:%M %p EST')}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Risk-grade and setup-strength helpers
# ---------------------------------------------------------------------------

def _risk_grade(confidence_pct: float) -> str:
    """Map confidence % to a risk grade label.

    Args:
        confidence_pct: Confidence score (0–100).

    Returns:
        ``"Low"`` / ``"Medium"`` / ``"High"`` risk label.
    """
    if confidence_pct >= 80:
        return "Low"
    if confidence_pct >= 65:
        return "Medium"
    return "High"


def _setup_strength(confirmation_count: int, confidence_pct: float) -> str:
    """Map confirmation count and confidence to a setup-strength grade.

    Args:
        confirmation_count: Number of independent confirmations.
        confidence_pct:     Average confidence (0–100).

    Returns:
        One of ``"A+"`` / ``"A"`` / ``"B"`` / ``"C"``.
    """
    if confirmation_count >= 3 and confidence_pct >= 80:
        return "A+"
    if confirmation_count >= 3 or confidence_pct >= 75:
        return "A"
    if confirmation_count >= 2 or confidence_pct >= 65:
        return "B"
    return "C"


def _options_flow_label(options: Optional[Dict]) -> str:
    """Derive a one-word options flow sentiment from raw options data.

    Args:
        options: Options dict with ``call_volume``, ``put_volume``, etc.

    Returns:
        ``"Bullish"`` / ``"Bearish"`` / ``"Neutral"`` / ``"N/A"``.
    """
    if options is None:
        return "N/A"
    call_vol = int(options.get("call_volume", 0) or 0)
    put_vol = int(options.get("put_volume", 0) or 0)
    total = call_vol + put_vol
    if total == 0:
        return "Neutral"
    cp_ratio = call_vol / total
    if cp_ratio >= 0.60:
        return "Bullish"
    if cp_ratio <= 0.40:
        return "Bearish"
    return "Neutral"


# ---------------------------------------------------------------------------
# Advanced alert formatter
# ---------------------------------------------------------------------------

def format_advanced_alert(
    ticker: str,
    setup_type: str,
    confidence_pct: float,
    entry: float,
    stop: float,
    target: float,
    rvol: float,
    market_regime: str,
    confirmation_count: int = 2,
    win_rate_pct: Optional[float] = None,
    options: Optional[Dict] = None,
    triggered_modules: Optional[List[str]] = None,
    module_details: Optional[Dict[str, str]] = None,
    timestamp: Optional[datetime] = None,
) -> str:
    """Build an institutional-grade structured alert string.

    Output format::

        🔥 HIGH CONFIDENCE SETUP — AAPL

        TICKER:        AAPL
        SETUP:         Resistance Breakout
        CONFIDENCE:    82%
        RISK:          Medium
        GRADE:         A

        ENTRY:         148.35
        STOP:          145.20
        TARGET:        154.50
        R:R:           2.1

        VOLUME:        2.4x Avg
        OPTIONS FLOW:  Bullish
        WIN RATE:      63%
        MARKET REGIME: Risk-On

        Triggers: Volume Spike, Ascending Triangle Breakout, Call Flow Spike

    Args:
        ticker:             Stock symbol.
        setup_type:         Primary setup name (e.g. ``"Resistance Breakout"``).
        confidence_pct:     Confidence score (0–100).
        entry:              Suggested entry price.
        stop:               Stop-loss price.
        target:             Take-profit price.
        rvol:               Relative volume ratio.
        market_regime:      Market context string (e.g. ``"Risk-On"``).
        confirmation_count: Number of independent signal confirmations.
        win_rate_pct:       Historical win rate (optional).
        options:            Raw options dict for flow sentiment classification.
        triggered_modules:  Names of signals that fired.
        module_details:     Detail strings per signal.
        timestamp:          Alert timestamp (defaults to ``datetime.now()``).

    Returns:
        Formatted multi-line alert string.
    """
    ts = timestamp or datetime.now()
    risk = _risk_grade(confidence_pct)
    grade = _setup_strength(confirmation_count, confidence_pct)
    options_flow = _options_flow_label(options)
    risk_amount = abs(entry - stop)
    rr = round(abs(target - entry) / risk_amount, 1) if risk_amount > 0 else 0.0
    direction = "Long" if target > entry else "Short"

    # Header emoji based on confidence
    if confidence_pct >= 80:
        header_emoji = "🔥"
        header_label = "HIGH CONFIDENCE SETUP"
    elif confidence_pct >= 70:
        header_emoji = "🚨"
        header_label = "HIGH PRIORITY SETUP"
    else:
        header_emoji = "👀"
        header_label = "WATCHLIST SETUP"

    lines: List[str] = [
        f"{header_emoji} {header_label} — {ticker}",
        "",
        f"TICKER:        {ticker}",
        f"SETUP:         {setup_type}",
        f"DIRECTION:     {direction}",
        f"CONFIDENCE:    {confidence_pct:.0f}%",
        f"RISK:          {risk}",
        f"GRADE:         {grade}",
        "",
        f"ENTRY:         {entry:.2f}",
        f"STOP:          {stop:.2f}",
        f"TARGET:        {target:.2f}",
        f"R:R:           {rr:.1f}",
        "",
        f"VOLUME:        {rvol:.1f}x Avg",
        f"OPTIONS FLOW:  {options_flow}",
    ]

    if win_rate_pct is not None:
        lines.append(f"WIN RATE:      {win_rate_pct:.0f}%")

    lines.append(f"MARKET REGIME: {market_regime}")

    if triggered_modules:
        module_names = [m.replace("_", " ").title() for m in triggered_modules]
        lines += [
            "",
            f"Triggers: {', '.join(module_names)}",
        ]
        if module_details:
            for mod in triggered_modules:
                detail = module_details.get(mod, "")
                if detail:
                    display = mod.replace("_", " ").title()
                    lines.append(f"  · {display}: {detail}")

    lines.append(f"\nTime: {ts.strftime('%Y-%m-%d %I:%M %p EST')}")

    return "\n".join(lines)

