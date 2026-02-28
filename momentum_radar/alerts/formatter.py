"""
formatter.py – Alert message formatting for console and Telegram.

The :func:`format_alert` function converts a scan result dict into a
human-readable string that matches the spec in the problem statement.
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
