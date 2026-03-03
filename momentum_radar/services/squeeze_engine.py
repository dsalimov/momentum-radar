"""
services/squeeze_engine.py – Public API for the short squeeze detection engine.

This module provides a clean service-layer facade over the underlying
:mod:`momentum_radar.premarket.squeeze_detector` implementation.  External
callers (scheduler, Telegram bot, CLI tools) should import from here rather
than calling the detector directly.

Key exports
-----------
- :func:`score_ticker`       – compute squeeze score for a single ticker
- :func:`scan_universe`      – rank a list of tickers by squeeze probability
- :func:`format_alert_text`  – render an alert-ready string for Telegram
"""

import logging
from typing import Dict, List, Optional

from momentum_radar.data.data_fetcher import BaseDataFetcher
from momentum_radar.premarket.squeeze_detector import (
    build_squeeze_report,
    compute_squeeze_score,
    format_squeeze_report,
    scan_squeeze_candidates,
)

__all__ = [
    "score_ticker",
    "scan_universe",
    "format_alert_text",
    "SqueezeReport",
]

logger = logging.getLogger(__name__)

# Type alias for clarity
SqueezeReport = Dict


def score_ticker(ticker: str, fetcher: BaseDataFetcher) -> Optional[SqueezeReport]:
    """Return a full squeeze report for *ticker*.

    Delegates to :func:`~momentum_radar.premarket.squeeze_detector.build_squeeze_report`.

    Args:
        ticker:  Stock symbol (e.g. ``"GME"``).
        fetcher: Data provider instance.

    Returns:
        Squeeze report dict, or ``None`` if data is unavailable.
    """
    return build_squeeze_report(ticker, fetcher)


def scan_universe(
    tickers: List[str],
    fetcher: BaseDataFetcher,
    min_score: int = 40,
    top_n: int = 10,
) -> List[SqueezeReport]:
    """Scan *tickers* and return the top squeeze candidates.

    Args:
        tickers:   List of symbols to evaluate.
        fetcher:   Data provider instance.
        min_score: Minimum squeeze score to include (0–100).
        top_n:     Maximum number of results.

    Returns:
        List of squeeze report dicts sorted by score descending.
    """
    return scan_squeeze_candidates(tickers, fetcher, min_score=min_score, top_n=top_n)


def format_alert_text(report: SqueezeReport, confirmations: Optional[List[str]] = None) -> str:
    """Render a professional Telegram-ready alert message.

    If *confirmations* is supplied (from the signal engine), the alert header
    is upgraded to use the 🔥 / 🚨 style with confirmation checkmarks.

    Args:
        report:        Dict returned by :func:`score_ticker`.
        confirmations: Optional list of confirmation description strings.

    Returns:
        Formatted multi-line string.
    """
    ticker = report["ticker"]
    score = report["squeeze_score"]

    # Choose header based on confirmation count
    n_conf = len(confirmations) if confirmations else 0
    if n_conf >= 3:
        header = f"🚨 HIGH CONFIDENCE SETUP DETECTED: {ticker}"
    elif score >= 75 or n_conf >= 2:
        header = f"🔥 HIGH PROBABILITY SQUEEZE ALERT: {ticker}"
    else:
        header = f"📊 SQUEEZE ALERT: {ticker}"

    si_str = (
        f"{report['short_interest_pct']:.1%}"
        if report.get("short_interest_pct") is not None
        else "N/A"
    )
    dtc_str = (
        f"{report['days_to_cover']:.1f}"
        if report.get("days_to_cover") is not None
        else "N/A"
    )
    rvol_str = (
        f"{report['rvol']:.1f}x"
        if report.get("rvol") is not None
        else "N/A"
    )
    cp_str = (
        f"{report['cp_ratio']:.2f}"
        if report.get("cp_ratio") is not None
        else "N/A"
    )
    borrow_str = (
        f"~{report['borrow_fee_estimate']:.0%} p.a. (est.)"
        if report.get("borrow_fee_estimate") is not None
        else "N/A"
    )
    brk_lvl = (
        f"${report['breakout_level']:.2f}"
        if report.get("breakout_level")
        else "N/A"
    )
    bull1 = (
        f"${report['bull_target1']:.2f}"
        if report.get("bull_target1")
        else "N/A"
    )
    bull2 = (
        f"${report['bull_target2']:.2f}"
        if report.get("bull_target2")
        else "N/A"
    )
    bear = (
        f"${report['bear_target']:.2f}"
        if report.get("bear_target")
        else "N/A"
    )
    resistance = (
        f"${report['resistance']:.2f}"
        if report.get("resistance")
        else "N/A"
    )

    lines = [header, ""]
    lines.append(f"Squeeze Score: {score}")

    if confirmations:
        lines.append("")
        lines.append("Confirmations:")
        for conf in confirmations:
            lines.append(f"✔ {conf}")

    lines += [
        "",
        f"Short Interest: {si_str}",
        f"Float: {report.get('float_str', 'N/A')}",
        f"Days to Cover: {dtc_str}",
        f"Cost to Borrow: {borrow_str}",
        f"Volume vs Avg: {rvol_str}",
        f"Call/Put Ratio: {cp_str}",
        f"Breakout Level: {brk_lvl}",
        f"Next Liquidity Zone: {resistance}",
        "",
        f"Bull Target 1: {bull1}",
        f"Bull Target 2: {bull2}",
        f"Invalidation: {bear}",
        "",
    ]

    # Risk classification
    if score >= 75:
        risk_level = "High"
        volatility = "Elevated"
    elif score >= 45:
        risk_level = "Medium"
        volatility = "Moderate"
    else:
        risk_level = "Low"
        volatility = "Normal"

    lines.append(f"Risk Level: {risk_level}")
    lines.append(f"Volatility: {volatility}")
    lines.append("")
    lines.append(
        "⚠️ This is a probability-based signal, not investment advice. "
        "Always apply your own risk management."
    )

    return "\n".join(lines)
