"""
tradingview.py – TradingView Integration Helpers.

Provides TradingView chart URLs and, when the optional *tradingview-ta*
package is installed, live technical-analysis signals pulled from
TradingView's public screener API.

No authentication or API key is required for either feature.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chart link builder
# ---------------------------------------------------------------------------

TRADINGVIEW_BASE = "https://www.tradingview.com"


def get_chart_url(ticker: str, exchange: str = "NASDAQ", interval: str = "D") -> str:
    """Return a TradingView chart URL for *ticker*.

    Args:
        ticker:   Stock symbol (e.g. ``"AAPL"``).
        exchange: Exchange code.  Common values: ``"NASDAQ"``, ``"NYSE"``,
                  ``"AMEX"``.  Defaults to ``"NASDAQ"``.
        interval: Chart interval.  TradingView codes: ``"1"`` (1 min),
                  ``"5"``, ``"15"``, ``"60"`` (1 h), ``"D"`` (daily),
                  ``"W"`` (weekly).  Defaults to ``"D"``.

    Returns:
        HTTPS URL string.
    """
    return f"{TRADINGVIEW_BASE}/chart/?symbol={exchange}%3A{ticker.upper()}&interval={interval}"


def get_screener_url(ticker: str) -> str:
    """Return a TradingView stock screener URL for *ticker*."""
    return f"{TRADINGVIEW_BASE}/symbols/{ticker.upper()}/"


# ---------------------------------------------------------------------------
# Technical Analysis via tradingview-ta (optional dependency)
# ---------------------------------------------------------------------------

def get_tradingview_analysis(
    ticker: str,
    exchange: str = "NASDAQ",
    screener: str = "america",
    interval: str = "1d",
) -> Optional[Dict]:
    """Fetch live technical-analysis signals from TradingView's screener API.

    This function requires the *tradingview-ta* package.  If it is not
    installed the function returns ``None`` silently.

    Args:
        ticker:   Stock symbol.
        exchange: TradingView exchange identifier (e.g. ``"NASDAQ"``).
        screener: TradingView screener region (e.g. ``"america"``).
        interval: Analysis interval.  Supported values: ``"1m"``, ``"5m"``,
                  ``"15m"``, ``"30m"``, ``"1h"``, ``"2h"``, ``"4h"``,
                  ``"1d"``, ``"1W"``, ``"1M"``.

    Returns:
        Dict with keys ``recommendation``, ``buy_signals``,
        ``sell_signals``, ``neutral_signals``, ``oscillators``,
        ``moving_averages``, ``indicators``; or ``None`` when the package
        is unavailable or the request fails.
    """
    try:
        from tradingview_ta import TA_Handler, Interval  # type: ignore
    except ImportError:
        logger.debug("tradingview-ta not installed; TradingView analysis unavailable.")
        return None

    _interval_map = {
        "1m": Interval.INTERVAL_1_MINUTE,
        "5m": Interval.INTERVAL_5_MINUTES,
        "15m": Interval.INTERVAL_15_MINUTES,
        "30m": Interval.INTERVAL_30_MINUTES,
        "1h": Interval.INTERVAL_1_HOUR,
        "2h": Interval.INTERVAL_2_HOURS,
        "4h": Interval.INTERVAL_4_HOURS,
        "1d": Interval.INTERVAL_1_DAY,
        "1W": Interval.INTERVAL_1_WEEK,
        "1M": Interval.INTERVAL_1_MONTH,
    }

    tv_interval = _interval_map.get(interval, Interval.INTERVAL_1_DAY)

    try:
        handler = TA_Handler(
            symbol=ticker.upper(),
            exchange=exchange.upper(),
            screener=screener,
            interval=tv_interval,
        )
        analysis = handler.get_analysis()
        summary = analysis.summary
        oscillators = analysis.oscillators
        moving_avgs = analysis.moving_averages
        indicators = analysis.indicators

        return {
            "recommendation": summary.get("RECOMMENDATION", "N/A"),
            "buy_signals": summary.get("BUY", 0),
            "sell_signals": summary.get("SELL", 0),
            "neutral_signals": summary.get("NEUTRAL", 0),
            "oscillators": {
                "recommendation": oscillators.get("RECOMMENDATION", "N/A"),
                "buy": oscillators.get("BUY", 0),
                "sell": oscillators.get("SELL", 0),
            },
            "moving_averages": {
                "recommendation": moving_avgs.get("RECOMMENDATION", "N/A"),
                "buy": moving_avgs.get("BUY", 0),
                "sell": moving_avgs.get("SELL", 0),
            },
            "rsi": indicators.get("RSI", None),
            "macd": indicators.get("MACD.macd", None),
            "macd_signal": indicators.get("MACD.signal", None),
            "adx": indicators.get("ADX", None),
            "cci": indicators.get("CCI20", None),
            "stoch_k": indicators.get("Stoch.K", None),
            "stoch_d": indicators.get("Stoch.D", None),
        }
    except Exception as exc:
        logger.debug("TradingView analysis failed for %s: %s", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

def format_tradingview_section(
    ticker: str,
    tv_analysis: Optional[Dict],
    exchange: str = "NASDAQ",
) -> str:
    """Render TradingView data as a formatted text section.

    Args:
        ticker:      Stock symbol.
        tv_analysis: Dict from :func:`get_tradingview_analysis`, or ``None``.
        exchange:    Exchange for URL generation.

    Returns:
        Multi-line string.
    """
    lines: List[str] = [
        "TRADINGVIEW",
        f"  Chart:    {get_chart_url(ticker, exchange)}",
        f"  Overview: {get_screener_url(ticker)}",
    ]

    if tv_analysis:
        rec = tv_analysis.get("recommendation", "N/A")
        buys = tv_analysis.get("buy_signals", 0)
        sells = tv_analysis.get("sell_signals", 0)
        neutrals = tv_analysis.get("neutral_signals", 0)
        lines.append("")
        lines.append(f"  TradingView Consensus: {rec}  ({buys} buy / {neutrals} neutral / {sells} sell)")

        osc = tv_analysis.get("oscillators", {})
        ma = tv_analysis.get("moving_averages", {})
        if osc.get("recommendation"):
            lines.append(f"  Oscillators: {osc['recommendation']}")
        if ma.get("recommendation"):
            lines.append(f"  Moving Averages: {ma['recommendation']}")

        rsi = tv_analysis.get("rsi")
        if rsi is not None:
            lines.append(f"  RSI(14): {rsi:.1f}")

        adx = tv_analysis.get("adx")
        if adx is not None:
            lines.append(f"  ADX: {adx:.1f}")

    return "\n".join(lines)
