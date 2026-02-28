"""
main.py – Momentum Signal Radar entry point.

Starts the asynchronous scanning loop that:
1. Builds the stock universe
2. Fetches market context (SPY/QQQ)
3. Scans every ticker in the universe
4. Evaluates all registered signal modules
5. Sends Telegram + console alerts for qualifying scores
6. Persists all alerts to SQLite and CSV
7. Respects market hours, lunch-lull suppression, and per-ticker cooldowns

Usage::

    python -m momentum_radar.main

Or simply::

    python main.py

Pattern research mode::

    python main.py --pattern "double bottom"
    python main.py --pattern "head and shoulders" --tickers AAPL,TSLA,MSFT
    python main.py --bot
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

from momentum_radar.config import config
from momentum_radar.data.data_fetcher import get_data_fetcher
from momentum_radar.data.universe_builder import UniverseBuilder

# Import signal modules so they register themselves
import momentum_radar.signals.volume  # noqa: F401
import momentum_radar.signals.volatility  # noqa: F401
import momentum_radar.signals.structure  # noqa: F401
import momentum_radar.signals.short_interest  # noqa: F401
import momentum_radar.signals.options_flow  # noqa: F401

from momentum_radar.signals.scoring import compute_score, AlertLevel
from momentum_radar.alerts.formatter import format_alert
from momentum_radar.alerts.telegram_alert import send_telegram_alert
from momentum_radar.storage.database import init_db, save_alert
from momentum_radar.storage.logger import log_alert_csv
from momentum_radar.utils.indicators import compute_rvol, compute_atr
from momentum_radar.utils.market_hours import (
    is_market_open,
    is_lunch_lull,
    get_market_score_penalty,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=config.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_SHUTDOWN = False


def _handle_shutdown(signum: int, frame) -> None:  # type: ignore[type-arg]
    global _SHUTDOWN
    logger.info("Shutdown signal received (%s). Stopping after current cycle…", signum)
    _SHUTDOWN = True


signal.signal(signal.SIGINT, _handle_shutdown)
signal.signal(signal.SIGTERM, _handle_shutdown)

# ---------------------------------------------------------------------------
# Per-ticker alert cooldown tracker
# ---------------------------------------------------------------------------

_last_alert_time: Dict[str, float] = {}


def _is_on_cooldown(ticker: str) -> bool:
    cooldown = config.scan.alert_cooldown_seconds
    last = _last_alert_time.get(ticker, 0.0)
    return (time.time() - last) < cooldown


def _mark_alerted(ticker: str) -> None:
    _last_alert_time[ticker] = time.time()


# ---------------------------------------------------------------------------
# Market context helper
# ---------------------------------------------------------------------------

def _get_market_context(fetcher) -> tuple:
    """Fetch SPY and QQQ percentage changes.

    Returns:
        Tuple (spy_pct, qqq_pct, penalty, condition_str).
    """
    spy_pct: Optional[float] = None
    qqq_pct: Optional[float] = None
    try:
        for sym, store in [("SPY", "spy"), ("QQQ", "qqq")]:
            quote = fetcher.get_quote(sym)
            if quote and quote.get("price") and quote.get("prev_close"):
                pct = (quote["price"] - quote["prev_close"]) / quote["prev_close"]
                if sym == "SPY":
                    spy_pct = pct
                else:
                    qqq_pct = pct
    except Exception as exc:
        logger.warning("Could not fetch market context: %s", exc)

    penalty, condition = get_market_score_penalty(spy_pct, qqq_pct)
    return spy_pct, qqq_pct, penalty, condition


# ---------------------------------------------------------------------------
# Single-ticker scan
# ---------------------------------------------------------------------------

def _scan_ticker(
    ticker: str,
    fetcher,
    market_penalty: int,
    market_condition: str,
) -> None:
    """Fetch data for *ticker* and evaluate all signal modules."""
    try:
        bars = fetcher.get_intraday_bars(ticker, interval="1m", period="1d")
        daily = fetcher.get_daily_bars(ticker, period="60d")
        fundamentals = fetcher.get_fundamentals(ticker)
        options = fetcher.get_options_volume(ticker)

        result = compute_score(
            ticker=ticker,
            bars=bars,
            daily=daily,
            fundamentals=fundamentals,
            options=options,
            market_score_penalty=market_penalty,
        )

        score: int = result["score"]
        alert_level: AlertLevel = result["alert_level"]
        triggered: List[str] = result["triggered_modules"]
        details: Dict[str, str] = result["module_details"]

        logger.debug("%s → score=%d level=%s", ticker, score, alert_level.value)

        if score < config.scores.alert_minimum:
            return

        if _is_on_cooldown(ticker):
            logger.debug("%s is on alert cooldown, skipping.", ticker)
            return

        # Gather display values
        price: Optional[float] = None
        pct_change: float = 0.0
        try:
            quote = fetcher.get_quote(ticker)
            if quote:
                price = quote.get("price")
                prev = quote.get("prev_close")
                if price and prev and prev != 0:
                    pct_change = ((price - prev) / prev) * 100
        except Exception:
            pass

        rvol = compute_rvol(bars, daily) or 0.0
        atr = compute_atr(daily) if daily is not None else None
        atr_ratio: Optional[float] = None
        if bars is not None and not bars.empty and atr:
            day_range = float(bars["high"].max()) - float(bars["low"].min())
            atr_ratio = day_range / atr if atr > 0 else None

        short_interest_val: Optional[float] = None
        float_shares_val: Optional[float] = None
        if fundamentals:
            short_interest_val = fundamentals.get("short_percent_of_float")
            float_shares_val = fundamentals.get("float_shares")

        now = datetime.now()
        message = format_alert(
            ticker=ticker,
            price=price or 0.0,
            pct_change=pct_change,
            rvol=rvol,
            score=score,
            alert_level=alert_level,
            triggered_modules=triggered,
            module_details=details,
            short_interest=short_interest_val,
            float_shares=float_shares_val,
            atr_ratio=atr_ratio,
            timestamp=now,
        )

        # Console output
        logger.info("\n%s", message)

        # Telegram
        if score >= config.scores.alert_minimum:
            send_telegram_alert(message)

        # Persist
        save_alert(
            ticker=ticker,
            price=price,
            score=score,
            alert_level=alert_level.value,
            modules_triggered=triggered,
            rvol=rvol,
            atr_ratio=atr_ratio,
            short_interest=short_interest_val,
            float_shares=float_shares_val,
            market_condition=market_condition,
            pct_change=pct_change,
            timestamp=now,
        )
        log_alert_csv(
            ticker=ticker,
            price=price,
            score=score,
            alert_level=alert_level.value,
            modules_triggered=triggered,
            rvol=rvol,
            atr_ratio=atr_ratio,
            short_interest=short_interest_val,
            float_shares=float_shares_val,
            market_condition=market_condition,
            pct_change=pct_change,
            timestamp=now,
        )

        _mark_alerted(ticker)

    except Exception as exc:
        logger.error("Unhandled error scanning %s: %s", ticker, exc, exc_info=True)


# ---------------------------------------------------------------------------
# Main scan loop
# ---------------------------------------------------------------------------

async def run_scanner() -> None:
    """Run the full scanning loop until a shutdown signal is received."""
    logger.info("Momentum Signal Radar starting up…")

    init_db()
    fetcher = get_data_fetcher(config.data.provider)
    universe_builder = UniverseBuilder(fetcher)

    logger.info("Building stock universe…")
    universe: List[str] = universe_builder.build()
    if not universe:
        logger.warning("Universe is empty – check data provider and filters.")

    logger.info("Universe ready: %d tickers. Starting scan loop.", len(universe))

    while not _SHUTDOWN:
        now = datetime.now()

        if not is_market_open(now):
            logger.info("Market closed. Waiting for next check…")
            await asyncio.sleep(60)
            continue

        lull = is_lunch_lull(now)
        if lull:
            logger.info("Lunch-lull window – scan frequency reduced.")

        _, _, market_penalty, market_condition = _get_market_context(fetcher)

        logger.info(
            "Scan cycle starting at %s | market=%s | penalty=%d",
            now.strftime("%H:%M:%S"),
            market_condition,
            market_penalty,
        )

        for ticker in universe:
            if _SHUTDOWN:
                break
            _scan_ticker(ticker, fetcher, market_penalty, market_condition)
            await asyncio.sleep(0)  # yield control to the event loop

        interval = config.scan.interval_seconds
        if lull:
            interval *= 2  # slower during lunch lull
        logger.info("Scan cycle complete. Next cycle in %ds.", interval)
        await asyncio.sleep(interval)

    logger.info("Momentum Signal Radar shut down cleanly.")


def main() -> None:
    """Entry point wrapper."""
    parser = argparse.ArgumentParser(
        description="Momentum Signal Radar",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Pattern to scan for (e.g. 'double bottom', 'head and shoulders')",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated tickers to scan (optional, default: full universe)",
    )
    parser.add_argument(
        "--bot",
        action="store_true",
        help="Start Telegram bot listener mode (runs continuously)",
    )
    args = parser.parse_args()

    if args.bot:
        # Start the Telegram pattern bot
        from momentum_radar.telegram.bot import start_telegram_bot

        asyncio.run(start_telegram_bot())
    elif args.pattern:
        # Run a one-shot pattern scan
        _run_pattern_scan(args.pattern, args.tickers)
    else:
        # Normal momentum scanner
        asyncio.run(run_scanner())


def _run_pattern_scan(pattern_name: str, tickers_arg: Optional[str]) -> None:
    """Run a one-shot pattern scan from the CLI.

    Args:
        pattern_name: Name of the pattern to detect.
        tickers_arg:  Optional comma-separated ticker override.
    """
    from momentum_radar.patterns.detector import scan_for_pattern, available_patterns
    from momentum_radar.patterns.charts import generate_pattern_chart
    from momentum_radar.alerts.telegram_alert import send_telegram_photo

    if pattern_name.lower() not in available_patterns():
        print(
            f"Unknown pattern: '{pattern_name}'\n"
            f"Available patterns:\n  " + "\n  ".join(available_patterns())
        )
        sys.exit(1)

    fetcher = get_data_fetcher(config.data.provider)

    if tickers_arg:
        tickers: List[str] = [t.strip().upper() for t in tickers_arg.split(",") if t.strip()]
    else:
        universe_builder = UniverseBuilder(fetcher)
        tickers = universe_builder.build()

    print(
        f"🔍 Scanning {len(tickers)} tickers for '{pattern_name}'… "
        "This may take a moment."
    )
    matches = scan_for_pattern(pattern_name, tickers, fetcher, top_n=5)

    if not matches:
        print(f"No matches found for '{pattern_name}' (confidence ≥ 60%).")
        return

    print(f"\n✅ Found {len(matches)} match(es):\n")
    for match in matches:
        ticker = match["ticker"]
        confidence = match.get("confidence", 0)
        description = match.get("description", "")
        df = match.get("df")

        print(f"  📊 {ticker} — {match['pattern']} ({confidence}%)")
        print(f"     {description}\n")

        # Generate and save chart
        if df is not None and not df.empty:
            try:
                chart_path = generate_pattern_chart(ticker, df, match)
                print(f"     Chart saved: {chart_path}")
                caption = (
                    f"{ticker} — {match['pattern']} "
                    f"(Confidence: {confidence}%)\n{description}"
                )
                if not send_telegram_photo(chart_path, caption):
                    logger.debug("Telegram photo delivery skipped or failed for %s.", ticker)
            except Exception as exc:
                logger.warning("Could not generate chart for %s: %s", ticker, exc)


if __name__ == "__main__":
    main()
