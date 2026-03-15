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
8. Automatically switches timeframe based on time of day
9. Sends first-15-minute opening-range breakout alerts for key assets

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
from typing import Dict, List, Optional, Tuple

from momentum_radar.config import config
from momentum_radar.data.data_fetcher import get_data_fetcher
from momentum_radar.data.universe_builder import UniverseBuilder

# Import signal modules so they register themselves
import momentum_radar.signals.volume  # noqa: F401
import momentum_radar.signals.volatility  # noqa: F401
import momentum_radar.signals.structure  # noqa: F401
import momentum_radar.signals.short_interest  # noqa: F401
import momentum_radar.signals.options_flow  # noqa: F401
import momentum_radar.signals.golden_sweep  # noqa: F401
import momentum_radar.signals.trend  # noqa: F401
import momentum_radar.signals.vwap_signal  # noqa: F401
import momentum_radar.signals.support_resistance  # noqa: F401
import momentum_radar.signals.squeeze  # noqa: F401
import momentum_radar.signals.supply_demand  # noqa: F401

from momentum_radar.signals.scoring import compute_score, AlertLevel
from momentum_radar.signals.setup_detector import detect_setups
from momentum_radar.alerts.formatter import format_alert
from momentum_radar.alerts.trade_formatter import format_trade_setup
from momentum_radar.alerts.discord_alert import send_discord_alert
from momentum_radar.alerts.telegram_alert import send_telegram_alert
from momentum_radar.storage.database import init_db, save_alert, record_signal_alert
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
# Timeframe selection based on time of day
# ---------------------------------------------------------------------------

#: Priority assets monitored for the first-15-minute opening range breakout.
_OPENING_RANGE_ASSETS: List[str] = [
    "SPY", "QQQ",
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "LLY", "AVGO",
]

#: Stores (high, low) of the first 15 minutes keyed by ticker.
_opening_range_cache: Dict[str, Tuple[float, float]] = {}
#: Tracks whether an opening-range alert has already been sent for each ticker today.
_opening_range_alerted: Dict[str, str] = {}


def get_active_timeframe(current_time: datetime) -> str:
    """Return the active intraday candle timeframe based on time of day.

    Only DAY TRADE timeframes are used during market hours:

    - 09:30–09:59 → ``"5m"``  (first-15-min strategy window – 5m candles track the breakout)
    - 10:00–10:59 → ``"5m"``  (intraday momentum – day trade primary TF)
    - 11:00–close → ``"15m"`` (trend continuation – day trade secondary TF)

    Args:
        current_time: Current local time.

    Returns:
        Timeframe string suitable for a data-fetcher ``interval`` parameter.
    """
    t = current_time.strftime("%H:%M")
    if "09:30" <= t < "10:00":
        return config.timeframes.day_trade_interval         # "5m" – first-15-min window
    if "10:00" <= t < "11:00":
        return config.timeframes.day_trade_interval         # "5m"
    return config.timeframes.day_trade_secondary_interval   # "15m"


def _update_opening_range(ticker: str, bars) -> None:
    """Populate the opening-range cache from the first 15-minute bars.

    Uses the first 15 one-minute bars (09:30–09:44) to derive the opening
    high and low.  Does nothing if the cache already contains an entry for
    *ticker* today.

    Args:
        ticker: Stock symbol.
        bars:   Intraday OHLCV DataFrame (1-min, DatetimeIndex).
    """
    if ticker in _opening_range_cache:
        return
    if bars is None or bars.empty:
        return

    try:
        first_15 = bars.between_time("09:30", "09:44")
        if first_15.empty:
            return
        or_high = float(first_15["high"].max())
        or_low = float(first_15["low"].min())
        _opening_range_cache[ticker] = (or_high, or_low)
        logger.debug(
            "Opening range cached %s: high=%.2f low=%.2f", ticker, or_high, or_low
        )
    except Exception as exc:
        logger.debug("Opening range calculation failed for %s: %s", ticker, exc)


def _check_opening_range_breakout(
    ticker: str,
    bars,
    fetcher,
    market_condition: str,
    now: datetime,
) -> None:
    """Send an alert when the first-15-minute candle range is broken with confirmation.

    Strategy rules (Section 5 & 6):

    1. Wait for the first 15-minute candle to close (after 09:45).
    2. Identify the high and low of that 15-minute opening range.
    3. Monitor subsequent 5-minute candles for a clean breakout.

    Signal conditions:
    * Price breaks **above** the OR high → potential LONG (DAY TRADE)
    * Price breaks **below** the OR low  → potential SHORT (DAY TRADE)

    Confirmation required before sending (structure break confirmation):
    * **Volume increasing** on the breakout candle (≥ 1.5× recent average)
    * **Strong candle body** – body ≥ 50 % of total candle range (clean momentum)

    If volume is not increasing or the candle body is weak, the signal is
    rejected to avoid false breakouts.

    An alert is sent at most once per direction per session.

    Args:
        ticker:           Stock symbol (must be in :data:`_OPENING_RANGE_ASSETS`).
        bars:             Intraday 1-min OHLCV DataFrame.
        fetcher:          Data fetcher (used to fetch a fresh quote).
        market_condition: Current market condition string.
        now:              Current datetime.
    """
    if ticker not in _OPENING_RANGE_ASSETS:
        return
    if bars is None or bars.empty:
        return

    t = now.strftime("%H:%M")
    # Only check after the 15-minute opening range is fully formed (after 09:45)
    if t < "09:45":
        return

    _update_opening_range(ticker, bars)
    if ticker not in _opening_range_cache:
        return

    or_high, or_low = _opening_range_cache[ticker]
    last_bar = bars.iloc[-1]
    last_close = float(last_bar["close"])
    last_open = float(last_bar["open"])
    last_high = float(last_bar["high"])
    last_low = float(last_bar["low"])
    last_vol = float(last_bar["volume"])

    # Volume confirmation: breakout candle volume must be ≥ 1.5× recent average
    avg_vol = float(bars["volume"].iloc[-21:-1].mean()) if len(bars) > 1 else 0.0
    has_volume = avg_vol > 0 and last_vol >= avg_vol * 1.5

    # Structure break confirmation: candle body ≥ 50% of total candle range
    candle_range = last_high - last_low
    candle_body = abs(last_close - last_open)
    has_strong_body = candle_range > 0 and (candle_body / candle_range) >= 0.5

    direction: Optional[str] = None
    if last_close > or_high and has_volume and has_strong_body:
        direction = "BUY"
    elif last_close < or_low and has_volume and has_strong_body:
        direction = "SELL"

    if direction is None:
        # Log why signal was rejected to aid debugging
        if last_close > or_high or last_close < or_low:
            logger.debug(
                "%s – ORB rejected: has_volume=%s has_strong_body=%s (vol=%.0f avg=%.0f body_pct=%.0f%%)",
                ticker, has_volume, has_strong_body, last_vol, avg_vol,
                (candle_body / candle_range * 100) if candle_range > 0 else 0,
            )
        return

    # One alert per direction per session
    prev_direction = _opening_range_alerted.get(ticker, "")
    if prev_direction == direction:
        return

    _opening_range_alerted[ticker] = direction

    vol_ratio = last_vol / avg_vol if avg_vol > 0 else 0.0
    body_pct = candle_body / candle_range * 100 if candle_range > 0 else 0.0
    message = (
        f"📈 FIRST 15-MIN BREAKOUT – {ticker}\n"
        f"\n"
        f"Strategy Type: DAY TRADE\n"
        f"Direction:  {direction}\n"
        f"OR High:    {or_high:.2f}  |  OR Low:  {or_low:.2f}\n"
        f"Last Close: {last_close:.2f}\n"
        f"Volume:     {vol_ratio:.1f}x average  ✅ Confirmed\n"
        f"Candle Body: {body_pct:.0f}% of range  ✅ Strong\n"
        f"Market:     {market_condition}\n"
        f"Time:       {now.strftime('%I:%M %p EST')}\n"
    )
    logger.info("\n%s", message)
    send_telegram_alert(message)



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
_last_alert_direction: Dict[str, str] = {}


def _signal_direction(pct_change: float) -> str:
    """Return ``"bull"`` for positive % change, ``"bear"`` for negative."""
    return "bull" if pct_change >= 0 else "bear"


def _is_on_cooldown(ticker: str, direction: str) -> bool:
    """Return True if *ticker* is on cooldown for the given *direction*.

    Signals in the **same direction** respect the full ``ALERT_COOLDOWN``
    window.  A signal in the **opposite direction** is allowed immediately
    (direction reversal is a new event).
    """
    cooldown = config.scan.alert_cooldown_seconds
    last = _last_alert_time.get(ticker, 0.0)
    elapsed = time.time() - last
    if elapsed >= cooldown:
        return False  # cooldown expired – always allow
    # If direction flipped since last alert, allow the new signal
    last_dir = _last_alert_direction.get(ticker, "")
    if last_dir and last_dir != direction:
        return False
    return True


def _mark_alerted(ticker: str, direction: str) -> None:
    _last_alert_time[ticker] = time.time()
    _last_alert_direction[ticker] = direction


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
    now: Optional[datetime] = None,
) -> None:
    """Fetch data for *ticker* and evaluate all signal modules."""
    if now is None:
        now = datetime.now()

    try:
        # Use the timeframe appropriate for the current session window
        active_tf = get_active_timeframe(now)
        bars = fetcher.get_intraday_bars(ticker, interval=active_tf, period="1d")
        # Always fetch 1-min bars separately for opening-range logic
        bars_1m = fetcher.get_intraday_bars(ticker, interval="1m", period="1d")
        daily = fetcher.get_daily_bars(ticker, period="60d")
        fundamentals = fetcher.get_fundamentals(ticker)
        options = fetcher.get_options_volume(ticker)

        # --- Opening range breakout check (key assets only) ---
        _check_opening_range_breakout(ticker, bars_1m, fetcher, market_condition, now)

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
        weighted_score: int = result["weighted_score"]
        confirmation_count: int = result["confirmation_count"]
        module_scores: Dict[str, int] = result["module_scores"]
        chop_suppressed: bool = result["chop_suppressed"]

        logger.debug(
            "%s → score=%d weighted=%d confs=%d chop=%s level=%s",
            ticker, score, weighted_score, confirmation_count,
            chop_suppressed, alert_level.value,
        )

        # High-probability quality gate: require sufficient weighted score AND
        # at least min_signal_confirmations independent modules
        min_weighted = config.scores.signal_score_minimum
        min_confs = config.scores.min_signal_confirmations

        if weighted_score < min_weighted or confirmation_count < min_confs:
            logger.debug(
                "%s – gate not met (weighted=%d/%d, confs=%d/%d)%s",
                ticker, weighted_score, min_weighted,
                confirmation_count, min_confs,
                " [chop suppressed]" if chop_suppressed else "",
            )
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

        direction = _signal_direction(pct_change)

        if _is_on_cooldown(ticker, direction):
            logger.debug(
                "%s is on %s cooldown, skipping.", ticker, direction
            )
            return

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
            weighted_score=weighted_score,
            module_scores=module_scores,
        )

        # Console output
        logger.info("\n%s", message)

        # Telegram
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

        _mark_alerted(ticker, direction)
        _cycle_alerted.add(ticker)  # prevent setup scanner from double-alerting this ticker
        # Register with the shared DB cooldown so the hourly scheduler does not
        # re-alert this ticker for the same general signal type within 4 hours.
        try:
            record_signal_alert(ticker, "general")
        except Exception:
            pass

    except Exception as exc:
        logger.error("Unhandled error scanning %s: %s", ticker, exc, exc_info=True)


# ---------------------------------------------------------------------------
# Professional setup scanner (new pipeline)
# ---------------------------------------------------------------------------

#: Per-ticker cooldown for trade setup alerts (maps ticker → last alert epoch).
_setup_alert_time: Dict[str, float] = {}
#: Cooldown in seconds between trade setup alerts for the same ticker.
_SETUP_COOLDOWN_SECONDS: int = 900  # 15 minutes

#: Tracks tickers that have already been alerted in the current scan cycle.
#: Cleared at the start of every new cycle to prevent cross-strategy duplicates.
_cycle_alerted: set = set()


def _is_setup_on_cooldown(ticker: str) -> bool:
    """Return True if a trade-setup alert was recently sent for *ticker*."""
    last = _setup_alert_time.get(ticker, 0.0)
    return (time.time() - last) < _SETUP_COOLDOWN_SECONDS


def _mark_setup_alerted(ticker: str) -> None:
    _setup_alert_time[ticker] = time.time()
    _cycle_alerted.add(ticker)


def _scan_setups(
    ticker: str,
    fetcher,
    now: Optional[datetime] = None,
) -> None:
    """Run the professional setup detector for *ticker* and alert on findings.

    Fetches data at strategy-appropriate timeframes:

    * Intraday bars at the active session timeframe (scalp/day/swing interval)
    * Hourly bars for swing-trade pattern detection
    * **200 days** of daily bars to cover 50- and 200-day moving averages

    Per-ticker cooldown of 15 minutes prevents duplicate alerts.  A per-cycle
    deduplication set (``_cycle_alerted``) prevents the same ticker appearing
    in both the legacy scorer and the setup scanner within a single cycle.

    Args:
        ticker:  Stock symbol.
        fetcher: Data fetcher instance.
        now:     Current datetime (defaults to ``datetime.now()``).
    """
    if now is None:
        now = datetime.now()

    if _is_setup_on_cooldown(ticker):
        logger.debug("%s – setup scanner: on cooldown, skipping.", ticker)
        return

    # Per-cycle duplicate guard: skip if this ticker was already alerted this cycle
    if ticker in _cycle_alerted:
        logger.debug("%s – already alerted this cycle, skipping setup scan.", ticker)
        return

    try:
        # Primary intraday bars at the session-appropriate interval
        active_tf = get_active_timeframe(now)
        bars = fetcher.get_intraday_bars(ticker, interval=active_tf, period="1d")

        # Hourly bars for swing-trade pattern detection (1H candles, last 30 days)
        bars_1h: Optional[object] = None
        try:
            bars_1h = fetcher.get_intraday_bars(
                ticker,
                interval=config.timeframes.swing_interval,
                period="30d",
            )
        except Exception:
            pass  # hourly bars are best-effort; daily zones still detected

        # 200 days of daily data to support 50/200 MA and swing zone detection
        swing_history = f"{config.timeframes.swing_history_days}d"
        daily = fetcher.get_daily_bars(ticker, period=swing_history)

        # Run setup detection on the primary (intraday) bars
        setups = detect_setups(ticker, bars, daily)

        # If no intraday setups and hourly bars are available, try swing detection
        if not setups and bars_1h is not None:
            try:
                setups = detect_setups(ticker, bars_1h, daily)
            except Exception as exc:
                logger.debug("%s – hourly setup detection error: %s", ticker, exc)

        if not setups:
            return

        for setup in setups:
            message = format_trade_setup(setup, timestamp=now)

            # Console output
            logger.info("\n%s", message)

            # Telegram (best-effort)
            send_telegram_alert(message)

            # Discord placeholder (activates when DISCORD_WEBHOOK_URL is set)
            try:
                chart_path: Optional[str] = None
                from momentum_radar.ui.chart_renderer import render_trade_setup_chart

                def _has_data(df_arg) -> bool:
                    return df_arg is not None and not (
                        hasattr(df_arg, "empty") and df_arg.empty
                    )

                chart_bars = bars if _has_data(bars) else bars_1h
                if _has_data(chart_bars):
                    chart_path = render_trade_setup_chart(setup, chart_bars)
            except Exception as chart_exc:
                logger.debug("Chart generation skipped for %s: %s", ticker, chart_exc)
                chart_path = None

            send_discord_alert(message, image_path=chart_path)

            logger.info(
                "Setup alert sent: %s | %s | %s | entry=%.2f stop=%.2f target=%.2f RR=%.1f",
                ticker,
                setup.setup_type.value,
                setup.direction.value,
                setup.entry,
                setup.stop,
                setup.target,
                setup.risk_reward,
            )

        _mark_setup_alerted(ticker)

    except Exception as exc:
        logger.error("Unhandled error in setup scan for %s: %s", ticker, exc, exc_info=True)


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

    # Start pre-market scheduler
    from momentum_radar.premarket.scheduler import start_scheduler, stop_scheduler
    scheduler = start_scheduler(universe, fetcher, send_telegram_alert)

    # Start hourly squeeze + signal scanner
    from momentum_radar.services.scheduler import (
        start_hourly_scheduler,
        stop_hourly_scheduler,
    )
    hourly_scheduler = start_hourly_scheduler(universe, fetcher, send_telegram_alert)

    try:
        _last_trading_date: Optional[str] = None
        while not _SHUTDOWN:
            now = datetime.now()

            if not is_market_open(now):
                logger.info("Market closed. Waiting for next check…")
                await asyncio.sleep(60)
                continue

            # Reset opening-range cache at the start of each new trading day
            today_str = now.strftime("%Y-%m-%d")
            if today_str != _last_trading_date:
                _opening_range_cache.clear()
                _opening_range_alerted.clear()
                _last_trading_date = today_str
                logger.info("Opening-range cache reset for new trading day %s.", today_str)

            lull = is_lunch_lull(now)
            if lull:
                logger.info("Lunch-lull window – scan frequency reduced.")

            _, _, market_penalty, market_condition = _get_market_context(fetcher)

            logger.info(
                "Scan cycle starting at %s | tf=%s | market=%s | penalty=%d",
                now.strftime("%H:%M:%S"),
                get_active_timeframe(now),
                market_condition,
                market_penalty,
            )

            # Clear per-cycle dedup set so each new cycle starts fresh
            _cycle_alerted.clear()

            for ticker in universe:
                if _SHUTDOWN:
                    break
                _scan_ticker(ticker, fetcher, market_penalty, market_condition, now)
                _scan_setups(ticker, fetcher, now)
                await asyncio.sleep(0)  # yield control to the event loop

            interval = config.scan.interval_seconds
            if lull:
                interval *= 2  # slower during lunch lull
            logger.info("Scan cycle complete. Next cycle in %ds.", interval)
            await asyncio.sleep(interval)
    finally:
        stop_scheduler(scheduler)
        stop_hourly_scheduler(hourly_scheduler)

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

    Only emits an alert for a match when at least 2 of 3 context checks
    (volume, MA position, ATR expansion) confirm the pattern direction.
    This ensures no pattern fires in isolation.

    Args:
        pattern_name: Name of the pattern to detect.
        tickers_arg:  Optional comma-separated ticker override.
    """
    from momentum_radar.patterns.detector import scan_for_pattern, available_patterns
    from momentum_radar.patterns.charts import generate_pattern_chart
    from momentum_radar.alerts.telegram_alert import send_telegram_photo
    from momentum_radar.alerts.golden_sweep_formatter import format_pattern_signal_alert
    from momentum_radar.services.signal_engine import get_pattern_confirmations
    from momentum_radar.utils.indicators import compute_atr

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
    alerted = 0
    for match in matches:
        ticker = match["ticker"]
        confidence = match.get("confidence", 0)
        df = match.get("df")
        direction = match.get("direction", "BULLISH")
        breakout_level = match.get("breakout_level") or match.get("neckline", 0.0)

        # Fetch daily bars for context validation (best-effort)
        daily_ctx = None
        try:
            daily_ctx = fetcher.get_daily_bars(ticker, period="60d")
        except Exception:
            pass

        # Validate: require at least 2 of 3 context confirmations
        confirmed, conf_tags = get_pattern_confirmations(
            ticker, df, daily_ctx, direction=direction
        )

        if not confirmed:
            logger.info(
                "%s – pattern '%s' detected but context unconfirmed – skipping alert.",
                ticker, pattern_name,
            )
            print(f"  ⏭  {ticker} — unconfirmed context, skipping.")
            continue

        # Calculate trade parameters
        # Prefer the pattern's own measured levels; fall back to ATR multiples
        atr_val = 0.0
        current_price = 0.0
        try:
            if daily_ctx is not None and not daily_ctx.empty:
                atr_val = compute_atr(daily_ctx) or 0.0
                current_price = float(daily_ctx["close"].iloc[-1])
        except Exception:
            pass

        direction_upper = direction.upper()
        # Entry = at the neckline (limit order waiting for breakout confirmation)
        entry = float(breakout_level) if breakout_level else current_price

        # Use pattern's own stop/target when available (most accurate)
        stop = match.get("stop_level") or 0.0
        target = match.get("target_level") or 0.0

        # Fall back to ATR multiples when pattern doesn't supply levels
        if not stop and atr_val > 0 and current_price > 0:
            stop = current_price + atr_val * 1.5 if direction_upper == "BEARISH" else current_price - atr_val * 1.5
        if not target and atr_val > 0 and current_price > 0:
            target = current_price - atr_val * 3.0 if direction_upper == "BEARISH" else current_price + atr_val * 3.0

        # Format the simplified 4–5 line alert
        msg = format_pattern_signal_alert(
            ticker=ticker,
            pattern_name=match["pattern"],
            confidence=float(confidence),
            direction=direction_upper,
            entry=entry,
            stop=float(stop) if stop else entry,
            target=float(target) if target else entry,
            breakout=entry,
            atr=atr_val,
            confirmations=conf_tags,
        )

        print(f"\n{msg}\n")

        # Generate and deliver chart
        if df is not None and not df.empty:
            try:
                chart_path = generate_pattern_chart(ticker, df, match)
                print(f"     Chart saved: {chart_path}")
                if not send_telegram_photo(chart_path, msg):
                    logger.debug("Telegram photo delivery skipped or failed for %s.", ticker)
                send_discord_alert(msg, image_path=chart_path)
            except Exception as exc:
                logger.warning("Could not generate chart for %s: %s", ticker, exc)
        else:
            send_telegram_alert(msg)
            send_discord_alert(msg)

        alerted += 1

    if alerted == 0:
        print("\n⚠️  All matches were filtered out — no context confirmations met.")


if __name__ == "__main__":
    main()
