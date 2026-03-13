"""
services/scheduler.py – Hourly automated squeeze, signal, and volume scan.

Uses APScheduler to run every hour.  Each cycle:

1. Scans the US stock universe
2. Runs :func:`~momentum_radar.services.squeeze_engine.scan_universe`
3. Runs :func:`~momentum_radar.services.signal_engine.evaluate` per candidate
4. Stores results in the database
5. Sends Telegram alerts (top 5 per hour max) when:
   - Squeeze score ≥ 75, **or**
   - 2+ signal confirmations triggered
6. Scans for unusual volume (RVOL ≥ 2.0 vs 30-day avg) and sends up to 3
   additional volume-spike alerts per hour
7. Runs a **categorized signal scan** that emits separate, focused alerts for:
   - 📊 Chart patterns (breakout, double bottom, retest, S&D zone)
   - 🕯️ Candlestick patterns (engulfing, hammer, shooting star, doji, etc.)
   - 📈 Options flow (golden sweeps, call/put spikes, gamma flip)
   - 💹 Momentum signals (RSI + MACD + HTF trend alignment)
   Each category has its own per-(ticker, type) cooldown to prevent the same
   stock appearing multiple times for the same reason.

Spam filtering is enforced via
:func:`~momentum_radar.storage.database.should_send_squeeze_alert` (squeeze)
and :func:`~momentum_radar.storage.database.should_send_signal_alert` (categories).

Usage::

    from momentum_radar.services.scheduler import (
        start_hourly_scheduler,
        stop_hourly_scheduler,
    )
    scheduler = start_hourly_scheduler(tickers, fetcher, send_fn)
    # later …
    stop_hourly_scheduler(scheduler)
"""

import logging
from typing import Callable, Dict, List, Optional, Set

from momentum_radar.data.data_fetcher import BaseDataFetcher

logger = logging.getLogger(__name__)

# Maximum alerts sent per hourly cycle to prevent spam
_MAX_ALERTS_PER_HOUR: int = 5

# Minimum squeeze score to trigger automated alert
_MIN_SQUEEZE_SCORE: int = 75

# Minimum RVOL (vs 30-day avg) to trigger an automated volume-spike alert
_MIN_RVOL_ALERT: float = 2.0

# Maximum volume-spike alerts per hourly cycle (separate cap from squeeze)
_MAX_VOLUME_ALERTS_PER_HOUR: int = 3

# Maximum categorized signal alerts per type per hourly cycle
_MAX_CHART_PATTERN_ALERTS: int = 3
_MAX_CANDLESTICK_ALERTS: int = 3
_MAX_OPTIONS_FLOW_ALERTS: int = 3
_MAX_MOMENTUM_ALERTS: int = 3

# Map each signal_engine category → categorized alert_type bucket
_CATEGORY_TO_ALERT_TYPE: Dict[str, str] = {
    "pattern": "chart_pattern",
    "retest": "chart_pattern",
    "supply_demand": "chart_pattern",
    "liquidity_sweep": "chart_pattern",
    "candlestick": "candlestick",
    "options": "options_flow",
    "volume": "squeeze_momentum",
    "htf_trend": "squeeze_momentum",
    "momentum": "squeeze_momentum",
}


def _format_volume_spike_alert(spike: dict) -> str:
    """Format a volume-spike alert string for Telegram delivery.

    Args:
        spike: Dict with keys ``ticker``, ``rvol``, ``today_volume``,
               ``last_close``, ``pct_change`` (from
               :func:`~momentum_radar.data.volume_scanner.scan_volume_spikes`).

    Returns:
        Formatted multi-line string.
    """
    ticker = spike["ticker"]
    rvol = spike["rvol"]
    today_vol = spike["today_volume"]
    last_close = spike["last_close"]
    pct_change = spike["pct_change"]

    direction = "+" if pct_change >= 0 else ""
    vol_str = (
        f"{today_vol / 1e6:.1f}M"
        if today_vol >= 1_000_000
        else f"{today_vol / 1e3:.0f}K"
    )

    lines = [
        f"📈 UNUSUAL VOLUME ALERT: {ticker}",
        "",
        f"RVOL:   {rvol:.1f}x 30-day average",
        f"Price:  ${last_close:.2f}  ({direction}{pct_change:.1f}%)",
        f"Volume: {vol_str}",
        "",
        "⚠️ High volume can precede significant price moves. "
        "Always apply your own risk management.",
    ]
    return "\n".join(lines)


def _format_chart_pattern_alert(
    ticker: str,
    confirmations: list,
    price: Optional[float] = None,
) -> str:
    """Format a chart-pattern signal alert for Telegram delivery.

    Covers confirmations with categories: ``pattern``, ``retest``,
    ``supply_demand``, ``liquidity_sweep``.

    Args:
        ticker:        Stock symbol.
        confirmations: List of :class:`~momentum_radar.services.signal_engine.Confirmation`
                       objects belonging to the chart-pattern bucket.
        price:         Current price (optional, for context).

    Returns:
        Formatted multi-line string.
    """
    price_str = f"  ${price:.2f}" if price else ""
    lines = [
        f"📊 CHART PATTERN SIGNAL: {ticker}{price_str}",
        "",
    ]
    for c in confirmations:
        lines.append(f"  ✅ {c.name}: {c.detail}  (confidence {c.confidence:.0f}%)")
    lines += [
        "",
        "💡 Chart patterns signal potential trend continuation or reversal.",
        "⚠️ Always confirm with volume and risk management before entering.",
    ]
    return "\n".join(lines)


def _format_candlestick_alert(
    ticker: str,
    confirmations: list,
    price: Optional[float] = None,
) -> str:
    """Format a candlestick-pattern signal alert for Telegram delivery.

    Args:
        ticker:        Stock symbol.
        confirmations: List of :class:`~momentum_radar.services.signal_engine.Confirmation`
                       objects with category ``"candlestick"``.
        price:         Current price (optional).

    Returns:
        Formatted multi-line string.
    """
    price_str = f"  ${price:.2f}" if price else ""
    lines = [
        f"🕯️ CANDLESTICK PATTERN SIGNAL: {ticker}{price_str}",
        "",
    ]
    for c in confirmations:
        lines.append(f"  ✅ {c.name}: {c.detail}  (confidence {c.confidence:.0f}%)")
    lines += [
        "",
        "💡 Candlestick patterns indicate potential price reversal or continuation.",
        "⚠️ Enter on confirmation candle. Apply your own risk management.",
    ]
    return "\n".join(lines)


def _format_options_flow_alert(
    ticker: str,
    confirmations: list,
    price: Optional[float] = None,
) -> str:
    """Format an options-flow signal alert for Telegram delivery.

    Covers golden sweeps, call/put volume spikes, and gamma-flip signals.

    Args:
        ticker:        Stock symbol.
        confirmations: List of :class:`~momentum_radar.services.signal_engine.Confirmation`
                       objects with category ``"options"``.
        price:         Current price (optional).

    Returns:
        Formatted multi-line string.
    """
    price_str = f"  ${price:.2f}" if price else ""
    lines = [
        f"📈 OPTIONS FLOW SIGNAL: {ticker}{price_str}",
        "",
    ]
    for c in confirmations:
        lines.append(f"  ✅ {c.name}: {c.detail}  (confidence {c.confidence:.0f}%)")
    lines += [
        "",
        "💡 Unusual options activity can precede significant directional moves.",
        "⚠️ Options carry elevated risk. Apply your own risk management.",
    ]
    return "\n".join(lines)


def _format_momentum_alert(
    ticker: str,
    confirmations: list,
    price: Optional[float] = None,
) -> str:
    """Format a momentum/squeeze signal alert for Telegram delivery.

    Covers volume spikes, HTF trend alignment, and RSI/MACD momentum.

    Args:
        ticker:        Stock symbol.
        confirmations: List of :class:`~momentum_radar.services.signal_engine.Confirmation`
                       objects with categories ``"volume"``, ``"htf_trend"``,
                       ``"momentum"``.
        price:         Current price (optional).

    Returns:
        Formatted multi-line string.
    """
    price_str = f"  ${price:.2f}" if price else ""
    lines = [
        f"💹 MOMENTUM SIGNAL: {ticker}{price_str}",
        "",
    ]
    for c in confirmations:
        lines.append(f"  ✅ {c.name}: {c.detail}  (confidence {c.confidence:.0f}%)")
    lines += [
        "",
        "💡 Multiple momentum indicators aligned – trend may be accelerating.",
        "⚠️ Always apply your own risk management.",
    ]
    return "\n".join(lines)


_ALERT_TYPE_FORMATTERS = {
    "chart_pattern": _format_chart_pattern_alert,
    "candlestick": _format_candlestick_alert,
    "options_flow": _format_options_flow_alert,
    "squeeze_momentum": _format_momentum_alert,
}

_ALERT_TYPE_CAPS = {
    "chart_pattern": _MAX_CHART_PATTERN_ALERTS,
    "candlestick": _MAX_CANDLESTICK_ALERTS,
    "options_flow": _MAX_OPTIONS_FLOW_ALERTS,
    "squeeze_momentum": _MAX_MOMENTUM_ALERTS,
}


def _run_categorized_signal_scan(
    tickers: List[str],
    fetcher: BaseDataFetcher,
    send_fn: Callable[[str], None],
    already_alerted: Optional[Set[str]] = None,
    top_n: int = 30,
) -> Set[str]:
    """Scan *tickers* for categorized signals and emit separate alerts per type.

    For each ticker a :func:`~momentum_radar.services.signal_engine.evaluate`
    call is made and the resulting confirmations are grouped into four buckets:

    - ``chart_pattern`` – breakout, double bottom, retest, S&D zone, liquidity sweep
    - ``candlestick`` – engulfing, hammer, shooting star, doji, harami, etc.
    - ``options_flow`` – golden sweep, call/put spike, gamma flip
    - ``squeeze_momentum`` – volume spike, HTF trend, RSI/MACD momentum

    Each bucket that has ≥ 1 confirmation is eligible for its own alert,
    subject to:

    - Per-(ticker, alert_type) DB cooldown of 4 hours via
      :func:`~momentum_radar.storage.database.should_send_signal_alert`.
    - Per-type alert cap (``_ALERT_TYPE_CAPS``).
    - Tickers in *already_alerted* are skipped (already covered this cycle).

    The function returns the set of all tickers that were alerted (useful for
    the caller to track cross-category dedup).

    Args:
        tickers:         Stock universe.
        fetcher:         Data provider.
        send_fn:         Alert delivery callable.
        already_alerted: Set of tickers already alerted this cycle (avoided).
        top_n:           Maximum tickers evaluated for categorized signals.

    Returns:
        Set of tickers that received at least one categorized alert.
    """
    from momentum_radar.services.signal_engine import evaluate
    from momentum_radar.storage.database import (
        should_send_signal_alert,
        record_signal_alert,
        save_alert,
    )

    if already_alerted is None:
        already_alerted = set()

    # Limit to top_n to keep runtime bounded
    eval_tickers = [t for t in tickers if t not in already_alerted][:top_n]

    # Per-type counters
    type_counts: Dict[str, int] = {t: 0 for t in _ALERT_TYPE_CAPS}
    newly_alerted: Set[str] = set()

    for ticker in eval_tickers:
        # Check if ALL category caps are reached – if so, nothing more to send
        if all(type_counts[at] >= cap for at, cap in _ALERT_TYPE_CAPS.items()):
            break

        try:
            bars = fetcher.get_intraday_bars(ticker, interval="1m", period="1d")
            daily = fetcher.get_daily_bars(ticker, period="60d")
            options = fetcher.get_options_volume(ticker)
            sig = evaluate(ticker, bars=bars, daily=daily, options=options)
        except Exception as exc:
            logger.debug("Categorized scan: evaluate failed for %s: %s", ticker, exc)
            continue

        if sig.priority == "NO_SIGNAL":
            continue

        # Get current price for alert formatting (best-effort)
        current_price: Optional[float] = None
        try:
            if daily is not None and not daily.empty and "close" in daily.columns:
                current_price = float(daily["close"].iloc[-1])
        except Exception:
            pass

        # Group confirmations by alert_type bucket
        bucket_confs: Dict[str, list] = {}
        for c in sig.confirmations:
            alert_type = _CATEGORY_TO_ALERT_TYPE.get(c.category)
            if alert_type is None:
                continue
            bucket_confs.setdefault(alert_type, []).append(c)

        # Emit one alert per triggered bucket (subject to cap + dedup)
        for alert_type, confs in bucket_confs.items():
            cap = _ALERT_TYPE_CAPS.get(alert_type, 3)
            if type_counts.get(alert_type, 0) >= cap:
                continue

            if not should_send_signal_alert(ticker, alert_type):
                logger.debug(
                    "Categorized scan: %s/%s suppressed (cooldown)", ticker, alert_type
                )
                continue

            formatter = _ALERT_TYPE_FORMATTERS.get(alert_type)
            if formatter is None:
                continue

            try:
                text = formatter(ticker, confs, price=current_price)
                send_fn(text)
                record_signal_alert(ticker, alert_type)
                save_alert(
                    ticker=ticker,
                    price=current_price,
                    score=int(sig.confidence_score),
                    alert_level=alert_type,
                    modules_triggered=[c.name for c in confs],
                )
                type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
                newly_alerted.add(ticker)
                logger.info(
                    "Categorized alert sent: %s | %s | %d confirmation(s)",
                    ticker, alert_type, len(confs),
                )
            except Exception as exc:
                logger.error(
                    "Failed to send categorized alert for %s/%s: %s",
                    ticker, alert_type, exc,
                )

    sent_counts = {k: v for k, v in type_counts.items() if v > 0}
    if sent_counts:
        logger.info("Categorized scan complete: %s", sent_counts)

    return newly_alerted


def _run_hourly_scan(
    tickers: List[str],
    fetcher: BaseDataFetcher,
    send_fn: Callable[[str], None],
) -> None:
    """Execute one hourly scan cycle.

    Args:
        tickers:  Stock symbols to scan.
        fetcher:  Data provider.
        send_fn:  Callable that delivers an alert string (e.g. to Telegram).
    """
    from momentum_radar.services.squeeze_engine import scan_universe, format_alert_text
    from momentum_radar.services.signal_engine import evaluate
    from momentum_radar.storage.database import (
        should_send_squeeze_alert,
        record_squeeze_alert,
        save_alert,
    )

    logger.info("Hourly scan: evaluating %d tickers…", len(tickers))

    # Step 1: Find top squeeze candidates (score ≥ 40)
    try:
        candidates = scan_universe(tickers, fetcher, min_score=40, top_n=20)
    except Exception as exc:
        logger.error("Hourly scan: squeeze scan failed: %s", exc)
        candidates = []

    alerts_sent = 0
    alert_queue = []

    for report in candidates:
        if alerts_sent >= _MAX_ALERTS_PER_HOUR:
            break

        ticker = report["ticker"]
        score = report["squeeze_score"]

        # Step 2: Run multi-confirmation signal engine
        try:
            bars = fetcher.get_intraday_bars(ticker, interval="1m", period="1d")
            daily = fetcher.get_daily_bars(ticker, period="60d")
            options = fetcher.get_options_volume(ticker)
            sig = evaluate(ticker, bars=bars, daily=daily, options=options)
        except Exception as exc:
            logger.debug("Signal engine failed for %s: %s", ticker, exc)
            sig = None

        conf_count = sig.confirmation_count if sig else 0
        confirmations = sig.confirmation_labels if sig else []

        should_alert = score >= _MIN_SQUEEZE_SCORE or conf_count >= 2
        if not should_alert:
            continue

        # Step 3: Spam filter
        if not should_send_squeeze_alert(ticker, score):
            logger.debug("Hourly scan: %s suppressed (spam filter)", ticker)
            continue

        alert_queue.append((score, conf_count, report, confirmations))

    # Sort by score desc, then confirmation count desc
    alert_queue.sort(key=lambda x: (x[0], x[1]), reverse=True)

    alerted_tickers: set = set()
    for score, conf_count, report, confirmations in alert_queue[:_MAX_ALERTS_PER_HOUR]:
        ticker = report["ticker"]
        try:
            text = format_alert_text(report, confirmations=confirmations or None)
            send_fn(text)
            record_squeeze_alert(ticker, score)
            # Also persist to the main alerts table
            save_alert(
                ticker=ticker,
                price=report.get("current_price"),
                score=score,
                alert_level="squeeze",
                modules_triggered=["squeeze_engine"] + (
                    [c.split(":")[0].strip() for c in confirmations] if confirmations else []
                ),
                rvol=report.get("rvol"),
                short_interest=report.get("short_interest_pct"),
                float_shares=report.get("float_shares"),
            )
            alerted_tickers.add(ticker)
            alerts_sent += 1
            logger.info(
                "Hourly alert sent: %s (score=%d, confs=%d)",
                ticker, score, conf_count,
            )
        except Exception as exc:
            logger.error("Failed to send hourly alert for %s: %s", ticker, exc)

    # -----------------------------------------------------------------------
    # Step 4: Unusual volume scan
    # Scan for stocks with RVOL >= _MIN_RVOL_ALERT that were NOT already
    # covered by the squeeze path.
    # -----------------------------------------------------------------------
    try:
        from momentum_radar.data.volume_scanner import scan_volume_spikes
        vol_spikes = scan_volume_spikes(
            tickers, fetcher, top_n=10, min_rvol=_MIN_RVOL_ALERT
        )
    except Exception as exc:
        logger.error("Hourly scan: volume spike scan failed: %s", exc)
        vol_spikes = []

    vol_alerts_sent = 0
    for spike in vol_spikes:
        if vol_alerts_sent >= _MAX_VOLUME_ALERTS_PER_HOUR:
            break

        ticker = spike["ticker"]

        # Don't double-alert tickers already covered by squeeze path
        if ticker in alerted_tickers:
            continue

        # Spam filter expects an integer score (0–100 range).  RVOL typically
        # falls in the 1–10× range, so multiplying by 10 maps it to 10–100.
        rvol_score = int(spike["rvol"] * 10)
        if not should_send_squeeze_alert(ticker, rvol_score):
            logger.debug("Hourly scan: volume alert for %s suppressed (spam filter)", ticker)
            continue

        try:
            text = _format_volume_spike_alert(spike)
            send_fn(text)
            record_squeeze_alert(ticker, rvol_score)
            save_alert(
                ticker=ticker,
                price=spike.get("last_close"),
                score=rvol_score,
                alert_level="volume_spike",
                modules_triggered=["volume_spike"],
                rvol=spike.get("rvol"),
            )
            alerted_tickers.add(ticker)
            vol_alerts_sent += 1
            logger.info(
                "Volume spike alert sent: %s (RVOL=%.1f)",
                ticker, spike["rvol"],
            )
        except Exception as exc:
            logger.error("Failed to send volume spike alert for %s: %s", ticker, exc)

    # -----------------------------------------------------------------------
    # Step 5: Categorized signal scan
    # Emit separate alerts for chart patterns, candlestick patterns, options
    # flow, and momentum signals.  Tickers already alerted above are skipped
    # to prevent the same stock appearing multiple times for the same reason.
    # -----------------------------------------------------------------------
    try:
        _run_categorized_signal_scan(
            tickers,
            fetcher,
            send_fn,
            already_alerted=set(alerted_tickers),
        )
    except Exception as exc:
        logger.error("Hourly scan: categorized signal scan failed: %s", exc)

    logger.info(
        "Hourly scan complete: %d candidate(s) evaluated, "
        "%d squeeze alert(s) sent, %d volume alert(s) sent.",
        len(candidates),
        alerts_sent,
        vol_alerts_sent,
    )


def start_hourly_scheduler(
    tickers: List[str],
    fetcher: BaseDataFetcher,
    send_fn: Callable[[str], None],
) -> Optional[object]:
    """Create and start an APScheduler ``BackgroundScheduler`` for hourly scans.

    The job fires at minute 0 of every hour (``cron hour=* minute=0``).

    Args:
        tickers:  Stock universe.
        fetcher:  Data provider.
        send_fn:  Alert delivery callable.

    Returns:
        The started :class:`~apscheduler.schedulers.background.BackgroundScheduler`,
        or ``None`` if APScheduler is not installed.
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.warning(
            "APScheduler is not installed – hourly squeeze scanning is disabled. "
            "Install it with: pip install apscheduler"
        )
        return None

    scheduler = BackgroundScheduler(timezone="America/New_York")

    scheduler.add_job(
        lambda: _run_hourly_scan(tickers, fetcher, send_fn),
        trigger=CronTrigger(minute=0, timezone="America/New_York"),
        id="hourly_squeeze_scan",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    scheduler.start()
    logger.info("Hourly squeeze scanner scheduled (every hour on the hour, ET).")
    return scheduler


def stop_hourly_scheduler(scheduler: Optional[object]) -> None:
    """Gracefully stop the hourly scheduler.

    Args:
        scheduler: Instance returned by :func:`start_hourly_scheduler`, or ``None``.
    """
    if scheduler is None:
        return
    try:
        scheduler.shutdown(wait=False)
        logger.info("Hourly squeeze scheduler stopped.")
    except Exception as exc:
        logger.warning("Error stopping hourly squeeze scheduler: %s", exc)
