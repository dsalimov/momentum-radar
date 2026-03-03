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

Spam filtering is enforced via
:func:`~momentum_radar.storage.database.should_send_squeeze_alert`.

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
from typing import Callable, List, Optional

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
