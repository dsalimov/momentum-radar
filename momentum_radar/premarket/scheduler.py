"""
scheduler.py – APScheduler-backed pre-market scan scheduler.

Schedules five daily pre-market briefing runs (all times in US/Eastern):

* 7:00 AM  – Early pre-market scan
* 8:00 AM  – Mid pre-market scan
* 9:00 AM  – Pre-open scan
* 9:20 AM  – Final pre-open briefing
* 9:25 AM  – Premarket Summary Report (most actionable movers before bell)

Usage::

    from momentum_radar.premarket.scheduler import start_scheduler, stop_scheduler
    scheduler = start_scheduler(tickers, fetcher, send_fn)
    # … later …
    stop_scheduler(scheduler)

The *send_fn* callable receives a single ``str`` argument (the formatted
brief) and is responsible for delivery (e.g. sending to Telegram).
"""

import logging
from typing import Callable, List, Optional

from momentum_radar.data.data_fetcher import BaseDataFetcher

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scheduler setup
# ---------------------------------------------------------------------------

def start_scheduler(
    tickers: List[str],
    fetcher: BaseDataFetcher,
    send_fn: Callable[[str], None],
) -> Optional[object]:
    """Create and start an APScheduler ``BackgroundScheduler``.

    Three cron jobs are registered (all in US/Eastern timezone):

    * 7:00 AM  – Early pre-market
    * 8:00 AM  – Mid pre-market
    * 9:00 AM  – Pre-open scan
    * 9:20 AM  – Final pre-open briefing
    * 9:25 AM  – Premarket Summary Report

    Args:
        tickers: Stock universe to include in the scan.
        fetcher: Data provider instance.
        send_fn: Callable that delivers the briefing string
                 (e.g. ``send_telegram_alert``).

    Returns:
        The started :class:`~apscheduler.schedulers.background.BackgroundScheduler`
        instance, or ``None`` if APScheduler is not installed.
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.warning(
            "APScheduler is not installed. Pre-market scheduling is disabled. "
            "Install it with: pip install apscheduler"
        )
        return None

    from momentum_radar.premarket.briefing import generate_market_brief

    scheduler = BackgroundScheduler(timezone="America/New_York")

    sessions = [
        ("07", "00", "Pre-Market 7:00 AM"),
        ("08", "00", "Pre-Market 8:00 AM"),
        ("09", "00", "Pre-Open Scan 9:00 AM"),
        ("09", "20", "Final Pre-Open 9:20 AM"),
        ("09", "25", "Premarket Summary Report 9:25 AM"),
    ]

    for hour, minute, label in sessions:
        session_label = label  # capture in closure

        def _job(lbl: str = session_label) -> None:
            logger.info("Running scheduled pre-market brief: %s", lbl)
            try:
                brief = generate_market_brief(tickers, fetcher, session_label=lbl)
                send_fn(brief)
            except Exception as exc:
                logger.error("Scheduled brief failed (%s): %s", lbl, exc)

        scheduler.add_job(
            _job,
            trigger=CronTrigger(hour=int(hour), minute=int(minute), timezone="America/New_York"),
            id=f"premarket_{hour}{minute}",
            replace_existing=True,
        )
        logger.info("Scheduled pre-market brief at %s:%s ET (%s)", hour, minute, label)

    scheduler.start()
    logger.info("Pre-market scheduler started.")
    return scheduler


def stop_scheduler(scheduler: Optional[object]) -> None:
    """Gracefully stop a running scheduler.

    Args:
        scheduler: Instance returned by :func:`start_scheduler`, or ``None``.
    """
    if scheduler is None:
        return
    try:
        scheduler.shutdown(wait=False)
        logger.info("Pre-market scheduler stopped.")
    except Exception as exc:
        logger.warning("Error stopping pre-market scheduler: %s", exc)
