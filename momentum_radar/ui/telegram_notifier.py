"""
ui/telegram_notifier.py – Telegram alert delivery using professional embed cards.

Wraps :mod:`momentum_radar.alerts.telegram_alert` and combines it with
:mod:`momentum_radar.ui.embed_formatter` to deliver strategy-grade alert cards.

Usage::

    from momentum_radar.ui.telegram_notifier import notify

    sent = notify(signal)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from momentum_radar.alerts.telegram_alert import send_telegram_alert, send_telegram_photo
from momentum_radar.strategies.base import StrategySignal
from momentum_radar.ui.embed_formatter import format_telegram_card

logger = logging.getLogger(__name__)


def notify(
    signal: StrategySignal,
    chart_path: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> bool:
    """Send a formatted strategy alert to the configured Telegram chat.

    When *chart_path* is provided the chart image is sent first as a photo
    with the card text as the caption.  Otherwise only the text card is sent.

    Args:
        signal:     Evaluated strategy signal.
        chart_path: Optional path to a chart PNG file.
        timestamp:  Alert timestamp (defaults to ``datetime.now()``).

    Returns:
        ``True`` if the message was delivered successfully.
    """
    card = format_telegram_card(signal, timestamp=timestamp)

    if chart_path:
        # Send chart image with card text as caption (Telegram limits: 1024 chars)
        caption = card[:1024]
        success = send_telegram_photo(chart_path, caption=caption)
        if not success:
            # Fall back to text-only on photo failure
            logger.warning("Photo send failed for %s – falling back to text", signal.ticker)
            success = send_telegram_alert(card)
    else:
        success = send_telegram_alert(card)

    if success:
        logger.info(
            "Telegram alert sent: %s | %s | score=%d",
            signal.ticker, signal.strategy, signal.score,
        )
    else:
        logger.warning("Failed to send Telegram alert for %s", signal.ticker)

    return success


def notify_daily_summary(
    signals: list,
    timestamp: Optional[datetime] = None,
) -> bool:
    """Send the end-of-session daily summary card.

    Args:
        signals:   All signals generated during the session.
        timestamp: Summary timestamp (defaults to ``datetime.now()``).

    Returns:
        ``True`` if the summary was delivered successfully.
    """
    from momentum_radar.ui.embed_formatter import format_daily_summary

    card = format_daily_summary(signals, timestamp=timestamp)
    return send_telegram_alert(card)
