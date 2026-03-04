"""
ui/discord_notifier.py – Discord webhook alert delivery.

Builds and sends a Discord embed object via the webhooks API.
No bot token needed – only a webhook URL is required.

Environment variables
---------------------
- ``DISCORD_WEBHOOK_URL`` – Full Discord webhook URL.

Usage::

    from momentum_radar.ui.discord_notifier import notify

    sent = notify(signal)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

import requests

from momentum_radar.strategies.base import StrategySignal
from momentum_radar.ui.embed_formatter import format_discord_embed

logger = logging.getLogger(__name__)

DISCORD_WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "")

_MAX_RETRIES: int = 3
_RETRY_DELAY: float = 2.0


def notify(
    signal: StrategySignal,
    webhook_url: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    bot_version: str = "1.0",
) -> bool:
    """Send a strategy alert embed to a Discord channel via webhook.

    Args:
        signal:      Evaluated strategy signal.
        webhook_url: Discord webhook URL.  Falls back to ``DISCORD_WEBHOOK_URL``
                     environment variable.
        timestamp:   Alert timestamp (defaults to ``datetime.now()``).
        bot_version: Version string shown in the embed footer.

    Returns:
        ``True`` if the embed was delivered successfully.
    """
    import time

    url = webhook_url or DISCORD_WEBHOOK_URL
    if not url:
        logger.warning(
            "Discord webhook URL not configured (DISCORD_WEBHOOK_URL). "
            "Skipping alert delivery."
        )
        return False

    embed = format_discord_embed(signal, timestamp=timestamp, bot_version=bot_version)
    payload = {"embeds": [embed]}

    delay = _RETRY_DELAY
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code in (200, 204):
                logger.info(
                    "Discord alert sent: %s | %s | score=%d",
                    signal.ticker, signal.strategy, signal.score,
                )
                return True
            logger.warning(
                "Discord webhook returned %d (attempt %d/%d): %s",
                resp.status_code, attempt, _MAX_RETRIES, resp.text[:200],
            )
        except requests.RequestException as exc:
            logger.warning(
                "Discord request failed (attempt %d/%d): %s",
                attempt, _MAX_RETRIES, exc,
            )
        if attempt < _MAX_RETRIES:
            time.sleep(delay)
            delay *= 2

    logger.error("Failed to send Discord alert after %d attempts.", _MAX_RETRIES)
    return False


def notify_daily_summary(
    signals: list,
    webhook_url: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> bool:
    """Send the end-of-session daily summary to Discord as a simple text message.

    Args:
        signals:     All signals generated during the session.
        webhook_url: Discord webhook URL.
        timestamp:   Summary timestamp.

    Returns:
        ``True`` if delivered successfully.
    """
    import time
    from momentum_radar.ui.embed_formatter import format_daily_summary

    url = webhook_url or DISCORD_WEBHOOK_URL
    if not url:
        logger.warning("Discord webhook URL not configured. Skipping daily summary.")
        return False

    summary = format_daily_summary(signals, timestamp=timestamp)
    payload = {"content": summary[:2000]}  # Discord text limit

    delay = _RETRY_DELAY
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code in (200, 204):
                return True
            logger.warning(
                "Discord summary returned %d (attempt %d/%d)",
                resp.status_code, attempt, _MAX_RETRIES,
            )
        except requests.RequestException as exc:
            logger.warning("Discord summary failed (attempt %d/%d): %s", attempt, _MAX_RETRIES, exc)
        if attempt < _MAX_RETRIES:
            time.sleep(delay)
            delay *= 2
    return False
