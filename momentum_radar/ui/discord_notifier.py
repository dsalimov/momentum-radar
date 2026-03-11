"""
ui/discord_notifier.py – Discord webhook alert delivery.

Builds and sends a Discord embed object via the webhooks API.
No bot token needed – only a webhook URL is required.

Environment variables
---------------------
- ``DISCORD_WEBHOOK_URL`` – Full Discord webhook URL.

Public API
----------
- :func:`send_discord_alert`  – send a plain-text alert (+ optional image)
- :func:`notify`              – send a :class:`StrategySignal` embed
- :func:`notify_trade_setup`  – send a :class:`TradeSetup` text alert
- :func:`notify_daily_summary`– send an end-of-session summary

Usage::

    from momentum_radar.ui.discord_notifier import send_discord_alert, notify

    # Plain-text alert with optional chart image
    send_discord_alert("🚨 GOLDEN SWEEP ALERT\\nTicker: TSLA ...", image=chart_bytes)

    # Full embed from a StrategySignal
    notify(signal)
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

# Discord API limit for a single message content field
_DISCORD_TEXT_LIMIT: int = 2000


# ---------------------------------------------------------------------------
# Internal HTTP helper
# ---------------------------------------------------------------------------

def _post_with_retry(
    url: str,
    payload: dict,
    files: Optional[dict] = None,
) -> bool:
    """POST *payload* to *url* with exponential-backoff retry.

    Args:
        url:     Discord webhook URL.
        payload: JSON-serialisable request body.
        files:   Optional ``requests``-style file dict for multipart upload.

    Returns:
        ``True`` if the request succeeded (HTTP 200 or 204).
    """
    import time

    delay = _RETRY_DELAY
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            if files:
                resp = requests.post(url, data=payload, files=files, timeout=15)
            else:
                resp = requests.post(url, json=payload, timeout=10)

            if resp.status_code in (200, 204):
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

    logger.error("Failed to deliver Discord alert after %d attempts.", _MAX_RETRIES)
    return False


def _build_message_payload(content: str, image: Optional[bytes]) -> tuple[dict, Optional[dict]]:
    """Build the request payload and optional files dict for a text + image post.

    Returns:
        ``(payload, files)`` where *files* is ``None`` for text-only posts.
    """
    if image is not None:
        import json
        payload = {"payload_json": json.dumps({"content": content})}
        files: Optional[dict] = {"file": ("chart.png", image, "image/png")}
    else:
        payload = {"content": content}
        files = None
    return payload, files


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def send_discord_alert(
    message: str,
    image: Optional[bytes] = None,
    webhook_url: Optional[str] = None,
) -> bool:
    """Send a plain-text alert to Discord, with an optional chart image.

    This is the primary integration point for Golden Sweep, trade setup,
    and autonomous pattern alerts.  Pass the PNG bytes of a chart to attach
    it as a file upload alongside the text message.

    Args:
        message:     Alert text (max :data:`_DISCORD_TEXT_LIMIT` chars).
        image:       Raw PNG/JPEG bytes of an annotated chart image, or
                     ``None`` to send text only.
        webhook_url: Discord webhook URL.  Defaults to the
                     ``DISCORD_WEBHOOK_URL`` environment variable.

    Returns:
        ``True`` if the alert was delivered successfully.

    Example::

        from momentum_radar.ui.discord_notifier import send_discord_alert

        # Text-only
        send_discord_alert("🚨 GOLDEN SWEEP ALERT\\nTicker: TSLA")

        # With chart image
        with open("chart.png", "rb") as fh:
            send_discord_alert("🚨 TRADE ALERT", image=fh.read())

    .. note::
        Set the ``DISCORD_WEBHOOK_URL`` environment variable or pass the URL
        explicitly.  When no URL is configured the function logs a warning
        and returns ``False`` without raising an exception.
    """
    url = webhook_url or DISCORD_WEBHOOK_URL
    if not url:
        logger.warning(
            "send_discord_alert: DISCORD_WEBHOOK_URL not configured – skipping delivery."
        )
        return False

    truncated = message[:_DISCORD_TEXT_LIMIT]
    payload, files = _build_message_payload(truncated, image)
    ok = _post_with_retry(url, payload=payload, files=files)

    if ok:
        logger.info("send_discord_alert: delivered to Discord (%d chars)", len(truncated))
    return ok


def notify_trade_setup(
    setup,
    webhook_url: Optional[str] = None,
    image: Optional[bytes] = None,
    timestamp: Optional[datetime] = None,
) -> bool:
    """Format and send a :class:`~momentum_radar.signals.setup_detector.TradeSetup`
    alert to Discord.

    Args:
        setup:       :class:`TradeSetup` from the setup detector.
        webhook_url: Discord webhook URL.
        image:       Optional chart image bytes.
        timestamp:   Alert timestamp override.

    Returns:
        ``True`` if delivered successfully.
    """
    from momentum_radar.alerts.trade_formatter import format_trade_setup

    message = format_trade_setup(setup, timestamp=timestamp)
    return send_discord_alert(message, image=image, webhook_url=webhook_url)


def notify_golden_sweep(
    sweep,
    webhook_url: Optional[str] = None,
    image: Optional[bytes] = None,
    timestamp: Optional[datetime] = None,
) -> bool:
    """Format and send a :class:`~momentum_radar.signals.golden_sweep.SweepAlert`
    to Discord using the blueprint Golden Sweep format.

    Args:
        sweep:       :class:`SweepAlert` from the golden sweep detector.
        webhook_url: Discord webhook URL.
        image:       Optional chart image bytes.
        timestamp:   Alert timestamp override.

    Returns:
        ``True`` if delivered successfully.
    """
    from momentum_radar.alerts.golden_sweep_formatter import format_golden_sweep_alert

    message = format_golden_sweep_alert(sweep, timestamp=timestamp)
    return send_discord_alert(message, image=image, webhook_url=webhook_url)


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
    url = webhook_url or DISCORD_WEBHOOK_URL
    if not url:
        logger.warning(
            "Discord webhook URL not configured (DISCORD_WEBHOOK_URL). "
            "Skipping alert delivery."
        )
        return False

    embed = format_discord_embed(signal, timestamp=timestamp, bot_version=bot_version)
    ok = _post_with_retry(url, payload={"embeds": [embed]})
    if ok:
        logger.info(
            "Discord alert sent: %s | %s | score=%d",
            signal.ticker, signal.strategy, signal.score,
        )
    return ok


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
    from momentum_radar.ui.embed_formatter import format_daily_summary

    url = webhook_url or DISCORD_WEBHOOK_URL
    if not url:
        logger.warning("Discord webhook URL not configured. Skipping daily summary.")
        return False

    summary = format_daily_summary(signals, timestamp=timestamp)
    return _post_with_retry(url, payload={"content": summary[:_DISCORD_TEXT_LIMIT]})

