"""
discord_alert.py – Discord alert delivery (trade setup format).

Provides :func:`send_discord_alert` as a placeholder function that can be
wired to a Discord webhook once the integration is ready.

The function signature is intentionally simple so that callers do not need
to change when full Discord support is added later.

Environment variable
--------------------
``DISCORD_WEBHOOK_URL`` – set this to enable live delivery.

Usage::

    from momentum_radar.alerts.discord_alert import send_discord_alert

    send_discord_alert(message="🚨 TRADE SETUP ...", image_path="/tmp/chart.png")
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

#: Discord webhook URL loaded from the environment.  Empty string = disabled.
DISCORD_WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "")


def send_discord_alert(
    message: str,
    image_path: Optional[str] = None,
    webhook_url: Optional[str] = None,
) -> bool:
    """Send a trade-setup alert to Discord via webhook.

    When ``DISCORD_WEBHOOK_URL`` is not configured the function logs a debug
    message and returns ``False`` without raising an exception, so callers can
    safely invoke it unconditionally.

    When a webhook URL *is* configured the function posts the message as the
    ``content`` field and, if *image_path* is provided, attaches the file as a
    multipart upload.

    Args:
        message:     The formatted alert text (plain text or Markdown).
        image_path:  Optional path to a PNG/JPEG chart image to attach.
        webhook_url: Override URL (falls back to ``DISCORD_WEBHOOK_URL`` env var).

    Returns:
        ``True`` if the message was delivered successfully, ``False`` otherwise.
    """
    url = webhook_url or DISCORD_WEBHOOK_URL
    if not url:
        logger.debug(
            "Discord webhook not configured (DISCORD_WEBHOOK_URL unset). "
            "Skipping trade setup alert."
        )
        return False

    try:
        import requests
    except ImportError:
        logger.warning("requests library not available; cannot send Discord alert.")
        return False

    try:
        if image_path and os.path.isfile(image_path):
            with open(image_path, "rb") as fh:
                resp = requests.post(
                    url,
                    data={"content": message[:2000]},
                    files={"file": (os.path.basename(image_path), fh, "image/png")},
                    timeout=10,
                )
        else:
            resp = requests.post(
                url,
                json={"content": message[:2000]},
                timeout=10,
            )

        if resp.status_code in (200, 204):
            logger.info("Discord trade-setup alert sent successfully.")
            return True

        logger.warning(
            "Discord webhook returned HTTP %d: %s",
            resp.status_code,
            resp.text[:200],
        )
        return False

    except Exception as exc:
        logger.warning("Discord alert delivery failed: %s", exc)
        return False
