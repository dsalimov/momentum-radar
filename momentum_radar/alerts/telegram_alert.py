"""
telegram_alert.py – Telegram Bot API delivery for formatted alerts.

Alerts are delivered to the configured chat using the Bot API
``sendMessage`` endpoint.  Transient send failures are retried with
exponential back-off.
"""

import logging
import time
from typing import Optional

import requests

from momentum_radar.config import config

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"


def send_telegram_alert(message: str) -> bool:
    """Send *message* to the configured Telegram chat.

    Uses :attr:`~momentum_radar.config.TelegramConfig.bot_token` and
    :attr:`~momentum_radar.config.TelegramConfig.chat_id` from the global
    config.  Retries up to :attr:`~momentum_radar.config.TelegramConfig.max_retries`
    times on failure, with exponential back-off.

    Args:
        message: Text message to send (plain text or Markdown).

    Returns:
        ``True`` if the message was delivered successfully, ``False`` otherwise.
    """
    cfg = config.telegram
    if not cfg.bot_token or not cfg.chat_id:
        logger.warning(
            "Telegram not configured (missing BOT_TOKEN or CHAT_ID). "
            "Skipping alert delivery."
        )
        return False

    url = f"{TELEGRAM_API_BASE}/bot{cfg.bot_token}/sendMessage"
    payload = {
        "chat_id": cfg.chat_id,
        "text": message,
        "parse_mode": "HTML",
    }

    delay = cfg.retry_delay
    for attempt in range(1, cfg.max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                logger.info("Telegram alert sent successfully.")
                return True
            logger.warning(
                "Telegram API returned %d (attempt %d/%d): %s",
                resp.status_code,
                attempt,
                cfg.max_retries,
                resp.text[:200],
            )
        except requests.RequestException as exc:
            logger.warning(
                "Telegram request failed (attempt %d/%d): %s",
                attempt,
                cfg.max_retries,
                exc,
            )
        if attempt < cfg.max_retries:
            time.sleep(delay)
            delay *= 2  # exponential back-off

    logger.error("Failed to send Telegram alert after %d attempts.", cfg.max_retries)
    return False


def send_telegram_photo(image_path: str, caption: str) -> bool:
    """Send a photo to the configured Telegram chat.

    Uses the Bot API ``sendPhoto`` endpoint with multipart form data.

    Args:
        image_path: Absolute path to the PNG file to send.
        caption:    Text caption for the image.

    Returns:
        ``True`` if the photo was delivered successfully, ``False`` otherwise.
    """
    cfg = config.telegram
    if not cfg.bot_token or not cfg.chat_id:
        logger.warning(
            "Telegram not configured (missing BOT_TOKEN or CHAT_ID). "
            "Skipping photo delivery."
        )
        return False

    url = f"{TELEGRAM_API_BASE}/bot{cfg.bot_token}/sendPhoto"
    delay = cfg.retry_delay
    for attempt in range(1, cfg.max_retries + 1):
        try:
            with open(image_path, "rb") as photo_file:
                resp = requests.post(
                    url,
                    data={"chat_id": cfg.chat_id, "caption": caption},
                    files={"photo": photo_file},
                    timeout=30,
                )
            if resp.status_code == 200:
                logger.info("Telegram photo sent successfully.")
                return True
            logger.warning(
                "Telegram API returned %d (attempt %d/%d): %s",
                resp.status_code,
                attempt,
                cfg.max_retries,
                resp.text[:200],
            )
        except requests.RequestException as exc:
            logger.warning(
                "Telegram photo request failed (attempt %d/%d): %s",
                attempt,
                cfg.max_retries,
                exc,
            )
        if attempt < cfg.max_retries:
            time.sleep(delay)
            delay *= 2

    logger.error("Failed to send Telegram photo after %d attempts.", cfg.max_retries)
    return False
