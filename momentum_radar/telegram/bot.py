"""
bot.py – Telegram bot listener for pattern research mode.

Users send a pattern name (e.g. "double bottom") and the bot replies with
annotated candlestick chart images for the top matches found by scanning
the stock universe.

Requires *python-telegram-bot* v20+ (async API).
"""

import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Recognised pattern names (must match detector registry keys)
# ---------------------------------------------------------------------------

_KNOWN_PATTERNS = [
    # Structure patterns
    "double bottom",
    "double top",
    "head and shoulders",
    "inverse head and shoulders",
    "bull flag",
    "bear flag",
    "cup and handle",
    "ascending triangle",
    "descending triangle",
    "symmetrical triangle",
    "rising wedge",
    "falling wedge",
    "pennant",
    "channel up",
    "channel down",
    "flat base",
    "broadening formation",
 <<<<<<< copilot/add-more-chart-patterns
=======
    # Candlestick patterns
    "hammer",
    "inverted hammer",
    "hanging man",
    "shooting star",
    "doji",
    "dragonfly doji",
    "gravestone doji",
    "bullish marubozu",
    "bearish marubozu",
    "spinning top",
    "bullish engulfing",
    "bearish engulfing",
    "bullish harami",
    "bearish harami",
    "tweezer top",
    "tweezer bottom",
    "piercing line",
    "dark cloud cover",
    "morning star",
    "evening star",
    "three white soldiers",
    "three black crows",
    "three inside up",
    "three inside down",
 >>>>>>> main
]

_HELP_TEXT = (
    "Pattern Research Bot\n\n"
    "Send a pattern name to scan 500+ stocks.\n\n"
    "Available patterns:\n"
    + "\n".join(f"  - {p}" for p in _KNOWN_PATTERNS)
    + "\n\n"
    "Or use /scan <pattern> - e.g. /scan double bottom\n"
    "Use /status to check bot health."
)


def _safe_text(text: str) -> str:
    """Strip Markdown special characters to send as plain text."""
    for char in ["*", "_", "`", "[", "]", "(", ")"]:
        text = text.replace(char, "")
    return text


async def start_telegram_bot() -> None:  # pragma: no cover
    """Start the Telegram bot for pattern research.

    Runs until interrupted.  Uses :mod:`python-telegram-bot` v20+ async API.
    """
    try:
        from telegram import Update
        from telegram.ext import (
            Application,
            CommandHandler,
            MessageHandler,
            ContextTypes,
            filters,
        )
    except ImportError as exc:
        raise ImportError(
            "python-telegram-bot is required for the Telegram bot. "
            "Install it with: pip install python-telegram-bot"
        ) from exc

    from momentum_radar.config import config
    from momentum_radar.data.data_fetcher import get_data_fetcher
    from momentum_radar.data.universe_builder import UniverseBuilder
    from momentum_radar.patterns.detector import scan_for_pattern, available_patterns
    from momentum_radar.patterns.charts import generate_pattern_chart

    bot_token = config.telegram.bot_token
    chat_id = config.telegram.chat_id

    if not bot_token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not configured.  "
            "Set it in your .env file or environment."
        )

    fetcher = get_data_fetcher(config.data.provider)
    universe_builder = UniverseBuilder(fetcher)
    logger.info("Building universe for pattern bot…")
    universe = universe_builder.build()
    logger.info("Universe ready: %d tickers", len(universe))

    async def _help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(_HELP_TEXT)

    async def _status_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            f"✅ Pattern bot is running.\n"
            f"Universe: {len(universe)} tickers\n"
            f"Available patterns: {len(available_patterns())}"
        )

    async def _scan_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        pattern = " ".join(context.args) if context.args else ""
        await _run_scan(update, context, pattern)

    async def _message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        text = (update.message.text or "").strip()
        await _run_scan(update, context, text)

    async def _run_scan(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        pattern_name: str,
    ) -> None:
        pattern_lower = pattern_name.strip().lower()
        if not pattern_lower or pattern_lower not in _KNOWN_PATTERNS:
            await update.message.reply_text(
                f"Unknown pattern: '{pattern_name}'\n\n{_HELP_TEXT}",
            )
            return

        await update.message.reply_text(
            f"Scanning {len(universe)}+ stocks for {pattern_name.title()} pattern..."
            f" This may take a minute.",
        )

        loop = asyncio.get_event_loop()
        matches = await loop.run_in_executor(
            None,
            lambda: scan_for_pattern(pattern_name, universe, fetcher, top_n=5),
        )

        if not matches:
            await update.message.reply_text(
                f"No strong matches found for {pattern_name.title()} "
                f"(confidence >= 60%) in the current universe.",
            )
            return

        await update.message.reply_text(
            f"Found {len(matches)} match(es) for {pattern_name.title()}:",
        )

        for match in matches:
            ticker = match["ticker"]
            df = match.get("df")
            confidence = match.get("confidence", 0)
            description = match.get("description", "")

            caption = (
                f"{ticker} - {pattern_name.title()}\n"
                f"Confidence: {confidence}%\n"
                f"{_safe_text(description)}"
            )

            if df is not None and not df.empty:
                try:
                    chart_path = await loop.run_in_executor(
                        None,
                        lambda: generate_pattern_chart(ticker, df, match),
                    )
                    with open(chart_path, "rb") as photo:
                        await context.bot.send_photo(
                            chat_id=update.effective_chat.id,
                            photo=photo,
                            caption=caption,
                        )
                    try:
                        os.remove(chart_path)
                    except OSError:
                        pass
                except Exception as exc:
                    logger.error("Chart generation failed for %s: %s", ticker, exc)
                    await update.message.reply_text(caption)
            else:
                await update.message.reply_text(caption)

    # Build and run the application
    app = Application.builder().token(bot_token).build()
    app.add_handler(CommandHandler("help", _help_handler))
    app.add_handler(CommandHandler("start", _help_handler))
    app.add_handler(CommandHandler("status", _status_handler))
    app.add_handler(CommandHandler("scan", _scan_command_handler))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, _message_handler)
    )

    logger.info("Starting Telegram pattern bot (polling)…")
    async with app:
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        logger.info("Bot is running. Press Ctrl+C to stop.")
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await app.updater.stop()
            await app.stop()
