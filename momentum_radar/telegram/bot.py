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
]

_HELP_TEXT = (
    "Pattern Research Bot\n\n"
    "Send a pattern name to scan 500+ stocks.\n\n"
    "Available patterns:\n"
    + "\n".join(f"  - {p}" for p in _KNOWN_PATTERNS)
    + "\n\n"
    "Or use /scan <pattern> - e.g. /scan double bottom\n\n"
    "Options Commands:\n"
    "  /options AAPL - Full options summary + chart\n"
    "  /flow AAPL - Options flow analysis (smart money)\n"
    "  /unusual - Scan for unusual options activity\n"
    "  /maxpain AAPL - Max pain calculation\n"
    "  /iv AAPL - Implied volatility analysis\n"
    "  /pcr AAPL - Put/call ratio\n\n"
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
        text_lower = text.lower()
        if text_lower.startswith("options "):
            ticker = text[len("options "):].strip().upper()
            await _options_handler_impl(update, context, ticker)
        elif text_lower.startswith("flow "):
            ticker = text[len("flow "):].strip().upper()
            await _flow_handler_impl(update, context, ticker)
        elif text_lower == "unusual":
            await _unusual_handler_impl(update, context)
        elif text_lower.startswith("maxpain "):
            ticker = text[len("maxpain "):].strip().upper()
            await _maxpain_handler_impl(update, context, ticker)
        elif text_lower.startswith("iv "):
            ticker = text[len("iv "):].strip().upper()
            await _iv_handler_impl(update, context, ticker)
        elif text_lower.startswith("pcr "):
            ticker = text[len("pcr "):].strip().upper()
            await _pcr_handler_impl(update, context, ticker)
        else:
            await _run_scan(update, context, text)

    async def _options_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        ticker = " ".join(context.args).strip().upper() if context.args else ""
        await _options_handler_impl(update, context, ticker)

    async def _flow_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        ticker = " ".join(context.args).strip().upper() if context.args else ""
        await _flow_handler_impl(update, context, ticker)

    async def _unusual_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await _unusual_handler_impl(update, context)

    async def _maxpain_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        ticker = " ".join(context.args).strip().upper() if context.args else ""
        await _maxpain_handler_impl(update, context, ticker)

    async def _iv_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        ticker = " ".join(context.args).strip().upper() if context.args else ""
        await _iv_handler_impl(update, context, ticker)

    async def _pcr_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        ticker = " ".join(context.args).strip().upper() if context.args else ""
        await _pcr_handler_impl(update, context, ticker)

    async def _options_handler_impl(
        update: Update, context: ContextTypes.DEFAULT_TYPE, ticker: str
    ) -> None:
        if not ticker:
            await update.message.reply_text("Usage: /options AAPL")
            return
        await update.message.reply_text(f"Fetching options data for {ticker}...")
        loop = asyncio.get_event_loop()
        try:
            from momentum_radar.options.options_analyzer import get_options_summary
            from momentum_radar.options.options_charts import generate_volume_chart
            summary = await loop.run_in_executor(None, lambda: get_options_summary(ticker))
        except Exception as exc:
            logger.error("Options summary failed for %s: %s", ticker, exc)
            await update.message.reply_text(f"Could not fetch options data for {ticker}. Make sure it's a valid US stock ticker.")
            return

        if not summary:
            await update.message.reply_text(f"No options data available for {ticker}")
            return

        current_price = summary.get("current_price", 0)
        pc_ratio = summary.get("put_call_ratio", 0)
        pc_interp = summary.get("put_call_interpretation", "")
        max_pain = summary.get("max_pain_strike")
        total_call_vol = summary.get("total_call_volume", 0)
        total_put_vol = summary.get("total_put_volume", 0)
        top_calls = summary.get("most_active_calls", [])
        top_puts = summary.get("most_active_puts", [])

        lines = [
            f"Options Summary: {ticker}",
            "",
            f"Current Price: ${current_price:.2f}",
            f"Put/Call Ratio: {pc_ratio:.2f} ({pc_interp})",
        ]
        if max_pain is not None:
            lines.append(f"Max Pain: ${max_pain:.2f} (price tends to gravitate here by expiry)")
        lines += [
            "",
            f"Total Call Volume: {total_call_vol:,}",
            f"Total Put Volume: {total_put_vol:,}",
        ]
        if top_calls:
            lines.append("\nMost Active Calls:")
            for c in top_calls[:5]:
                lines.append(
                    f"  ${c['strike']:.0f} Call ({c['expiry']}) - "
                    f"Vol: {int(c['volume']):,} | OI: {int(c['openInterest']):,} | "
                    f"IV: {c['impliedVolatility']*100:.1f}%"
                )
        if top_puts:
            lines.append("\nMost Active Puts:")
            for p in top_puts[:5]:
                lines.append(
                    f"  ${p['strike']:.0f} Put ({p['expiry']}) - "
                    f"Vol: {int(p['volume']):,} | OI: {int(p['openInterest']):,} | "
                    f"IV: {p['impliedVolatility']*100:.1f}%"
                )

        msg = _safe_text("\n".join(lines))
        await update.message.reply_text(msg)

        try:
            chart_path = await loop.run_in_executor(
                None, lambda: generate_volume_chart(ticker, summary)
            )
            with open(chart_path, "rb") as photo:
                await context.bot.send_photo(
                    chat_id=update.effective_chat.id, photo=photo
                )
            try:
                os.remove(chart_path)
            except OSError:
                pass
        except Exception as exc:
            logger.error("Options chart failed for %s: %s", ticker, exc)

    async def _flow_handler_impl(
        update: Update, context: ContextTypes.DEFAULT_TYPE, ticker: str
    ) -> None:
        if not ticker:
            await update.message.reply_text("Usage: /flow AAPL")
            return
        await update.message.reply_text(f"Analyzing options flow for {ticker}...")
        loop = asyncio.get_event_loop()
        try:
            from momentum_radar.options.options_analyzer import get_options_flow
            flow = await loop.run_in_executor(None, lambda: get_options_flow(ticker))
        except Exception as exc:
            logger.error("Options flow failed for %s: %s", ticker, exc)
            await update.message.reply_text(f"Could not fetch options flow for {ticker}. Make sure it's a valid US stock ticker.")
            return

        if not flow:
            await update.message.reply_text(f"No options data available for {ticker}")
            return

        net_sentiment = flow.get("net_sentiment", "NEUTRAL").upper()
        call_vol = flow.get("total_call_volume", 0)
        put_vol = flow.get("total_put_volume", 0)
        dollar_calls = flow.get("dollar_call_flow", 0)
        dollar_puts = flow.get("dollar_put_flow", 0)
        call_sweeps = flow.get("call_sweeps", [])
        put_sweeps = flow.get("put_sweeps", [])

        lines = [
            f"Options Flow: {ticker}",
            "",
            f"Net Sentiment: {net_sentiment}",
            f"Call Volume: {call_vol:,} | Put Volume: {put_vol:,}",
            f"Dollar Flow: ${dollar_calls/1e6:.1f}M calls vs ${dollar_puts/1e6:.1f}M puts",
        ]

        unusual = call_sweeps[:3] + put_sweeps[:3]
        unusual.sort(key=lambda x: x.get("volume", 0), reverse=True)
        if unusual:
            lines.append("\nUnusual Activity:")
            for u in unusual[:5]:
                opt_type = "Call" if u.get("type") == "call" else "Put"
                ratio = u.get("vol_oi_ratio", 0)
                lines.append(
                    f"  ${u['strike']:.0f} {opt_type} {u['expiry']} - "
                    f"{int(u['volume']):,} vol ({ratio:.1f}x OI) - Possible sweep"
                )

        smart_signals = flow.get("smart_money_signals", [])
        if smart_signals:
            lines.append("\nSmart Money Signals:")
            for s in smart_signals[:3]:
                lines.append(f"  {s}")

        msg = _safe_text("\n".join(lines))
        await update.message.reply_text(msg)

    async def _unusual_handler_impl(
        update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        await update.message.reply_text("Scanning for unusual options activity...")
        loop = asyncio.get_event_loop()
        try:
            from momentum_radar.options.options_analyzer import scan_unusual_volume
            results = await loop.run_in_executor(
                None, lambda: scan_unusual_volume(universe[:50], top_n=10)
            )
        except Exception as exc:
            logger.error("Unusual volume scan failed: %s", exc)
            await update.message.reply_text("Unusual options scan failed. Please try again later.")
            return

        if not results:
            await update.message.reply_text("No unusual options activity detected.")
            return

        lines = ["Unusual Options Activity Scan", ""]
        for i, item in enumerate(results[:10], 1):
            opt_type = "Call" if item.get("type") == "call" else "Put"
            ratio = item.get("vol_oi_ratio", 0)
            lines.append(
                f"{i}. {item['ticker']} ${item['strike']:.0f} {opt_type} ({item['expiry']}) - "
                f"Vol: {int(item['volume']):,} | OI: {int(item['openInterest']):,} | "
                f"Ratio: {ratio:.1f}x"
            )

        msg = _safe_text("\n".join(lines))
        await update.message.reply_text(msg)

    async def _maxpain_handler_impl(
        update: Update, context: ContextTypes.DEFAULT_TYPE, ticker: str
    ) -> None:
        if not ticker:
            await update.message.reply_text("Usage: /maxpain AAPL")
            return
        await update.message.reply_text(f"Calculating max pain for {ticker}...")
        loop = asyncio.get_event_loop()
        try:
            from momentum_radar.options.options_analyzer import get_max_pain
            result = await loop.run_in_executor(None, lambda: get_max_pain(ticker))
        except Exception as exc:
            logger.error("Max pain failed for %s: %s", ticker, exc)
            await update.message.reply_text(f"Could not calculate max pain for {ticker}. Make sure it's a valid US stock ticker.")
            return

        if not result:
            await update.message.reply_text(f"No options data available for {ticker}")
            return

        expiry = result.get("expiry", "N/A")
        max_pain_strike = result.get("max_pain_strike", 0)
        current_price = result.get("current_price", 0)
        distance = result.get("distance", 0)
        pct_distance = result.get("percentage_distance", 0)

        if current_price > max_pain_strike:
            direction = "ABOVE max pain - slight downward pressure expected"
        elif current_price < max_pain_strike:
            direction = "BELOW max pain - slight upward pressure expected"
        else:
            direction = "AT max pain"

        lines = [
            f"Max Pain: {ticker}",
            "",
            f"Expiry: {expiry}",
            f"Max Pain Strike: ${max_pain_strike:.2f}",
            f"Current Price: ${current_price:.2f}",
            f"Distance: ${abs(distance):.2f} ({abs(pct_distance):.1f}% {'above' if distance > 0 else 'below'} max pain)",
            "",
            "Price tends to gravitate toward max pain by expiration.",
            f"Current price is {direction}.",
        ]

        msg = _safe_text("\n".join(lines))
        await update.message.reply_text(msg)

    async def _iv_handler_impl(
        update: Update, context: ContextTypes.DEFAULT_TYPE, ticker: str
    ) -> None:
        if not ticker:
            await update.message.reply_text("Usage: /iv AAPL")
            return
        await update.message.reply_text(f"Analyzing implied volatility for {ticker}...")
        loop = asyncio.get_event_loop()
        try:
            from momentum_radar.options.options_analyzer import get_iv_analysis
            result = await loop.run_in_executor(None, lambda: get_iv_analysis(ticker))
        except Exception as exc:
            logger.error("IV analysis failed for %s: %s", ticker, exc)
            await update.message.reply_text(f"Could not fetch IV data for {ticker}. Make sure it's a valid US stock ticker.")
            return

        if not result:
            await update.message.reply_text(f"No options data available for {ticker}")
            return

        atm_iv = result.get("atm_iv", 0)
        skew = result.get("skew_description", "N/A")
        term_structure = result.get("term_structure_description", "N/A")
        high_iv = result.get("highest_iv_contracts", [])

        lines = [
            f"IV Analysis: {ticker}",
            "",
            f"ATM IV (30-day): {atm_iv*100:.1f}%",
            f"IV Skew: {skew}",
            f"Term Structure: {term_structure}",
        ]

        if high_iv:
            lines.append("\nHighest IV Strikes:")
            for c in high_iv[:5]:
                opt_type = "Call" if c.get("type") == "call" else "Put"
                lines.append(
                    f"  ${c['strike']:.0f} {opt_type} {c['expiry']} - "
                    f"IV: {c['impliedVolatility']*100:.1f}%"
                )

        msg = _safe_text("\n".join(lines))
        await update.message.reply_text(msg)

    async def _pcr_handler_impl(
        update: Update, context: ContextTypes.DEFAULT_TYPE, ticker: str
    ) -> None:
        if not ticker:
            await update.message.reply_text("Usage: /pcr AAPL")
            return
        await update.message.reply_text(f"Calculating put/call ratio for {ticker}...")
        loop = asyncio.get_event_loop()
        try:
            from momentum_radar.options.options_analyzer import get_put_call_ratio
            result = await loop.run_in_executor(None, lambda: get_put_call_ratio(ticker))
        except Exception as exc:
            logger.error("PCR failed for %s: %s", ticker, exc)
            await update.message.reply_text(f"Could not fetch put/call ratio for {ticker}. Make sure it's a valid US stock ticker.")
            return

        if not result:
            await update.message.reply_text(f"No options data available for {ticker}")
            return

        vol_pcr = result.get("volume_pc_ratio", 0)
        oi_pcr = result.get("oi_pc_ratio", 0)
        interpretation = result.get("interpretation", "N/A")
        description = result.get("description", "")

        lines = [
            f"Put/Call Ratio: {ticker}",
            "",
            f"Volume PC Ratio: {vol_pcr:.2f}",
            f"OI PC Ratio: {oi_pcr:.2f}",
            "",
            f"Interpretation: {interpretation}",
        ]
        if description:
            lines.append(description)
        if vol_pcr < 0.7:
            lines.append("Note: PC ratio below 0.7 can signal excessive bullishness (contrarian bearish).")
        elif vol_pcr > 1.5:
            lines.append("Note: PC ratio above 1.5 signals extreme fear (contrarian bullish signal).")

        msg = _safe_text("\n".join(lines))
        await update.message.reply_text(msg)

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
    app.add_handler(CommandHandler("options", _options_handler))
    app.add_handler(CommandHandler("flow", _flow_handler))
    app.add_handler(CommandHandler("unusual", _unusual_handler))
    app.add_handler(CommandHandler("maxpain", _maxpain_handler))
    app.add_handler(CommandHandler("iv", _iv_handler))
    app.add_handler(CommandHandler("pcr", _pcr_handler))
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
