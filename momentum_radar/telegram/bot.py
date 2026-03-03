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
    "Volume Commands:\n"
    "  /volspike - Scan for unusual volume vs 30-day average (top 15)\n"
    "  /analyze AAPL - Full institutional-level analysis + AI summary\n\n"
    "Pre-Market Intelligence:\n"
    "  /premarket - Run pre-market scan (unusual vol + most active volume leaders + options spikes)\n"
    "  /squeeze [AAPL] - Short squeeze candidates or single-ticker squeeze report\n"
    "  /brief - Generate daily market intelligence brief\n"
    "  /morningbrief - Comprehensive morning briefing (futures, earnings, sector, key levels)\n\n"
    "Fundamentals & Earnings:\n"
    "  /fundamentals AAPL - Income statement, cash flow, assets & liabilities\n"
    "  /earnings AAPL - Earnings history, EPS beat/miss trend + AI guidance summary\n\n"
    "News & Sentiment:\n"
    "  /news AAPL - Latest news for a specific ticker with AI sentiment summary\n"
    "  /news market - Market-wide news with AI sentiment summary\n"
    "  /news premarket - Pre-market focused news scan\n"
    "  /marketnews - Full market-wide news search with AI summary\n"
    "  /sentiment - Market sentiment engine (regime + confidence score)\n\n"
    "Market Calendar:\n"
    "  /dates - Weekly economic calendar (CPI, NFP, FOMC, earnings, etc.)\n\n"
    "Automated Alerts:\n"
    "  /alerts on     - Enable hourly squeeze + signal alerts\n"
    "  /alerts off    - Disable automated alerts\n"
    "  /alerts status - Show your current alert preference\n\n"
    "Market Heatmap:\n"
    "  /heatmap - Live market sector heatmap (color-coded by performance)\n\n"
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
        elif text_lower == "volspike":
            await _volspike_handler_impl(update, context)
        elif text_lower.startswith("analyze "):
            ticker = text[len("analyze "):].strip().upper()
            await _analyze_handler_impl(update, context, ticker)
        elif text_lower == "premarket":
            await _premarket_handler_impl(update, context)
        elif text_lower.startswith("squeeze"):
            ticker = text[len("squeeze"):].strip().upper()
            await _squeeze_handler_impl(update, context, ticker)
        elif text_lower == "brief":
            await _brief_handler_impl(update, context)
        elif text_lower == "morningbrief":
            await _morningbrief_handler_impl(update, context)
        elif text_lower.startswith("news "):
            arg = text[len("news "):].strip()
            await _news_handler_impl(update, context, arg)
        elif text_lower == "news":
            await _news_handler_impl(update, context, "")
        elif text_lower == "heatmap":
            await _heatmap_handler_impl(update, context)
        elif text_lower == "marketnews":
            await _marketnews_handler_impl(update, context)
        elif text_lower == "sentiment":
            await _sentiment_handler_impl(update, context)
        elif text_lower == "dates":
            await _dates_handler_impl(update, context)
        elif text_lower.startswith("fundamentals "):
            ticker = text[len("fundamentals "):].strip().upper()
            await _fundamentals_handler_impl(update, context, ticker)
        elif text_lower.startswith("earnings "):
            ticker = text[len("earnings "):].strip().upper()
            await _earnings_handler_impl(update, context, ticker)
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

    async def _volspike_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await _volspike_handler_impl(update, context)

    async def _analyze_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        ticker = " ".join(context.args).strip().upper() if context.args else ""
        await _analyze_handler_impl(update, context, ticker)

    async def _premarket_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await _premarket_handler_impl(update, context)

    async def _squeeze_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        ticker = " ".join(context.args).strip().upper() if context.args else ""
        await _squeeze_handler_impl(update, context, ticker)

    async def _brief_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await _brief_handler_impl(update, context)

    async def _morningbrief_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await _morningbrief_handler_impl(update, context)

    async def _news_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        arg = " ".join(context.args).strip() if context.args else ""
        await _news_handler_impl(update, context, arg)

    async def _marketnews_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await _marketnews_handler_impl(update, context)

    async def _sentiment_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await _sentiment_handler_impl(update, context)

    async def _dates_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await _dates_handler_impl(update, context)

    async def _fundamentals_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        ticker = " ".join(context.args).strip().upper() if context.args else ""
        await _fundamentals_handler_impl(update, context, ticker)

    async def _earnings_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        ticker = " ".join(context.args).strip().upper() if context.args else ""
        await _earnings_handler_impl(update, context, ticker)

    async def _heatmap_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await _heatmap_handler_impl(update, context)

    async def _alerts_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await _alerts_handler_impl(update, context)

    async def _alerts_handler_impl(
        update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /alerts on|off|status command."""
        from momentum_radar.storage.database import (
            get_alert_preference,
            set_alert_preference,
        )

        chat_id = str(update.effective_chat.id)
        arg = (" ".join(context.args) if context.args else "").strip().lower()

        if arg == "on":
            set_alert_preference(chat_id, True)
            await update.message.reply_text(
                "✅ Automated squeeze alerts ENABLED.\n"
                "You will receive up to 5 alerts per hour when high-probability "
                "setups are detected."
            )
        elif arg == "off":
            set_alert_preference(chat_id, False)
            await update.message.reply_text(
                "🔕 Automated squeeze alerts DISABLED.\n"
                "You can re-enable them with /alerts on"
            )
        elif arg == "status":
            enabled = get_alert_preference(chat_id)
            status_str = "ENABLED ✅" if enabled else "DISABLED 🔕"
            await update.message.reply_text(
                f"Automated alerts: {status_str}\n\n"
                "Commands:\n"
                "  /alerts on    – enable automated alerts\n"
                "  /alerts off   – disable automated alerts\n"
                "  /alerts status – show current setting"
            )
        else:
            await update.message.reply_text(
                "Usage:\n"
                "  /alerts on     – enable automated hourly squeeze alerts\n"
                "  /alerts off    – disable automated alerts\n"
                "  /alerts status – show current alert preference"
            )

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

    async def _volspike_handler_impl(
        update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        await update.message.reply_text(
            f"Scanning {len(universe)} stocks for unusual volume vs 30-day average..."
            " This may take a minute."
        )
        loop = asyncio.get_event_loop()
        try:
            from momentum_radar.data.volume_scanner import (
                scan_volume_spikes,
                generate_volume_spike_chart,
            )
            spikes = await loop.run_in_executor(
                None, lambda: scan_volume_spikes(universe, fetcher, top_n=15)
            )
        except Exception as exc:
            logger.error("Volume spike scan failed: %s", exc)
            await update.message.reply_text("Volume spike scan failed. Please try again later.")
            return

        if not spikes:
            await update.message.reply_text("No unusual volume spikes detected in the current universe.")
            return

        # Text summary
        lines = [f"Volume Spike Scanner — Top {len(spikes)} (vs 30-Day Avg)", ""]
        for i, s in enumerate(spikes, 1):
            direction = "▲" if s["pct_change"] >= 0 else "▼"
            vol_str = (
                f"{s['today_volume']/1e6:.1f}M"
                if s["today_volume"] >= 1_000_000
                else f"{s['today_volume']/1e3:.0f}K"
            )
            lines.append(
                f"{i:2d}. {s['ticker']:6s}  RVOL {s['rvol']:.1f}x  "
                f"{direction}{abs(s['pct_change']):.1f}%  "
                f"${s['last_close']:.2f}  Vol {vol_str}"
            )

        msg = _safe_text("\n".join(lines))
        await update.message.reply_text(msg)

        # Chart image
        try:
            chart_path = await loop.run_in_executor(
                None, lambda: generate_volume_spike_chart(spikes)
            )
            with open(chart_path, "rb") as photo:
                await context.bot.send_photo(
                    chat_id=update.effective_chat.id,
                    photo=photo,
                    caption="Volume Spike Scanner — stocks with unusual volume vs 30-day average",
                )
            try:
                os.remove(chart_path)
            except OSError:
                pass
        except Exception as exc:
            logger.error("Volume spike chart failed: %s", exc)

    async def _analyze_handler_impl(
        update: Update, context: ContextTypes.DEFAULT_TYPE, ticker: str
    ) -> None:
        if not ticker:
            await update.message.reply_text("Usage: /analyze AAPL")
            return
        await update.message.reply_text(f"Running full institutional analysis for {ticker}…")
        loop = asyncio.get_event_loop()
        try:
            from momentum_radar.premarket.full_analysis import run_full_analysis, format_full_analysis
            analysis = await loop.run_in_executor(
                None, lambda: run_full_analysis(ticker, fetcher)
            )
        except Exception as exc:
            logger.error("Full analysis failed for %s: %s", ticker, exc)
            await update.message.reply_text(
                f"Could not run analysis for {ticker}. Make sure it's a valid US stock ticker."
            )
            return

        text_report = format_full_analysis(analysis)

        msg = _safe_text(text_report)
        # Telegram message limit is 4096 characters – split if needed
        for chunk in [msg[i:i + 4000] for i in range(0, len(msg), 4000)]:
            await update.message.reply_text(chunk)

        # Also send the chart if data is available
        try:
            daily = analysis.get("_daily")  # not stored; re-fetch below
            daily = await loop.run_in_executor(
                None, lambda: fetcher.get_daily_bars(ticker, period="90d")
            )
            if daily is not None and not daily.empty:
                from momentum_radar.utils.stock_chart import generate_analysis_chart
                chart_path = await loop.run_in_executor(
                    None,
                    lambda: generate_analysis_chart(
                        ticker=ticker,
                        daily=daily,
                        rvol=analysis.get("technical", {}).get("rvol"),
                        short_interest=analysis.get("flow", {}).get("short_interest_pct"),
                        float_shares=analysis.get("market_data", {}).get("float_shares"),
                    ),
                )
                with open(chart_path, "rb") as photo:
                    await context.bot.send_photo(
                        chat_id=update.effective_chat.id,
                        photo=photo,
                        caption=f"{ticker} — Professional Analysis Chart",
                    )
                try:
                    os.remove(chart_path)
                except OSError:
                    pass
        except Exception as exc:
            logger.debug("Analysis chart skipped for %s: %s", ticker, exc)

    async def _premarket_handler_impl(
        update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        await update.message.reply_text(
            f"Running pre-market scan on {len(universe)} stocks… This may take a moment."
        )
        loop = asyncio.get_event_loop()
        try:
            from momentum_radar.premarket.scanner import (
                scan_unusual_volume,
                scan_most_active,
                scan_options_spikes,
            )
            vol_spikes = await loop.run_in_executor(
                None, lambda: scan_unusual_volume(universe, fetcher, top_n=10)
            )
            active = await loop.run_in_executor(
                None, lambda: scan_most_active(universe, fetcher, top_n=10)
            )
            opt_spikes = await loop.run_in_executor(
                None, lambda: scan_options_spikes(universe[:100], fetcher, top_n=10)
            )
        except Exception as exc:
            logger.error("Pre-market scan failed: %s", exc)
            await update.message.reply_text("Pre-market scan failed. Please try again later.")
            return

        lines = ["PRE-MARKET SCAN RESULTS", ""]

        lines.append("UNUSUAL VOLUME (RVOL >= 2x)")
        if vol_spikes:
            for s in vol_spikes[:10]:
                direction = "+" if s["pct_change"] >= 0 else ""
                vol_str = (
                    f"{s['today_volume'] / 1e6:.1f}M"
                    if s["today_volume"] >= 1_000_000
                    else f"{s['today_volume'] / 1e3:.0f}K"
                )
                lines.append(
                    f"  {s['ticker']:6s}  RVOL {s['rvol']:.1f}x  "
                    f"{direction}{s['pct_change']:.1f}%  ${s['last_close']}  Vol {vol_str}"
                )
        else:
            lines.append("  None detected.")
        lines.append("")

        lines.append("MOST ACTIVE BY VOLUME")
        for v in active.get("highest_volume", [])[:5]:
            vol_str = (
                f"{v['today_volume'] / 1e6:.1f}M"
                if v["today_volume"] >= 1_000_000
                else f"{v['today_volume'] / 1e3:.0f}K"
            )
            direction = "+" if v["pct_change"] >= 0 else ""
            lines.append(f"  {v['ticker']:6s}  Vol {vol_str}  {direction}{v['pct_change']:.1f}%  ${v['last_close']}")
        lines.append("")

        lines.append("TOP GAINERS")
        for g in active.get("top_gainers", [])[:5]:
            lines.append(f"  {g['ticker']:6s}  +{g['pct_change']:.1f}%  ${g['last_close']}")
        lines.append("")

        lines.append("TOP LOSERS")
        for l in active.get("top_losers", [])[:5]:
            lines.append(f"  {l['ticker']:6s}  {l['pct_change']:.1f}%  ${l['last_close']}")
        lines.append("")

        lines.append("OPTIONS VOLUME SPIKES")
        if opt_spikes:
            for o in opt_spikes[:5]:
                lines.append(
                    f"  {o['ticker']:6s}  C/P {o['cp_ratio']}  "
                    f"Calls {o['call_volume']:,}  Puts {o['put_volume']:,}  "
                    f"Bias: {o['bias']}"
                )
        else:
            lines.append("  None detected.")

        msg = _safe_text("\n".join(lines))
        for chunk in [msg[i:i + 4000] for i in range(0, len(msg), 4000)]:
            await update.message.reply_text(chunk)

    async def _squeeze_handler_impl(
        update: Update, context: ContextTypes.DEFAULT_TYPE, ticker: str
    ) -> None:
        loop = asyncio.get_event_loop()
        if ticker:
            # Single-ticker full squeeze report
            await update.message.reply_text(f"Running squeeze analysis for {ticker}…")
            try:
                from momentum_radar.premarket.squeeze_detector import (
                    build_squeeze_report,
                    format_squeeze_report,
                )
                report = await loop.run_in_executor(
                    None, lambda: build_squeeze_report(ticker, fetcher)
                )
            except Exception as exc:
                logger.error("Squeeze report failed for %s: %s", ticker, exc)
                await update.message.reply_text(f"Could not run squeeze analysis for {ticker}.")
                return

            if not report:
                await update.message.reply_text(f"No data available for {ticker}.")
                return

            msg = _safe_text(format_squeeze_report(report))
            await update.message.reply_text(msg)
        else:
            # Scan full universe for top candidates
            await update.message.reply_text(
                f"Scanning {len(universe)} stocks for short squeeze setups "
                f"(high SI, rising borrow fees, unusual volume)… This may take a moment."
            )
            try:
                from momentum_radar.premarket.squeeze_detector import (
                    scan_squeeze_candidates,
                    format_squeeze_report,
                )
                candidates = await loop.run_in_executor(
                    None,
                    lambda: scan_squeeze_candidates(universe, fetcher, min_score=40, top_n=10),
                )
            except Exception as exc:
                logger.error("Squeeze scan failed: %s", exc)
                await update.message.reply_text("Squeeze scan failed. Please try again.")
                return

            if not candidates:
                await update.message.reply_text("No squeeze candidates above threshold found.")
                return

            lines = ["TOP SHORT SQUEEZE CANDIDATES", ""]
            for i, c in enumerate(candidates, 1):
                si_str = f"{c['short_interest_pct']:.1%}" if c.get("short_interest_pct") is not None else "N/A"
                dtc_str = f"{c['days_to_cover']:.1f}" if c.get("days_to_cover") is not None else "N/A"
                borrow_str = (
                    f"~{c['borrow_fee_estimate']:.0%}"
                    if c.get("borrow_fee_estimate") is not None
                    else "N/A"
                )
                lines.append(
                    f"{i:2d}. {c['ticker']:6s}  Score {c['squeeze_score']}%  "
                    f"SI {si_str}  DTC {dtc_str}  Float {c.get('float_str', 'N/A')}  "
                    f"RVOL {c.get('rvol', 'N/A')}x  Borrow {borrow_str}"
                )
                lines.append(f"    {c['squeeze_label']}")
            lines.append("")
            lines.append("Use /squeeze TICKER for a full report on any candidate.")

            msg = _safe_text("\n".join(lines))
            await update.message.reply_text(msg)

    async def _brief_handler_impl(
        update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        await update.message.reply_text(
            "Generating market intelligence brief… This may take a moment."
        )
        loop = asyncio.get_event_loop()
        try:
            from momentum_radar.premarket.briefing import generate_market_brief
            brief = await loop.run_in_executor(
                None,
                lambda: generate_market_brief(universe, fetcher, session_label="On-Demand"),
            )
        except Exception as exc:
            logger.error("Market brief failed: %s", exc)
            await update.message.reply_text("Could not generate market brief. Please try again.")
            return

        msg = _safe_text(brief)
        for chunk in [msg[i:i + 4000] for i in range(0, len(msg), 4000)]:
            await update.message.reply_text(chunk)

    async def _news_handler_impl(
        update: Update, context: ContextTypes.DEFAULT_TYPE, arg: str
    ) -> None:
        """Handle /news [TICKER | market | premarket].

        - /news AAPL      → ticker-specific news with sentiment
        - /news market    → broad market-wide news
        - /news premarket → premarket-focused news (top movers, catalysts)
        - /news           → usage hint
        """
        arg_lower = arg.lower().strip()

        # No argument – show usage hint
        if not arg_lower:
            await update.message.reply_text(
                "Usage:\n"
                "  /news AAPL       – ticker news\n"
                "  /news market     – market-wide news\n"
                "  /news premarket  – pre-market news & catalysts"
            )
            return

        # Route to market-wide news
        if arg_lower == "market":
            await _marketnews_handler_impl(update, context)
            return

        if arg_lower == "premarket":
            await update.message.reply_text(
                "Fetching pre-market news & catalysts… This may take a moment."
            )
            loop = asyncio.get_event_loop()
            try:
                from momentum_radar.news.news_fetcher import (
                    fetch_market_news,
                    summarize_news,
                    format_news_report,
                )
                articles = await loop.run_in_executor(None, fetch_market_news)
            except Exception as exc:
                logger.error("Premarket news fetch failed: %s", exc)
                await update.message.reply_text("Could not fetch pre-market news. Please try again.")
                return

            if not articles:
                await update.message.reply_text("No pre-market news available at this time.")
                return

            summary = summarize_news(articles)
            report = format_news_report(articles, summary, title="Pre-Market News & Catalysts")
            msg = _safe_text(report)
            for chunk in [msg[i:i + 4000] for i in range(0, len(msg), 4000)]:
                await update.message.reply_text(chunk)
            return

        # Ticker-specific news
        ticker = arg.upper().strip()
        if not ticker:
            await update.message.reply_text("Usage: /news AAPL")
            return
        await update.message.reply_text(
            f"Fetching latest news for {ticker} from all sources… This may take a moment."
        )
        loop = asyncio.get_event_loop()
        try:
            from momentum_radar.news.news_fetcher import (
                fetch_ticker_news,
                summarize_news,
                format_news_report,
            )
            articles = await loop.run_in_executor(
                None, lambda: fetch_ticker_news(ticker)
            )
        except Exception as exc:
            logger.error("Ticker news fetch failed for %s: %s", ticker, exc)
            await update.message.reply_text(
                f"Could not fetch news for {ticker}. Make sure it's a valid US stock ticker."
            )
            return

        if not articles:
            await update.message.reply_text(
                f"No news found for {ticker} at this time."
            )
            return

        summary = summarize_news(articles)
        report = format_news_report(articles, summary, title=f"News: {ticker}")
        msg = _safe_text(report)
        for chunk in [msg[i:i + 4000] for i in range(0, len(msg), 4000)]:
            await update.message.reply_text(chunk)

    async def _marketnews_handler_impl(
        update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        await update.message.reply_text(
            "Fetching full market news from all sources… This may take a moment."
        )
        loop = asyncio.get_event_loop()
        try:
            from momentum_radar.news.news_fetcher import (
                fetch_market_news,
                summarize_news,
                format_news_report,
            )
            articles = await loop.run_in_executor(None, fetch_market_news)
        except Exception as exc:
            logger.error("Market news fetch failed: %s", exc)
            await update.message.reply_text(
                "Could not fetch market news. Please try again later."
            )
            return

        if not articles:
            await update.message.reply_text("No market news available at this time.")
            return

        summary = summarize_news(articles)
        report = format_news_report(articles, summary, title="Full Market News")
        msg = _safe_text(report)
        for chunk in [msg[i:i + 4000] for i in range(0, len(msg), 4000)]:
            await update.message.reply_text(chunk)

    async def _fundamentals_handler_impl(
        update: Update, context: ContextTypes.DEFAULT_TYPE, ticker: str
    ) -> None:
        if not ticker:
            await update.message.reply_text("Usage: /fundamentals AAPL")
            return
        await update.message.reply_text(
            f"Fetching financial statements for {ticker}… This may take a moment."
        )
        loop = asyncio.get_event_loop()
        try:
            from momentum_radar.premarket.fundamentals import (
                get_financial_statements,
                format_fundamentals_report,
            )
            data = await loop.run_in_executor(
                None, lambda: get_financial_statements(ticker)
            )
        except Exception as exc:
            logger.error("Fundamentals fetch failed for %s: %s", ticker, exc)
            await update.message.reply_text(
                f"Could not fetch financial statements for {ticker}. Make sure it's a valid US stock ticker."
            )
            return

        report = format_fundamentals_report(data)
        msg = _safe_text(report)
        for chunk in [msg[i:i + 4000] for i in range(0, len(msg), 4000)]:
            await update.message.reply_text(chunk)

    async def _earnings_handler_impl(
        update: Update, context: ContextTypes.DEFAULT_TYPE, ticker: str
    ) -> None:
        if not ticker:
            await update.message.reply_text("Usage: /earnings AAPL")
            return
        await update.message.reply_text(
            f"Analyzing earnings for {ticker}… This may take a moment."
        )
        loop = asyncio.get_event_loop()
        try:
            from momentum_radar.premarket.fundamentals import (
                get_earnings_analysis,
                format_earnings_report,
            )
            data = await loop.run_in_executor(
                None, lambda: get_earnings_analysis(ticker)
            )
        except Exception as exc:
            logger.error("Earnings analysis failed for %s: %s", ticker, exc)
            await update.message.reply_text(
                f"Could not fetch earnings data for {ticker}. Make sure it's a valid US stock ticker."
            )
            return

        report = format_earnings_report(data)
        msg = _safe_text(report)
        for chunk in [msg[i:i + 4000] for i in range(0, len(msg), 4000)]:
            await update.message.reply_text(chunk)

    async def _morningbrief_handler_impl(
        update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Generate and send a comprehensive morning briefing."""
        await update.message.reply_text(
            "📰 Generating comprehensive morning brief… This may take a moment."
        )
        loop = asyncio.get_event_loop()
        try:
            from momentum_radar.premarket.briefing import generate_market_brief
            from momentum_radar.news.news_fetcher import (
                fetch_market_news,
                summarize_news,
            )
            from momentum_radar.utils.economic_calendar import (
                get_weekly_calendar,
                format_calendar_report,
            )

            brief = await loop.run_in_executor(
                None,
                lambda: generate_market_brief(universe, fetcher, session_label="Morning Brief"),
            )

            # Append news sentiment summary
            try:
                articles = await loop.run_in_executor(None, fetch_market_news)
                news_summary = summarize_news(articles)
                overall = news_summary.get("overall_sentiment", "NEUTRAL")
                bd = news_summary.get("sentiment_breakdown", {})
                themes = news_summary.get("key_themes", [])
                news_section = (
                    "\n\nNEWS SENTIMENT\n" + "-" * 30 + "\n"
                    f"Overall: {overall}\n"
                    f"Bullish: {bd.get('BULLISH', 0)}  Neutral: {bd.get('NEUTRAL', 0)}"
                    f"  Bearish: {bd.get('BEARISH', 0)}\n"
                )
                if themes:
                    news_section += f"Key Themes: {', '.join(themes)}\n"
                brief += news_section
            except Exception as exc:
                logger.debug("Morning brief news section failed: %s", exc)

            # Append today's economic events
            try:
                cal_events = await loop.run_in_executor(None, lambda: get_weekly_calendar(0))
                from datetime import date as _date
                today_events = [e for e in cal_events if e.get("date") == _date.today()]
                if today_events:
                    cal_section = "\n\nTODAY'S KEY EVENTS\n" + "-" * 30 + "\n"
                    for ev in today_events:
                        impact = ev.get("impact", "")
                        cal_section += f"  [{impact}] {ev.get('time', 'TBD')}  {ev.get('name', '')}\n"
                    brief += cal_section
            except Exception as exc:
                logger.debug("Morning brief calendar section failed: %s", exc)

        except Exception as exc:
            logger.error("Morning brief failed: %s", exc)
            await update.message.reply_text("Could not generate morning brief. Please try again.")
            return

        msg = _safe_text(brief)
        for chunk in [msg[i:i + 4000] for i in range(0, len(msg), 4000)]:
            await update.message.reply_text(chunk)

    async def _sentiment_handler_impl(
        update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Generate and send the market sentiment engine report."""
        await update.message.reply_text(
            "🧠 Computing market sentiment… This may take a moment."
        )
        loop = asyncio.get_event_loop()
        try:
            from momentum_radar.services.sentiment_engine import (
                get_market_sentiment,
                format_sentiment_report,
            )
            result = await loop.run_in_executor(
                None, lambda: get_market_sentiment(fetcher)
            )
            report = format_sentiment_report(result)
        except Exception as exc:
            logger.error("Sentiment engine failed: %s", exc)
            await update.message.reply_text("Could not compute market sentiment. Please try again.")
            return

        msg = _safe_text(report)
        for chunk in [msg[i:i + 4000] for i in range(0, len(msg), 4000)]:
            await update.message.reply_text(chunk)

    async def _dates_handler_impl(
        update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Send the weekly economic calendar."""
        await update.message.reply_text("📅 Building economic calendar for this week…")
        loop = asyncio.get_event_loop()
        try:
            from momentum_radar.utils.economic_calendar import (
                get_weekly_calendar,
                format_calendar_report,
            )
            events = await loop.run_in_executor(None, lambda: get_weekly_calendar(0))
            report = format_calendar_report(events, week_offset=0)
        except Exception as exc:
            logger.error("Economic calendar failed: %s", exc)
            await update.message.reply_text("Could not build economic calendar. Please try again.")
            return

        msg = _safe_text(report)
        for chunk in [msg[i:i + 4000] for i in range(0, len(msg), 4000)]:
            await update.message.reply_text(chunk)

    async def _heatmap_handler_impl(
        update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Generate and send a market sector heatmap."""
        await update.message.reply_text("📊 Generating market sector heatmap… please wait.")
        loop = asyncio.get_event_loop()
        try:
            from momentum_radar.utils.heatmap import generate_market_heatmap
            chart_path, text_summary = await loop.run_in_executor(None, generate_market_heatmap)
            if chart_path:
                try:
                    with open(chart_path, "rb") as photo:
                        await context.bot.send_photo(
                            chat_id=update.effective_chat.id,
                            photo=photo,
                            caption=text_summary[:1024],
                        )
                except Exception:
                    await update.message.reply_text(text_summary)
                finally:
                    try:
                        import os
                        os.remove(chart_path)
                    except OSError:
                        pass
            else:
                await update.message.reply_text(text_summary)
        except Exception as exc:
            logger.error("Heatmap generation error: %s", exc)
            await update.message.reply_text(f"⚠️ Failed to generate heatmap: {exc}")

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
    app.add_handler(CommandHandler("volspike", _volspike_handler))
    app.add_handler(CommandHandler("analyze", _analyze_handler))
    app.add_handler(CommandHandler("premarket", _premarket_handler))
    app.add_handler(CommandHandler("squeeze", _squeeze_handler))
    app.add_handler(CommandHandler("brief", _brief_handler))
    app.add_handler(CommandHandler("morningbrief", _morningbrief_handler))
    app.add_handler(CommandHandler("news", _news_handler))
    app.add_handler(CommandHandler("marketnews", _marketnews_handler))
    app.add_handler(CommandHandler("sentiment", _sentiment_handler))
    app.add_handler(CommandHandler("dates", _dates_handler))
    app.add_handler(CommandHandler("fundamentals", _fundamentals_handler))
    app.add_handler(CommandHandler("earnings", _earnings_handler))
    app.add_handler(CommandHandler("alerts", _alerts_handler))
    app.add_handler(CommandHandler("heatmap", _heatmap_handler))
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
