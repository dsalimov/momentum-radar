"""
ui/embed_formatter.py – Professional alert card formatter.

Produces the structured institutional-grade alert cards for Telegram and
Discord delivery.

Telegram format (Markdown card)::

    ━━━━━━━━━━━━━━━━━━
    📊 SPY | BUY | 5m
    ━━━━━━━━━━━━━━━━━━

    ⭐ Score: 84 / 100  (Grade A)

    📈 Market Context
    • Regime: Trending
    • HTF Bias: Bullish
    • Session: Open Momentum

    🧠 Confirmations (4)
    ✔ HTF Alignment
    ✔ Break of Structure
    ✔ Volume Expansion
    ✔ Demand Zone Retest

    💰 Trade Plan
    Entry: 512.40
    Stop:  510.80
    Target: 516.80
    R:R:  2.75

    🛡 Validation
    Fake Breakout: PASSED

    ━━━━━━━━━━━━━━━━━━

Discord format: a structured embed dict ready for the webhooks API.

Rules:
* No paragraphs – bullet formatting only
* Emojis used sparingly and consistently
* Max 15 lines per alert
* No raw indicator values
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

from momentum_radar.strategies.base import StrategySignal

logger = logging.getLogger(__name__)

_SEPARATOR = "━━━━━━━━━━━━━━━━━━"

# Discord embed colours (integer, not hex string)
_COLOR_GREEN  = 0x27AE60  # score ≥ 80
_COLOR_YELLOW = 0xF39C12  # score 70–79
_COLOR_BLUE   = 0x2980B9  # score ≥ 90 (premium)
_COLOR_RED    = 0xC0392B  # sell setup

# Strategy display names (only DAY TRADE and SWING TRADE categories remain)
_STRATEGY_NAMES: Dict[str, str] = {
    "scalp":          "DAY TRADE",
    "intraday":       "INTRADAY",
    "swing":          "SWING",
    "chart_pattern":  "PATTERN",
    "unusual_volume": "VOL BREAK",
}

# Session display names
_SESSION_NAMES: Dict[str, str] = {
    "open":      "Open Momentum",
    "morning":   "Morning Session",
    "midday":    "Midday",
    "afternoon": "Afternoon Trend",
    "premarket": "Pre-Market",
    "afterhours": "After Hours",
}


def _score_grade_label(score: int) -> str:
    """Return a score / grade display string like ``"84 / 100  (Grade A)"``."""
    if score >= 90:
        return f"{score} / 100  (Grade A+)"
    if score >= 80:
        return f"{score} / 100  (Grade A)"
    if score >= 70:
        return f"{score} / 100  (Grade B)"
    return f"{score} / 100  (Grade C)"


def _fake_breakout_label(passed: bool) -> str:
    return "PASSED ✅" if passed else "FAILED ❌"


def _rr_label(rr: float) -> str:
    return f"{rr:.2f}"


def format_telegram_card(
    signal: StrategySignal,
    timestamp: Optional[datetime] = None,
) -> str:
    """Build a structured Telegram alert card from a :class:`StrategySignal`.

    Args:
        signal:    The evaluated strategy signal.
        timestamp: Alert timestamp (defaults to ``datetime.now()``).

    Returns:
        Multi-line Markdown-style string ready for ``sendMessage``.
    """
    ts        = timestamp or datetime.now()
    strat     = _STRATEGY_NAMES.get(signal.strategy, signal.strategy.upper())
    header    = f"📊 {signal.ticker} | {signal.direction} | {signal.timeframe}"
    session   = _SESSION_NAMES.get(signal.session, signal.session.capitalize())
    grade_str = _score_grade_label(signal.score)
    confs     = signal.confirmations[:4]  # max 4 displayed

    lines: List[str] = [
        _SEPARATOR,
        header,
        _SEPARATOR,
        "",
        f"⭐ Score: {grade_str}",
        f"🎯 Strategy: {strat}",
        "",
        "📈 Market Context",
        f"• Regime: {signal.regime}",
        f"• HTF Bias: {signal.htf_bias}",
        f"• Session: {session}",
        "",
        f"🧠 Confirmations ({len(confs)})",
    ]
    for conf in confs:
        lines.append(f"✔ {conf}")

    lines += [
        "",
        "💰 Trade Plan",
        f"Entry:  {signal.entry:.2f}",
        f"Stop:   {signal.stop:.2f}",
        f"Target: {signal.target:.2f}",
        f"R:R:    {_rr_label(signal.rr)}",
        "",
        "🛡 Validation",
        f"Fake Breakout: {_fake_breakout_label(signal.fake_breakout_passed)}",
        "",
        _SEPARATOR,
        f"🕐 {ts.strftime('%Y-%m-%d %H:%M')} ET",
    ]
    return "\n".join(lines)


def _discord_color(signal: StrategySignal) -> int:
    """Return Discord embed colour based on score and direction."""
    if signal.direction == "SELL":
        return _COLOR_RED
    if signal.score >= 90:
        return _COLOR_BLUE
    if signal.score >= 80:
        return _COLOR_GREEN
    return _COLOR_YELLOW


def format_discord_embed(
    signal: StrategySignal,
    timestamp: Optional[datetime] = None,
    bot_version: str = "1.0",
) -> Dict:
    """Build a Discord embed object from a :class:`StrategySignal`.

    The returned dict is ready to be serialised into the Discord webhooks API
    ``embeds`` array.

    Args:
        signal:      The evaluated strategy signal.
        timestamp:   Alert timestamp (defaults to ``datetime.now()``).
        bot_version: Version string shown in the embed footer.

    Returns:
        Dict conforming to the Discord embed object schema.
    """
    ts    = timestamp or datetime.now()
    strat = _STRATEGY_NAMES.get(signal.strategy, signal.strategy.upper())
    title = f"{signal.ticker} | {signal.direction} | {signal.timeframe}"

    conf_text = "\n".join(f"✔ {c}" for c in signal.confirmations[:5]) or "—"
    session   = _SESSION_NAMES.get(signal.session, signal.session.capitalize())

    embed = {
        "title":       title,
        "color":       _discord_color(signal),
        "timestamp":   ts.isoformat(),
        "fields": [
            {
                "name":   "Score",
                "value":  _score_grade_label(signal.score),
                "inline": True,
            },
            {
                "name":   "Strategy",
                "value":  strat,
                "inline": True,
            },
            {
                "name":  "Market Context",
                "value": (
                    f"Regime: {signal.regime}\n"
                    f"HTF Bias: {signal.htf_bias}\n"
                    f"Session: {session}"
                ),
                "inline": False,
            },
            {
                "name":   "Confirmations",
                "value":  conf_text,
                "inline": False,
            },
            {
                "name":  "Trade Plan",
                "value": (
                    f"Entry:  {signal.entry:.2f}\n"
                    f"Stop:   {signal.stop:.2f}\n"
                    f"Target: {signal.target:.2f}\n"
                    f"R:R:    {_rr_label(signal.rr)}"
                ),
                "inline": False,
            },
            {
                "name":  "Validation",
                "value": f"Fake Breakout: {_fake_breakout_label(signal.fake_breakout_passed)}",
                "inline": False,
            },
        ],
        "footer": {
            "text": f"v{bot_version} | {ts.strftime('%Y-%m-%d')}",
        },
    }
    return embed


def format_daily_summary(
    signals: List[StrategySignal],
    timestamp: Optional[datetime] = None,
) -> str:
    """Build a daily summary card for end-of-session delivery.

    Args:
        signals:   All signals generated during the session.
        timestamp: Summary timestamp (defaults to ``datetime.now()``).

    Returns:
        Formatted Telegram card string.
    """
    ts = timestamp or datetime.now()
    total    = len(signals)
    a_plus   = sum(1 for s in signals if s.grade == "A+")
    a_grade  = sum(1 for s in signals if s.grade == "A")
    b_grade  = sum(1 for s in signals if s.grade == "B")
    valid    = [s for s in signals if s.valid]
    best_rr  = max((s.rr for s in valid), default=0.0)

    lines: List[str] = [
        _SEPARATOR,
        "📊 Daily Summary",
        _SEPARATOR,
        "",
        f"Total Signals:  {total}",
        f"A+ Grade:       {a_plus}",
        f"A  Grade:       {a_grade}",
        f"B  Grade:       {b_grade}",
        f"Valid Setups:   {len(valid)}",
        f"Best R:R:       {best_rr:.2f}",
        "",
        _SEPARATOR,
        f"🕐 {ts.strftime('%Y-%m-%d')} End of Session",
    ]
    return "\n".join(lines)
