"""
economic_calendar.py – Weekly Market Economic Calendar.

Provides a comprehensive weekly schedule of market-moving economic data releases,
Federal Reserve events, and other macro catalysts.

Data is sourced from:
* yfinance calendar (earnings, macro events)
* Hardcoded fixed/recurring weekly schedule (CPI, NFP, FOMC, etc.)
* Free economic calendar APIs where available

Public API
----------
get_weekly_calendar(week_offset=0) -> List[Dict]
format_calendar_report(events)     -> str
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional

import pytz

logger = logging.getLogger(__name__)

EST = pytz.timezone("America/New_York")

# ---------------------------------------------------------------------------
# Recurring weekly event templates (day_of_week: 0=Mon, 4=Fri)
# month_day pairs indicate which week/month they typically occur
# ---------------------------------------------------------------------------

# Major recurring events with typical release day of week
_RECURRING_EVENTS: List[Dict] = [
    # Monday
    {
        "name": "ISM Manufacturing PMI",
        "day_of_week": 0,  # Monday (first business day of month)
        "frequency": "monthly",
        "time": "10:00 ET",
        "impact": "HIGH",
        "description": "Manufacturing sector activity index. >50 = expansion.",
        "category": "Economic",
    },
    # Tuesday
    {
        "name": "JOLTS Job Openings",
        "day_of_week": 1,
        "frequency": "monthly",
        "time": "10:00 ET",
        "impact": "HIGH",
        "description": "Job openings, hires, separations data.",
        "category": "Employment",
    },
    {
        "name": "Consumer Confidence",
        "day_of_week": 1,
        "frequency": "monthly",
        "time": "10:00 ET",
        "impact": "MEDIUM",
        "description": "Conference Board Consumer Confidence Index.",
        "category": "Economic",
    },
    # Wednesday
    {
        "name": "ADP Employment Change",
        "day_of_week": 2,
        "frequency": "monthly",
        "time": "08:15 ET",
        "impact": "HIGH",
        "description": "Private sector employment change (preview of NFP).",
        "category": "Employment",
    },
    {
        "name": "ISM Services PMI",
        "day_of_week": 2,
        "frequency": "monthly",
        "time": "10:00 ET",
        "impact": "HIGH",
        "description": "Services sector activity index. >50 = expansion.",
        "category": "Economic",
    },
    {
        "name": "EIA Crude Oil Inventories",
        "day_of_week": 2,
        "frequency": "weekly",
        "time": "10:30 ET",
        "impact": "MEDIUM",
        "description": "Weekly US crude oil inventory levels.",
        "category": "Energy",
    },
    {
        "name": "FOMC Minutes",
        "day_of_week": 2,
        "frequency": "6-weekly",
        "time": "14:00 ET",
        "impact": "VERY HIGH",
        "description": "Federal Reserve meeting minutes. Market-moving.",
        "category": "Fed/Rates",
    },
    # Thursday
    {
        "name": "Initial Jobless Claims",
        "day_of_week": 3,
        "frequency": "weekly",
        "time": "08:30 ET",
        "impact": "MEDIUM",
        "description": "Weekly new unemployment insurance claims.",
        "category": "Employment",
    },
    {
        "name": "Producer Price Index (PPI)",
        "day_of_week": 3,
        "frequency": "monthly",
        "time": "08:30 ET",
        "impact": "HIGH",
        "description": "Wholesale inflation indicator.",
        "category": "Inflation",
    },
    # Friday
    {
        "name": "Nonfarm Payrolls (NFP)",
        "day_of_week": 4,
        "frequency": "monthly",
        "time": "08:30 ET",
        "impact": "VERY HIGH",
        "description": "Monthly jobs report. Most watched economic indicator.",
        "category": "Employment",
    },
    {
        "name": "Unemployment Rate",
        "day_of_week": 4,
        "frequency": "monthly",
        "time": "08:30 ET",
        "impact": "HIGH",
        "description": "Released with NFP. US unemployment rate.",
        "category": "Employment",
    },
    {
        "name": "Avg Hourly Earnings",
        "day_of_week": 4,
        "frequency": "monthly",
        "time": "08:30 ET",
        "impact": "HIGH",
        "description": "Wage inflation indicator released with NFP.",
        "category": "Inflation",
    },
    {
        "name": "University of Michigan Sentiment",
        "day_of_week": 4,
        "frequency": "monthly",
        "time": "10:00 ET",
        "impact": "MEDIUM",
        "description": "Consumer sentiment and inflation expectations survey.",
        "category": "Economic",
    },
]

# Events that occur on specific dates each month (approximate)
_MONTHLY_EVENTS: List[Dict] = [
    {
        "name": "CPI (Consumer Price Index)",
        "typical_day": 10,  # around 10th of month
        "time": "08:30 ET",
        "impact": "VERY HIGH",
        "description": "Core inflation data. Directly influences Fed policy.",
        "category": "Inflation",
    },
    {
        "name": "Core PCE Price Index",
        "typical_day": 28,
        "time": "08:30 ET",
        "impact": "VERY HIGH",
        "description": "Fed's preferred inflation measure.",
        "category": "Inflation",
    },
    {
        "name": "Retail Sales",
        "typical_day": 15,
        "time": "08:30 ET",
        "impact": "HIGH",
        "description": "Monthly consumer spending data.",
        "category": "Economic",
    },
    {
        "name": "GDP (Advance Estimate)",
        "typical_day": 27,
        "time": "08:30 ET",
        "impact": "VERY HIGH",
        "description": "Quarterly GDP growth rate (advance estimate). Released in Jan, Apr, Jul, Oct.",
        "category": "Economic",
        "months": [1, 4, 7, 10],  # Only in these months (approximate release months)
    },
    {
        "name": "Housing Starts",
        "typical_day": 18,
        "time": "08:30 ET",
        "impact": "MEDIUM",
        "description": "New residential construction starts.",
        "category": "Real Estate",
    },
    {
        "name": "Existing Home Sales",
        "typical_day": 22,
        "time": "10:00 ET",
        "impact": "MEDIUM",
        "description": "Monthly completed residential transactions.",
        "category": "Real Estate",
    },
    {
        "name": "Durable Goods Orders",
        "typical_day": 24,
        "time": "08:30 ET",
        "impact": "HIGH",
        "description": "Factory orders for long-lasting manufactured goods.",
        "category": "Economic",
    },
    {
        "name": "Fed Interest Rate Decision (FOMC)",
        "typical_day": 1,  # roughly 8 times/year; shown as approximate
        "time": "14:00 ET",
        "impact": "VERY HIGH",
        "description": "Federal Reserve interest rate decision + press conference.",
        "category": "Fed/Rates",
        "note": "Scheduled ~8x per year. Verify exact date on federalreserve.gov.",
    },
    {
        "name": "Jackson Hole Symposium",
        "typical_day": 22,
        "time": "All Day",
        "impact": "VERY HIGH",
        "description": "Annual Fed symposium — major policy signal venue.",
        "category": "Fed/Rates",
        "months": [8],  # August only
        "note": "Annual event in August. Verify exact date each year.",
    },
]

# Market-specific recurring weekly events
_WEEKLY_RECURRING: List[Dict] = [
    {
        "name": "EIA Crude Oil Inventories",
        "day_of_week": 2,
        "time": "10:30 ET",
        "impact": "MEDIUM",
        "description": "Weekly crude oil inventory report from EIA.",
        "category": "Energy",
    },
    {
        "name": "Initial Jobless Claims",
        "day_of_week": 3,
        "time": "08:30 ET",
        "impact": "MEDIUM",
        "description": "Weekly new unemployment insurance filings.",
        "category": "Employment",
    },
    {
        "name": "Fed Balance Sheet",
        "day_of_week": 3,
        "time": "16:30 ET",
        "impact": "LOW",
        "description": "Weekly Federal Reserve balance sheet update.",
        "category": "Fed/Rates",
    },
    {
        "name": "Baker Hughes Rig Count",
        "day_of_week": 4,
        "time": "13:00 ET",
        "impact": "LOW",
        "description": "Weekly US oil & gas rig count.",
        "category": "Energy",
    },
]


# ---------------------------------------------------------------------------
# Calendar builder
# ---------------------------------------------------------------------------

def _get_week_dates(week_offset: int = 0) -> List[date]:
    """Return list of weekday dates for the target week.

    Args:
        week_offset: 0 = current week, 1 = next week, -1 = last week.

    Returns:
        List of 5 :class:`datetime.date` objects (Mon–Fri).
    """
    today = date.today()
    # Monday of current week
    monday = today - timedelta(days=today.weekday())
    monday += timedelta(weeks=week_offset)
    return [monday + timedelta(days=i) for i in range(5)]


def _fetch_earnings_this_week(week_dates: List[date]) -> List[Dict]:
    """Fetch earnings announcements for tickers during the target week.

    Uses yfinance to get earnings calendars for major tickers.
    """
    major_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
        "JPM", "BAC", "GS", "V", "MA", "UNH", "XOM", "CVX",
        "JNJ", "PFE", "WMT", "COST", "HD", "DIS", "NFLX",
        "AMD", "INTC", "CRM", "ORCL", "AVGO", "QCOM",
    ]

    week_start = week_dates[0]
    week_end = week_dates[-1]
    events: List[Dict] = []

    try:
        import yfinance as yf
        for ticker in major_tickers:
            try:
                t = yf.Ticker(ticker)
                cal = t.calendar
                if cal is None:
                    continue

                # yfinance calendar can be a dict or DataFrame
                if hasattr(cal, "items"):
                    # dict-like
                    earnings_date = cal.get("Earnings Date")
                    if earnings_date is not None:
                        if hasattr(earnings_date, "__iter__"):
                            earnings_date = list(earnings_date)[0] if earnings_date else None
                        if earnings_date is not None:
                            if hasattr(earnings_date, "date"):
                                ed = earnings_date.date()
                            elif isinstance(earnings_date, date):
                                ed = earnings_date
                            else:
                                continue
                            if week_start <= ed <= week_end:
                                events.append({
                                    "name": f"Earnings: {ticker}",
                                    "date": ed,
                                    "time": "Pre/Post Market",
                                    "impact": "HIGH",
                                    "description": f"{ticker} quarterly earnings release.",
                                    "category": "Earnings",
                                })
            except Exception:
                continue
    except Exception as exc:
        logger.debug("Earnings calendar fetch failed: %s", exc)

    return events


def get_weekly_calendar(week_offset: int = 0) -> List[Dict]:
    """Build a full weekly economic and earnings calendar.

    Args:
        week_offset: 0 = current week, 1 = next week, -1 = previous week.

    Returns:
        List of event dicts sorted by date and time, each with keys:
        ``name``, ``date``, ``day``, ``time``, ``impact``, ``description``, ``category``.
    """
    week_dates = _get_week_dates(week_offset)
    week_start = week_dates[0]
    week_end = week_dates[-1]
    events: List[Dict] = []

    # Add weekly recurring events (every week)
    for event in _WEEKLY_RECURRING:
        dow = event["day_of_week"]
        if 0 <= dow < len(week_dates):
            ev_date = week_dates[dow]
            events.append({
                "name": event["name"],
                "date": ev_date,
                "day": ev_date.strftime("%A"),
                "time": event.get("time", "TBD"),
                "impact": event.get("impact", "LOW"),
                "description": event.get("description", ""),
                "category": event.get("category", "Other"),
            })

    # Add monthly events that fall within this week
    for event in _MONTHLY_EVENTS:
        typical_day = event.get("typical_day", 1)
        allowed_months = event.get("months")  # None = every month
        # Create candidate dates across the month
        for d in week_dates:
            # Skip if event is restricted to specific months
            if allowed_months is not None and d.month not in allowed_months:
                continue
            # Match if the month day is within ±4 days of the typical day
            if abs(d.day - typical_day) <= 4:
                note = event.get("note", "Approximate date – verify exact release schedule")
                events.append({
                    "name": event["name"],
                    "date": d,
                    "day": d.strftime("%A"),
                    "time": event.get("time", "TBD"),
                    "impact": event.get("impact", "MEDIUM"),
                    "description": event.get("description", ""),
                    "category": event.get("category", "Other"),
                    "note": note,
                })
                break

    # Add earnings events
    earnings_events = _fetch_earnings_this_week(week_dates)
    for ev in earnings_events:
        ev_date = ev["date"]
        events.append({
            "name": ev["name"],
            "date": ev_date,
            "day": ev_date.strftime("%A"),
            "time": ev.get("time", "TBD"),
            "impact": ev.get("impact", "HIGH"),
            "description": ev.get("description", ""),
            "category": ev.get("category", "Earnings"),
        })

    # Sort by date then by impact priority
    impact_order = {"VERY HIGH": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    events.sort(key=lambda e: (e["date"], impact_order.get(e["impact"], 4)))

    return events


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

_IMPACT_EMOJI = {
    "VERY HIGH": "🔴",
    "HIGH": "🟠",
    "MEDIUM": "🟡",
    "LOW": "⚪",
}

_CATEGORY_EMOJI = {
    "Inflation": "📊",
    "Employment": "👷",
    "Fed/Rates": "🏦",
    "Economic": "📈",
    "Earnings": "💰",
    "Energy": "⛽",
    "Real Estate": "🏠",
    "Other": "📅",
}


def format_calendar_report(events: List[Dict], week_offset: int = 0) -> str:
    """Render an economic calendar as a formatted Telegram/console string.

    Args:
        events:      List from :func:`get_weekly_calendar`.
        week_offset: Used only for the header label.

    Returns:
        Multi-line formatted string.
    """
    week_dates = _get_week_dates(week_offset)
    week_start = week_dates[0].strftime("%b %d")
    week_end = week_dates[-1].strftime("%b %d, %Y")

    if week_offset == 0:
        week_label = "This Week"
    elif week_offset == 1:
        week_label = "Next Week"
    elif week_offset == -1:
        week_label = "Last Week"
    else:
        week_label = f"Week of {week_start}"

    now_str = datetime.now(tz=EST).strftime("%Y-%m-%d %H:%M ET")

    lines = [
        f"MARKET ECONOMIC CALENDAR – {week_label}",
        f"{week_start} – {week_end}",
        f"Generated: {now_str}",
        "",
    ]

    # Group by day
    current_day: Optional[str] = None
    today = date.today()

    for event in events:
        ev_date = event.get("date")
        if not isinstance(ev_date, date):
            continue
        day_str = event.get("day", ev_date.strftime("%A"))

        if day_str != current_day:
            current_day = day_str
            lines.append("")
            # Mark today
            marker = " ← TODAY" if ev_date == today else ""
            lines.append(f"{'=' * 35}")
            lines.append(f"{day_str.upper()} {ev_date.strftime('%b %d')}{marker}")
            lines.append(f"{'=' * 35}")

        impact = event.get("impact", "LOW")
        impact_emoji = _IMPACT_EMOJI.get(impact, "⚪")
        cat_emoji = _CATEGORY_EMOJI.get(event.get("category", "Other"), "📅")
        name = event.get("name", "Unknown")
        time_str = event.get("time", "TBD")
        desc = event.get("description", "")
        note = event.get("note", "")

        lines.append(f"{impact_emoji} {cat_emoji} {time_str}  {name}")
        if desc:
            lines.append(f"         {desc[:80]}")
        if note:
            lines.append(f"         ⚠️  {note}")

    if not any(event.get("date") for event in events):
        lines.append("  No scheduled events found for this week.")

    lines.append("")
    lines.append("=" * 35)
    lines.append("Impact: 🔴 VERY HIGH  🟠 HIGH  🟡 MEDIUM  ⚪ LOW")
    lines.append("Dates are approximate. Always verify official sources.")

    return "\n".join(lines)
