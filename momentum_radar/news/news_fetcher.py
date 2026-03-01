"""
news_fetcher.py – Multi-source stock news aggregator with AI-style summary.

Aggregates news from Finnhub, yfinance, and Polygon.io and synthesises an
AI-style sentiment summary with key themes, bullish/bearish headlines, and
a structured market narrative.

Public API
----------
fetch_market_news()           -> List[Dict]
fetch_ticker_news(ticker)     -> List[Dict]
summarize_news(articles)      -> Dict
format_news_report(...)       -> str
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pytz

logger = logging.getLogger(__name__)

EST = pytz.timezone("America/New_York")


# ---------------------------------------------------------------------------
# Sentiment keyword lists
# ---------------------------------------------------------------------------

_BULLISH_KEYWORDS = [
    "beat",
    "beats",
    "upgrade",
    "upgraded",
    "outperform",
    "buy",
    "bullish",
    "growth",
    "raised",
    "raising",
    "strong",
    "record",
    "gain",
    "gains",
    "rally",
    "surges",
    "surge",
    "breakout",
    "positive",
    "exceeds",
    "exceeded",
    "acquisition",
    "approved",
    "approval",
    "contract",
    "deal",
    "partnership",
    "dividend",
    "buyback",
    "profit",
    "revenue growth",
    "expansion",
    "launch",
    "launches",
    "innovation",
    "breakthrough",
    "rebound",
    "recovery",
]

_BEARISH_KEYWORDS = [
    "miss",
    "misses",
    "missed",
    "downgrade",
    "downgraded",
    "underperform",
    "sell",
    "bearish",
    "decline",
    "lowered",
    "lowering",
    "weak",
    "loss",
    "losses",
    "drop",
    "drops",
    "fall",
    "falls",
    "crash",
    "negative",
    "below",
    "cut",
    "cuts",
    "lawsuit",
    "investigation",
    "recall",
    "warning",
    "concern",
    "concerns",
    "debt",
    "bankruptcy",
    "layoffs",
    "restructuring",
    "fine",
    "penalty",
    "disappointing",
    "disappoints",
    "guidance cut",
    "guidance lowered",
]

_MARKET_THEMES: Dict[str, List[str]] = {
    "Fed/Rates": [
        "fed",
        "federal reserve",
        "interest rate",
        "fomc",
        "powell",
        "inflation",
        "cpi",
        "pce",
        "rate hike",
        "rate cut",
    ],
    "Earnings": [
        "earnings",
        "eps",
        "revenue",
        "guidance",
        "quarterly",
        "results",
        "quarter",
        "q1",
        "q2",
        "q3",
        "q4",
    ],
    "M&A": [
        "acquisition",
        "merger",
        "takeover",
        "deal",
        "buyout",
        "acquired",
        "merge",
    ],
    "AI/Tech": [
        "artificial intelligence",
        "ai",
        "cloud",
        "software",
        "semiconductor",
        "chip",
        "data center",
        "nvidia",
    ],
    "Energy": [
        "oil",
        "gas",
        "energy",
        "crude",
        "opec",
        "renewable",
        "solar",
        "wind",
    ],
    "Macro": [
        "gdp",
        "jobs",
        "unemployment",
        "economy",
        "recession",
        "growth",
        "consumer",
        "spending",
    ],
    "Geopolitics": [
        "china",
        "russia",
        "war",
        "tariff",
        "trade",
        "sanctions",
        "ukraine",
        "middle east",
    ],
    "Regulation": [
        "sec",
        "ftc",
        "doj",
        "regulation",
        "antitrust",
        "compliance",
        "investigation",
        "probe",
    ],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _score_sentiment(text: str) -> int:
    """Return a raw sentiment score (positive = bullish, negative = bearish)."""
    text_lower = text.lower()
    bull = sum(1 for kw in _BULLISH_KEYWORDS if kw in text_lower)
    bear = sum(1 for kw in _BEARISH_KEYWORDS if kw in text_lower)
    return bull - bear


def _classify_sentiment(score: int) -> str:
    if score >= 2:
        return "BULLISH"
    if score <= -2:
        return "BEARISH"
    return "NEUTRAL"


def _detect_themes(articles: List[Dict]) -> List[str]:
    """Return the top market themes detected across all article text."""
    combined = " ".join(
        (a.get("headline", "") + " " + a.get("summary", "")).lower()
        for a in articles
    )
    counts: Dict[str, int] = {}
    for theme, keywords in _MARKET_THEMES.items():
        hits = sum(1 for kw in keywords if kw in combined)
        if hits:
            counts[theme] = hits
    return sorted(counts, key=counts.__getitem__, reverse=True)[:5]


def _deduplicate(articles: List[Dict]) -> List[Dict]:
    """Remove articles with duplicate headlines (first 80 chars, case-insensitive)."""
    seen: set = set()
    unique: List[Dict] = []
    for a in articles:
        key = a.get("headline", "").strip().lower()[:80]
        if key and key not in seen:
            seen.add(key)
            unique.append(a)
    return unique


# ---------------------------------------------------------------------------
# Source-specific fetchers
# ---------------------------------------------------------------------------

def _fetch_finnhub_market_news() -> List[Dict]:
    """Fetch general market news from Finnhub (category: general)."""
    try:
        from momentum_radar.config import config
        if not config.data.finnhub_api_key:
            return []
        import finnhub
        client = finnhub.Client(api_key=config.data.finnhub_api_key)
        raw = client.general_news("general", min_id=0) or []
        return [
            {
                "source": "Finnhub",
                "headline": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "url": item.get("url", ""),
                "datetime": item.get("datetime", 0),
                "ticker": None,
            }
            for item in raw[:30]
        ]
    except Exception as exc:
        logger.debug("Finnhub market news failed: %s", exc)
        return []


def _fetch_finnhub_company_news(ticker: str) -> List[Dict]:
    """Fetch company-specific news from Finnhub for the past 7 days."""
    try:
        from momentum_radar.config import config
        if not config.data.finnhub_api_key:
            return []
        import finnhub
        client = finnhub.Client(api_key=config.data.finnhub_api_key)
        to_date = datetime.now(tz=EST).strftime("%Y-%m-%d")
        from_date = (datetime.now(tz=EST) - timedelta(days=7)).strftime("%Y-%m-%d")
        raw = client.company_news(ticker, _from=from_date, to=to_date) or []
        return [
            {
                "source": "Finnhub",
                "headline": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "url": item.get("url", ""),
                "datetime": item.get("datetime", 0),
                "ticker": ticker,
            }
            for item in raw[:20]
        ]
    except Exception as exc:
        logger.debug("Finnhub company news failed for %s: %s", ticker, exc)
        return []


def _fetch_yfinance_news(ticker: str) -> List[Dict]:
    """Fetch news from yfinance for a ticker (or index symbol)."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        raw = t.news or []
        articles: List[Dict] = []
        for item in raw[:20]:
            content = item.get("content", {})
            headline = content.get("title") or item.get("title", "")
            summary = content.get("summary", "")
            url = (
                content.get("canonicalUrl", {}).get("url", "")
                or item.get("link", "")
            )
            pub_date = content.get("pubDate", "") or ""
            articles.append(
                {
                    "source": "Yahoo Finance",
                    "headline": headline,
                    "summary": summary,
                    "url": url,
                    "datetime": pub_date,
                    "ticker": ticker,
                }
            )
        return articles
    except Exception as exc:
        logger.debug("yfinance news failed for %s: %s", ticker, exc)
        return []


def _fetch_polygon_news(ticker: Optional[str] = None) -> List[Dict]:
    """Fetch news from Polygon.io (company-specific or market-wide)."""
    try:
        from momentum_radar.config import config
        if not config.data.polygon_api_key:
            return []
        import requests
        params: Dict = {
            "apiKey": config.data.polygon_api_key,
            "limit": 20,
            "order": "desc",
            "sort": "published_utc",
        }
        if ticker:
            params["ticker"] = ticker
        resp = requests.get(
            "https://api.polygon.io/v2/reference/news",
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        articles: List[Dict] = []
        for item in data.get("results", []):
            tickers = item.get("tickers", [])
            articles.append(
                {
                    "source": "Polygon",
                    "headline": item.get("title", ""),
                    "summary": item.get("description", ""),
                    "url": item.get("article_url", ""),
                    "datetime": item.get("published_utc", ""),
                    "ticker": tickers[0] if tickers else ticker,
                }
            )
        return articles
    except Exception as exc:
        logger.debug("Polygon news failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_market_news() -> List[Dict]:
    """Fetch broad market news from all available sources.

    Sources tried (in order): Finnhub general news, yfinance for major
    indices (SPY / QQQ / DIA / IWM), Polygon market-wide news.

    Returns:
        Up to 50 deduplicated article dicts with keys:
        ``source``, ``headline``, ``summary``, ``url``, ``datetime``, ``ticker``.
    """
    articles: List[Dict] = []
    articles.extend(_fetch_finnhub_market_news())
    for sym in ("SPY", "QQQ", "DIA", "IWM"):
        articles.extend(_fetch_yfinance_news(sym))
    articles.extend(_fetch_polygon_news())
    return _deduplicate(articles)[:50]


def fetch_ticker_news(ticker: str) -> List[Dict]:
    """Fetch news for a specific stock ticker from all available sources.

    Sources tried (in order): Finnhub company news, yfinance, Polygon.

    Args:
        ticker: Stock symbol (e.g. ``"AAPL"``).

    Returns:
        Up to 30 deduplicated article dicts with keys:
        ``source``, ``headline``, ``summary``, ``url``, ``datetime``, ``ticker``.
    """
    ticker = ticker.upper().strip()
    articles: List[Dict] = []
    articles.extend(_fetch_finnhub_company_news(ticker))
    articles.extend(_fetch_yfinance_news(ticker))
    articles.extend(_fetch_polygon_news(ticker))
    return _deduplicate(articles)[:30]


def summarize_news(articles: List[Dict]) -> Dict:
    """Generate an AI-style summary of news articles.

    Analyses sentiment using keyword scoring, detects key market themes,
    and surfaces the most impactful bullish and bearish headlines.

    Args:
        articles: Article list from :func:`fetch_market_news` or
                  :func:`fetch_ticker_news`.

    Returns:
        Dict with keys:
        ``overall_sentiment``, ``sentiment_breakdown``, ``key_themes``,
        ``bullish_headlines``, ``bearish_headlines``, ``total_articles``.
    """
    if not articles:
        return {
            "overall_sentiment": "NEUTRAL",
            "sentiment_breakdown": {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0},
            "key_themes": [],
            "bullish_headlines": [],
            "bearish_headlines": [],
            "total_articles": 0,
        }

    scores: List[int] = []
    classified: Dict[str, List[Dict]] = {"BULLISH": [], "BEARISH": [], "NEUTRAL": []}

    for article in articles:
        text = article.get("headline", "") + " " + article.get("summary", "")
        score = _score_sentiment(text)
        sentiment = _classify_sentiment(score)
        article["_sentiment"] = sentiment
        article["_score"] = score
        scores.append(score)
        classified[sentiment].append(article)

    avg_score = sum(scores) / len(scores) if scores else 0
    overall = _classify_sentiment(int(round(avg_score)))
    breakdown = {k: len(v) for k, v in classified.items()}

    bullish = sorted(classified["BULLISH"], key=lambda a: a["_score"], reverse=True)
    bearish = sorted(classified["BEARISH"], key=lambda a: a["_score"])

    return {
        "overall_sentiment": overall,
        "sentiment_breakdown": breakdown,
        "key_themes": _detect_themes(articles),
        "bullish_headlines": [a["headline"] for a in bullish[:5]],
        "bearish_headlines": [a["headline"] for a in bearish[:5]],
        "total_articles": len(articles),
    }


def format_news_report(
    articles: List[Dict],
    summary: Dict,
    title: str = "Market News",
) -> str:
    """Render a news report as a multi-line string for Telegram / console.

    Args:
        articles: Article list (with ``_sentiment`` keys added by
                  :func:`summarize_news`).
        summary:  Dict returned by :func:`summarize_news`.
        title:    Header line of the report.

    Returns:
        Formatted string ready for Telegram or console output.
    """
    now = datetime.now(tz=EST).strftime("%Y-%m-%d %H:%M ET")
    lines: List[str] = [
        title,
        f"Generated: {now}",
        "",
        "=" * 40,
        "AI SUMMARY",
        "=" * 40,
    ]

    overall = summary.get("overall_sentiment", "NEUTRAL")
    emoji = "📈" if overall == "BULLISH" else ("📉" if overall == "BEARISH" else "➡️")
    lines.append(f"{emoji} Overall Sentiment: {overall}")

    bd = summary.get("sentiment_breakdown", {})
    lines.append(
        f"Breakdown: {bd.get('BULLISH', 0)} bullish / "
        f"{bd.get('NEUTRAL', 0)} neutral / "
        f"{bd.get('BEARISH', 0)} bearish "
        f"({summary.get('total_articles', 0)} articles scanned)"
    )

    themes = summary.get("key_themes", [])
    if themes:
        lines.append(f"Key Themes: {', '.join(themes)}")

    bullish_heads = summary.get("bullish_headlines", [])
    if bullish_heads:
        lines.append("")
        lines.append("TOP BULLISH STORIES")
        for h in bullish_heads[:3]:
            lines.append(f"  + {h}")

    bearish_heads = summary.get("bearish_headlines", [])
    if bearish_heads:
        lines.append("")
        lines.append("TOP BEARISH STORIES")
        for h in bearish_heads[:3]:
            lines.append(f"  - {h}")

    # Latest headlines
    lines.append("")
    lines.append("=" * 40)
    lines.append("LATEST HEADLINES")
    lines.append("=" * 40)

    shown = 0
    for article in articles[:15]:
        headline = article.get("headline", "").strip()
        if not headline:
            continue
        source = article.get("source", "")
        sentiment = article.get("_sentiment", "")
        marker = "+" if sentiment == "BULLISH" else ("-" if sentiment == "BEARISH" else " ")
        lines.append(f"[{marker}] {headline} ({source})")
        shown += 1
        if shown >= 15:
            break

    if not shown:
        lines.append("  No headlines available.")

    return "\n".join(lines)
