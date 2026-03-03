"""
sentiment_engine.py – Probabilistic Market Sentiment Engine.

Combines multiple signals to produce a holistic market regime assessment:

* News sentiment (via news_fetcher)
* Index trend (SPY / QQQ percentage change)
* VIX regime (fear vs calm)
* Market breadth (advance/decline proxy via sector ETFs)

Output:
    market_regime  – "Strong Bullish" | "Bullish" | "Neutral" | "Bearish" | "Risk-Off"
    confidence_pct – 0–100 float
    components     – breakdown of each signal contribution
    summary        – human-readable paragraph

Usage::

    from momentum_radar.services.sentiment_engine import get_market_sentiment, format_sentiment_report
    result = get_market_sentiment(fetcher)
    print(format_sentiment_report(result))
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import pytz

from momentum_radar.data.data_fetcher import BaseDataFetcher

logger = logging.getLogger(__name__)

EST = pytz.timezone("America/New_York")

# ---------------------------------------------------------------------------
# Scoring weights (sum = 100)
# ---------------------------------------------------------------------------

_WEIGHTS = {
    "news": 25,
    "spy_trend": 25,
    "qqq_trend": 15,
    "vix": 20,
    "breadth": 15,
}

# ---------------------------------------------------------------------------
# Sector ETFs used for breadth calculation
# ---------------------------------------------------------------------------

_BREADTH_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLY", "XLP", "XLI", "XLB", "XLRE", "XLU", "XLC"]


# ---------------------------------------------------------------------------
# Signal scorers (each returns a value from -1.0 to +1.0)
# ---------------------------------------------------------------------------

def _score_news_sentiment(fetcher: Optional[BaseDataFetcher] = None) -> float:
    """Return sentiment score in [-1, 1] based on recent market news."""
    try:
        from momentum_radar.news.news_fetcher import fetch_market_news, summarize_news
        articles = fetch_market_news()
        summary = summarize_news(articles)
        bd = summary.get("sentiment_breakdown", {})
        total = summary.get("total_articles", 0)
        if total == 0:
            return 0.0
        bull = bd.get("BULLISH", 0)
        bear = bd.get("BEARISH", 0)
        # Normalize to [-1, 1]
        return round((bull - bear) / total, 3)
    except Exception as exc:
        logger.debug("News sentiment scoring failed: %s", exc)
        return 0.0


def _score_index(pct_change: Optional[float]) -> float:
    """Map index percentage change to a score in [-1, 1].

    Ranges (empirically calibrated):
        >= +1.5% → strong bullish (+1.0)
        +0.5% .. +1.5% → bullish (+0.5)
        -0.5% .. +0.5% → neutral (0.0)
        -1.5% .. -0.5% → bearish (-0.5)
        <= -1.5% → strong bearish (-1.0)
    """
    if pct_change is None:
        return 0.0
    if pct_change >= 1.5:
        return 1.0
    if pct_change >= 0.5:
        return 0.5
    if pct_change >= -0.5:
        return 0.0
    if pct_change >= -1.5:
        return -0.5
    return -1.0


def _score_vix(vix_price: Optional[float]) -> float:
    """Return sentiment score based on VIX (inverse – high VIX = bearish).

    Ranges:
        <= 15 → strong bullish (+1.0)
        15–20 → bullish (+0.5)
        20–25 → neutral (0.0)
        25–30 → bearish (-0.5)
        > 30  → risk-off (-1.0)
    """
    if vix_price is None:
        return 0.0
    if vix_price <= 15:
        return 1.0
    if vix_price <= 20:
        return 0.5
    if vix_price <= 25:
        return 0.0
    if vix_price <= 30:
        return -0.5
    return -1.0


def _score_breadth(fetcher: BaseDataFetcher) -> float:
    """Return breadth score from sector ETF performance.

    Computes the fraction of sector ETFs that are positive on the day
    and maps to [-1, 1].
    """
    try:
        positive = 0
        measured = 0
        for sym in _BREADTH_ETFS:
            try:
                quote = fetcher.get_quote(sym)
                if quote and quote.get("price") and quote.get("prev_close"):
                    pct = (quote["price"] - quote["prev_close"]) / quote["prev_close"]
                    measured += 1
                    if pct > 0:
                        positive += 1
            except Exception:
                continue

        if measured == 0:
            return 0.0

        breadth_ratio = positive / measured  # 0.0 to 1.0
        # Map [0,1] → [-1, 1]
        return round(2 * breadth_ratio - 1, 3)
    except Exception as exc:
        logger.debug("Breadth scoring failed: %s", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Regime classifier
# ---------------------------------------------------------------------------

def _classify_regime(weighted_score: float) -> str:
    """Map weighted composite score to a named regime.

    Score range: [-1, +1] (weighted average of component scores).
    """
    if weighted_score >= 0.6:
        return "Strong Bullish"
    if weighted_score >= 0.2:
        return "Bullish"
    if weighted_score >= -0.2:
        return "Neutral"
    if weighted_score >= -0.6:
        return "Bearish"
    return "Risk-Off"


def _compute_confidence(component_scores: Dict[str, float]) -> float:
    """Convert agreement across components into a confidence percentage.

    Higher agreement among signals = higher confidence.
    Returns value in [0, 100].
    """
    scores = list(component_scores.values())
    if not scores:
        return 50.0

    # Average absolute deviation from the mean (lower = more agreement)
    mean = sum(scores) / len(scores)
    avg_dev = sum(abs(s - mean) for s in scores) / len(scores)

    # avg_dev is in [0, 2]; map to confidence [0, 100]
    # Perfect agreement (avg_dev=0) → 95% confidence
    # Maximum disagreement (avg_dev=1) → 50% confidence
    confidence = max(0.0, 95.0 - avg_dev * 45.0)
    return round(confidence, 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_market_sentiment(fetcher: BaseDataFetcher) -> Dict:
    """Compute a probabilistic market sentiment assessment.

    Args:
        fetcher: Data provider instance used to fetch quotes.

    Returns:
        Dict with keys:
        ``market_regime``, ``confidence_pct``, ``components``,
        ``weighted_score``, ``timestamp``.
    """
    now_est = datetime.now(tz=EST)

    # Fetch index quotes
    spy_pct: Optional[float] = None
    qqq_pct: Optional[float] = None
    vix_price: Optional[float] = None

    try:
        spy_quote = fetcher.get_quote("SPY")
        if spy_quote and spy_quote.get("price") and spy_quote.get("prev_close"):
            spy_pct = (spy_quote["price"] - spy_quote["prev_close"]) / spy_quote["prev_close"] * 100
    except Exception as exc:
        logger.debug("SPY quote failed: %s", exc)

    try:
        qqq_quote = fetcher.get_quote("QQQ")
        if qqq_quote and qqq_quote.get("price") and qqq_quote.get("prev_close"):
            qqq_pct = (qqq_quote["price"] - qqq_quote["prev_close"]) / qqq_quote["prev_close"] * 100
    except Exception as exc:
        logger.debug("QQQ quote failed: %s", exc)

    try:
        vix_quote = fetcher.get_quote("VIX")
        if vix_quote and vix_quote.get("price"):
            vix_price = float(vix_quote["price"])
    except Exception as exc:
        logger.debug("VIX quote failed: %s", exc)

    # Compute component scores
    news_score = _score_news_sentiment()
    spy_score = _score_index(spy_pct)
    qqq_score = _score_index(qqq_pct)
    vix_score = _score_vix(vix_price)
    breadth_score = _score_breadth(fetcher)

    component_scores = {
        "news": news_score,
        "spy_trend": spy_score,
        "qqq_trend": qqq_score,
        "vix": vix_score,
        "breadth": breadth_score,
    }

    # Weighted composite score
    total_weight = sum(_WEIGHTS.values())
    weighted_score = sum(
        component_scores[k] * _WEIGHTS[k] / total_weight
        for k in component_scores
    )
    weighted_score = round(weighted_score, 4)

    regime = _classify_regime(weighted_score)
    confidence = _compute_confidence(component_scores)

    components = {
        "news_sentiment": {
            "score": news_score,
            "weight": _WEIGHTS["news"],
            "label": "News Sentiment",
        },
        "spy_trend": {
            "score": spy_score,
            "pct_change": round(spy_pct, 2) if spy_pct is not None else None,
            "weight": _WEIGHTS["spy_trend"],
            "label": "SPY Trend",
        },
        "qqq_trend": {
            "score": qqq_score,
            "pct_change": round(qqq_pct, 2) if qqq_pct is not None else None,
            "weight": _WEIGHTS["qqq_trend"],
            "label": "QQQ Trend",
        },
        "vix": {
            "score": vix_score,
            "vix_price": round(vix_price, 2) if vix_price is not None else None,
            "weight": _WEIGHTS["vix"],
            "label": "VIX / Volatility",
        },
        "breadth": {
            "score": breadth_score,
            "weight": _WEIGHTS["breadth"],
            "label": "Market Breadth",
        },
    }

    return {
        "market_regime": regime,
        "confidence_pct": confidence,
        "weighted_score": weighted_score,
        "components": components,
        "timestamp": now_est.strftime("%Y-%m-%d %H:%M ET"),
    }


def format_sentiment_report(result: Dict) -> str:
    """Render a sentiment report as a multi-line string.

    Args:
        result: Dict from :func:`get_market_sentiment`.

    Returns:
        Formatted string ready for Telegram or console output.
    """
    regime = result.get("market_regime", "Unknown")
    confidence = result.get("confidence_pct", 0.0)
    score = result.get("weighted_score", 0.0)
    ts = result.get("timestamp", "")
    components = result.get("components", {})

    regime_emoji = {
        "Strong Bullish": "🟢",
        "Bullish": "📈",
        "Neutral": "➡️",
        "Bearish": "📉",
        "Risk-Off": "🔴",
    }.get(regime, "❓")

    lines = [
        "MARKET SENTIMENT ENGINE",
        f"Generated: {ts}",
        "",
        "=" * 40,
        "MARKET REGIME",
        "=" * 40,
        f"{regime_emoji} {regime}",
        f"Confidence: {confidence:.0f}%",
        f"Composite Score: {score:+.2f} (scale: -1.0 to +1.0)",
        "",
        "=" * 40,
        "SIGNAL BREAKDOWN",
        "=" * 40,
    ]

    def _score_to_bar(s: float) -> str:
        """Map any score in [-1, +1] to a visual bar label."""
        if s >= 0.75:
            return "████████ STRONG BULL"
        if s >= 0.25:
            return "████     BULL"
        if s >= -0.25:
            return "██       NEUTRAL"
        if s >= -0.75:
            return "▌        BEAR"
        return "         STRONG BEAR"

    for key in ("news_sentiment", "spy_trend", "qqq_trend", "vix", "breadth"):
        comp = components.get(key, {})
        label = comp.get("label", key)
        s = comp.get("score", 0.0)
        weight = comp.get("weight", 0)

        bar = _score_to_bar(s)
        extra = ""
        if key == "spy_trend" and comp.get("pct_change") is not None:
            extra = f"  SPY {comp['pct_change']:+.2f}%"
        elif key == "qqq_trend" and comp.get("pct_change") is not None:
            extra = f"  QQQ {comp['pct_change']:+.2f}%"
        elif key == "vix" and comp.get("vix_price") is not None:
            extra = f"  VIX {comp['vix_price']:.1f}"

        lines.append(f"  {label:<22} [{weight:2d}%] {bar}{extra}")

    # Narrative
    lines.append("")
    lines.append("ANALYSIS")
    lines.append("-" * 30)
    narrative = _build_narrative(regime, confidence, components)
    lines.append(narrative)

    return "\n".join(lines)


def _build_narrative(
    regime: str,
    confidence: float,
    components: Dict,
) -> str:
    """Build a short human-readable summary paragraph."""
    parts: List[str] = []

    vix_comp = components.get("vix", {})
    vix_score = vix_comp.get("score", 0.0)
    vix_price = vix_comp.get("vix_price")

    spy_comp = components.get("spy_trend", {})
    spy_pct = spy_comp.get("pct_change")

    news_comp = components.get("news_sentiment", {})
    news_score = news_comp.get("score", 0.0)

    breadth_comp = components.get("breadth", {})
    breadth_score = breadth_comp.get("score", 0.0)

    # Index
    if spy_pct is not None:
        direction = "advancing" if spy_pct > 0 else "declining"
        parts.append(f"Equities are {direction} with SPY {spy_pct:+.2f}%.")

    # VIX
    if vix_price is not None:
        if vix_price > 30:
            parts.append(f"VIX at {vix_price:.1f} signals elevated fear.")
        elif vix_price > 20:
            parts.append(f"VIX at {vix_price:.1f} indicates elevated caution.")
        else:
            parts.append(f"VIX at {vix_price:.1f} reflects a calm environment.")

    # News
    if news_score >= 0.3:
        parts.append("News flow is predominantly bullish.")
    elif news_score <= -0.3:
        parts.append("News flow is predominantly bearish.")
    else:
        parts.append("News flow is mixed.")

    # Breadth
    if breadth_score >= 0.4:
        parts.append("Broad market participation supports the move.")
    elif breadth_score <= -0.4:
        parts.append("Market breadth is narrow — limited participation.")

    # Confidence note
    if confidence >= 80:
        parts.append(f"Signal alignment is strong ({confidence:.0f}% confidence).")
    elif confidence <= 55:
        parts.append(
            f"Conflicting signals reduce conviction ({confidence:.0f}% confidence)."
        )

    return " ".join(parts) if parts else f"Regime: {regime}. Confidence: {confidence:.0f}%."
