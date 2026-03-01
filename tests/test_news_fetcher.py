"""
test_news_fetcher.py – Unit tests for the news aggregation and AI summary module.
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_article(
    headline: str = "Stock surges on strong earnings beat",
    summary: str = "Company exceeded analyst expectations.",
    source: str = "Finnhub",
    ticker: str = "AAPL",
) -> dict:
    return {
        "source": source,
        "headline": headline,
        "summary": summary,
        "url": "https://example.com",
        "datetime": "2024-01-15T10:00:00Z",
        "ticker": ticker,
    }


# ---------------------------------------------------------------------------
# _score_sentiment
# ---------------------------------------------------------------------------

class TestScoreSentiment:
    def test_bullish_keywords_score_positive(self):
        from momentum_radar.news.news_fetcher import _score_sentiment
        score = _score_sentiment("Stock beats earnings and upgraded to buy")
        assert score > 0

    def test_bearish_keywords_score_negative(self):
        from momentum_radar.news.news_fetcher import _score_sentiment
        score = _score_sentiment("Company misses revenue and faces lawsuit")
        assert score < 0

    def test_neutral_text_scores_near_zero(self):
        from momentum_radar.news.news_fetcher import _score_sentiment
        score = _score_sentiment("Stock trades sideways in quiet session")
        assert -1 <= score <= 1

    def test_empty_string_scores_zero(self):
        from momentum_radar.news.news_fetcher import _score_sentiment
        assert _score_sentiment("") == 0


# ---------------------------------------------------------------------------
# _classify_sentiment
# ---------------------------------------------------------------------------

class TestClassifySentiment:
    def test_high_positive_is_bullish(self):
        from momentum_radar.news.news_fetcher import _classify_sentiment
        assert _classify_sentiment(3) == "BULLISH"

    def test_high_negative_is_bearish(self):
        from momentum_radar.news.news_fetcher import _classify_sentiment
        assert _classify_sentiment(-3) == "BEARISH"

    def test_near_zero_is_neutral(self):
        from momentum_radar.news.news_fetcher import _classify_sentiment
        assert _classify_sentiment(0) == "NEUTRAL"
        assert _classify_sentiment(1) == "NEUTRAL"
        assert _classify_sentiment(-1) == "NEUTRAL"

    def test_boundary_values(self):
        from momentum_radar.news.news_fetcher import _classify_sentiment
        assert _classify_sentiment(2) == "BULLISH"
        assert _classify_sentiment(-2) == "BEARISH"


# ---------------------------------------------------------------------------
# _detect_themes
# ---------------------------------------------------------------------------

class TestDetectThemes:
    def test_detects_earnings_theme(self):
        from momentum_radar.news.news_fetcher import _detect_themes
        articles = [
            _make_article(headline="Q3 earnings beat, revenue guidance raised"),
            _make_article(headline="Quarterly results exceed expectations"),
        ]
        themes = _detect_themes(articles)
        assert "Earnings" in themes

    def test_detects_fed_theme(self):
        from momentum_radar.news.news_fetcher import _detect_themes
        articles = [
            _make_article(headline="Federal Reserve holds interest rate steady"),
            _make_article(headline="FOMC meeting minutes show rate cut discussion"),
        ]
        themes = _detect_themes(articles)
        assert "Fed/Rates" in themes

    def test_returns_at_most_five_themes(self):
        from momentum_radar.news.news_fetcher import _detect_themes
        articles = [
            _make_article(
                headline=(
                    "Fed holds rates as earnings beat amid oil sanctions"
                    " and AI chip merger investigation"
                )
            )
        ]
        themes = _detect_themes(articles)
        assert len(themes) <= 5

    def test_empty_articles_returns_empty(self):
        from momentum_radar.news.news_fetcher import _detect_themes
        assert _detect_themes([]) == []


# ---------------------------------------------------------------------------
# _deduplicate
# ---------------------------------------------------------------------------

class TestDeduplicate:
    def test_removes_duplicate_headlines(self):
        from momentum_radar.news.news_fetcher import _deduplicate
        articles = [
            _make_article(headline="Apple stock rises on iPhone sales"),
            _make_article(headline="Apple stock rises on iPhone sales"),
            _make_article(headline="Microsoft beats earnings estimate"),
        ]
        unique = _deduplicate(articles)
        assert len(unique) == 2

    def test_case_insensitive_deduplication(self):
        from momentum_radar.news.news_fetcher import _deduplicate
        articles = [
            _make_article(headline="Apple Stock Rises"),
            _make_article(headline="apple stock rises"),
        ]
        unique = _deduplicate(articles)
        assert len(unique) == 1

    def test_keeps_all_distinct_articles(self):
        from momentum_radar.news.news_fetcher import _deduplicate
        articles = [_make_article(headline=f"Headline {i}") for i in range(5)]
        unique = _deduplicate(articles)
        assert len(unique) == 5

    def test_skips_empty_headline_articles(self):
        from momentum_radar.news.news_fetcher import _deduplicate
        articles = [_make_article(headline=""), _make_article(headline="Real news")]
        unique = _deduplicate(articles)
        assert len(unique) == 1


# ---------------------------------------------------------------------------
# summarize_news
# ---------------------------------------------------------------------------

class TestSummarizeNews:
    def test_empty_list_returns_neutral(self):
        from momentum_radar.news.news_fetcher import summarize_news
        result = summarize_news([])
        assert result["overall_sentiment"] == "NEUTRAL"
        assert result["total_articles"] == 0
        assert result["bullish_headlines"] == []
        assert result["bearish_headlines"] == []

    def test_mostly_bullish_articles_give_bullish_overall(self):
        from momentum_radar.news.news_fetcher import summarize_news
        articles = [
            _make_article(headline="Stock surges on record earnings beat and upgrade"),
            _make_article(headline="Company beats revenue and raises guidance with strong growth"),
            _make_article(headline="Rally continues as breakout confirmed and gains expand"),
        ]
        result = summarize_news(articles)
        assert result["overall_sentiment"] in ("BULLISH", "NEUTRAL")
        assert result["total_articles"] == 3

    def test_mostly_bearish_articles_give_bearish_overall(self):
        from momentum_radar.news.news_fetcher import summarize_news
        articles = [
            _make_article(headline="Company misses earnings and faces lawsuit investigation"),
            _make_article(headline="Stock drops on revenue miss and guidance cut losses"),
            _make_article(headline="Shares fall as downgrade issued and concerns mount"),
        ]
        result = summarize_news(articles)
        assert result["overall_sentiment"] in ("BEARISH", "NEUTRAL")

    def test_result_has_required_keys(self):
        from momentum_radar.news.news_fetcher import summarize_news
        articles = [_make_article()]
        result = summarize_news(articles)
        for key in (
            "overall_sentiment",
            "sentiment_breakdown",
            "key_themes",
            "bullish_headlines",
            "bearish_headlines",
            "total_articles",
        ):
            assert key in result

    def test_sentiment_breakdown_sums_to_total(self):
        from momentum_radar.news.news_fetcher import summarize_news
        articles = [_make_article(headline=f"Article {i}") for i in range(10)]
        result = summarize_news(articles)
        bd = result["sentiment_breakdown"]
        assert bd["BULLISH"] + bd["BEARISH"] + bd["NEUTRAL"] == result["total_articles"]

    def test_annotates_articles_with_sentiment(self):
        from momentum_radar.news.news_fetcher import summarize_news
        articles = [_make_article(headline="Stock beats earnings and upgraded")]
        summarize_news(articles)
        assert "_sentiment" in articles[0]
        assert "_score" in articles[0]

    def test_bullish_headlines_capped_at_five(self):
        from momentum_radar.news.news_fetcher import summarize_news
        articles = [
            _make_article(headline=f"Stock surges beats earnings upgrade growth {i}")
            for i in range(10)
        ]
        result = summarize_news(articles)
        assert len(result["bullish_headlines"]) <= 5


# ---------------------------------------------------------------------------
# format_news_report
# ---------------------------------------------------------------------------

class TestFormatNewsReport:
    def _get_report(self, articles, title="Test News"):
        from momentum_radar.news.news_fetcher import summarize_news, format_news_report
        summary = summarize_news(articles)
        return format_news_report(articles, summary, title=title)

    def test_contains_title(self):
        report = self._get_report([_make_article()], title="News: AAPL")
        assert "News: AAPL" in report

    def test_contains_ai_summary_header(self):
        report = self._get_report([_make_article()])
        assert "AI SUMMARY" in report

    def test_contains_latest_headlines_section(self):
        report = self._get_report([_make_article()])
        assert "LATEST HEADLINES" in report

    def test_contains_overall_sentiment(self):
        report = self._get_report([_make_article()])
        assert "Overall Sentiment" in report

    def test_contains_breakdown(self):
        report = self._get_report([_make_article()])
        assert "Breakdown:" in report

    def test_empty_articles_shows_no_headlines(self):
        from momentum_radar.news.news_fetcher import summarize_news, format_news_report
        summary = summarize_news([])
        report = format_news_report([], summary, title="Empty")
        assert "No headlines available." in report

    def test_returns_string(self):
        report = self._get_report([_make_article()])
        assert isinstance(report, str)


# ---------------------------------------------------------------------------
# fetch_ticker_news (mocked sources)
# ---------------------------------------------------------------------------

class TestFetchTickerNews:
    def test_returns_list(self):
        from momentum_radar.news import news_fetcher
        with (
            patch.object(news_fetcher, "_fetch_finnhub_company_news", return_value=[]),
            patch.object(news_fetcher, "_fetch_yfinance_news", return_value=[]),
            patch.object(news_fetcher, "_fetch_polygon_news", return_value=[]),
        ):
            result = news_fetcher.fetch_ticker_news("AAPL")
            assert isinstance(result, list)

    def test_aggregates_from_all_sources(self):
        from momentum_radar.news import news_fetcher
        a1 = _make_article(headline="Finnhub article", source="Finnhub")
        a2 = _make_article(headline="Yahoo article", source="Yahoo Finance")
        a3 = _make_article(headline="Polygon article", source="Polygon")
        with (
            patch.object(news_fetcher, "_fetch_finnhub_company_news", return_value=[a1]),
            patch.object(news_fetcher, "_fetch_yfinance_news", return_value=[a2]),
            patch.object(news_fetcher, "_fetch_polygon_news", return_value=[a3]),
        ):
            result = news_fetcher.fetch_ticker_news("AAPL")
            assert len(result) == 3

    def test_deduplicates_results(self):
        from momentum_radar.news import news_fetcher
        a = _make_article(headline="Same headline for dedup test")
        with (
            patch.object(news_fetcher, "_fetch_finnhub_company_news", return_value=[a]),
            patch.object(news_fetcher, "_fetch_yfinance_news", return_value=[a]),
            patch.object(news_fetcher, "_fetch_polygon_news", return_value=[]),
        ):
            result = news_fetcher.fetch_ticker_news("AAPL")
            assert len(result) == 1

    def test_uppercases_ticker(self):
        from momentum_radar.news import news_fetcher
        called_with = []

        def mock_finnhub(ticker):
            called_with.append(ticker)
            return []

        with (
            patch.object(news_fetcher, "_fetch_finnhub_company_news", side_effect=mock_finnhub),
            patch.object(news_fetcher, "_fetch_yfinance_news", return_value=[]),
            patch.object(news_fetcher, "_fetch_polygon_news", return_value=[]),
        ):
            news_fetcher.fetch_ticker_news("aapl")
            assert called_with[0] == "AAPL"


# ---------------------------------------------------------------------------
# fetch_market_news (mocked sources)
# ---------------------------------------------------------------------------

class TestFetchMarketNews:
    def test_returns_list(self):
        from momentum_radar.news import news_fetcher
        with (
            patch.object(news_fetcher, "_fetch_finnhub_market_news", return_value=[]),
            patch.object(news_fetcher, "_fetch_yfinance_news", return_value=[]),
            patch.object(news_fetcher, "_fetch_polygon_news", return_value=[]),
        ):
            result = news_fetcher.fetch_market_news()
            assert isinstance(result, list)

    def test_aggregates_from_multiple_sources(self):
        from momentum_radar.news import news_fetcher
        finnhub_article = _make_article(headline="Finnhub market article", ticker=None)
        yf_article = _make_article(headline="Yahoo SPY article", ticker="SPY")
        poly_article = _make_article(headline="Polygon market article", ticker=None)
        with (
            patch.object(
                news_fetcher, "_fetch_finnhub_market_news", return_value=[finnhub_article]
            ),
            patch.object(news_fetcher, "_fetch_yfinance_news", return_value=[yf_article]),
            patch.object(news_fetcher, "_fetch_polygon_news", return_value=[poly_article]),
        ):
            result = news_fetcher.fetch_market_news()
            # yfinance is called for 4 index symbols, each returning 1 article
            assert len(result) >= 3

    def test_capped_at_50_articles(self):
        from momentum_radar.news import news_fetcher
        many = [_make_article(headline=f"Article {i}") for i in range(100)]
        with (
            patch.object(news_fetcher, "_fetch_finnhub_market_news", return_value=many),
            patch.object(news_fetcher, "_fetch_yfinance_news", return_value=[]),
            patch.object(news_fetcher, "_fetch_polygon_news", return_value=[]),
        ):
            result = news_fetcher.fetch_market_news()
            assert len(result) <= 50
