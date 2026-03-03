"""
test_sentiment_engine.py – Unit tests for the market sentiment engine.
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fetcher(spy_pct=0.5, qqq_pct=0.3, vix=18.0):
    """Create a mock fetcher that returns configured quote values."""
    fetcher = MagicMock()

    def get_quote(sym):
        if sym == "SPY":
            prev = 400.0
            price = prev * (1 + spy_pct / 100)
            return {"price": price, "prev_close": prev}
        if sym == "QQQ":
            prev = 350.0
            price = prev * (1 + qqq_pct / 100)
            return {"price": price, "prev_close": prev}
        if sym == "VIX":
            return {"price": vix, "prev_close": vix - 1}
        # Breadth ETFs – return slightly positive
        return {"price": 50.0, "prev_close": 49.5}

    fetcher.get_quote.side_effect = get_quote
    return fetcher


# ---------------------------------------------------------------------------
# _score_index
# ---------------------------------------------------------------------------

class TestScoreIndex:
    def test_strong_positive_returns_one(self):
        from momentum_radar.services.sentiment_engine import _score_index
        assert _score_index(2.0) == 1.0

    def test_moderate_positive_returns_half(self):
        from momentum_radar.services.sentiment_engine import _score_index
        assert _score_index(1.0) == 0.5

    def test_near_zero_returns_zero(self):
        from momentum_radar.services.sentiment_engine import _score_index
        assert _score_index(0.0) == 0.0
        assert _score_index(0.3) == 0.0

    def test_moderate_negative_returns_neg_half(self):
        from momentum_radar.services.sentiment_engine import _score_index
        assert _score_index(-1.0) == -0.5

    def test_strong_negative_returns_neg_one(self):
        from momentum_radar.services.sentiment_engine import _score_index
        assert _score_index(-2.0) == -1.0

    def test_none_returns_zero(self):
        from momentum_radar.services.sentiment_engine import _score_index
        assert _score_index(None) == 0.0


# ---------------------------------------------------------------------------
# _score_vix
# ---------------------------------------------------------------------------

class TestScoreVix:
    def test_low_vix_is_bullish(self):
        from momentum_radar.services.sentiment_engine import _score_vix
        assert _score_vix(12.0) == 1.0

    def test_moderate_vix_is_bullish(self):
        from momentum_radar.services.sentiment_engine import _score_vix
        assert _score_vix(17.0) == 0.5

    def test_neutral_vix(self):
        from momentum_radar.services.sentiment_engine import _score_vix
        assert _score_vix(22.0) == 0.0

    def test_elevated_vix_is_bearish(self):
        from momentum_radar.services.sentiment_engine import _score_vix
        assert _score_vix(27.0) == -0.5

    def test_fear_vix_is_risk_off(self):
        from momentum_radar.services.sentiment_engine import _score_vix
        assert _score_vix(35.0) == -1.0

    def test_none_returns_zero(self):
        from momentum_radar.services.sentiment_engine import _score_vix
        assert _score_vix(None) == 0.0


# ---------------------------------------------------------------------------
# _classify_regime
# ---------------------------------------------------------------------------

class TestClassifyRegime:
    def test_high_positive_is_strong_bullish(self):
        from momentum_radar.services.sentiment_engine import _classify_regime
        assert _classify_regime(0.7) == "Strong Bullish"

    def test_moderate_positive_is_bullish(self):
        from momentum_radar.services.sentiment_engine import _classify_regime
        assert _classify_regime(0.4) == "Bullish"

    def test_near_zero_is_neutral(self):
        from momentum_radar.services.sentiment_engine import _classify_regime
        assert _classify_regime(0.0) == "Neutral"
        assert _classify_regime(0.1) == "Neutral"
        assert _classify_regime(-0.1) == "Neutral"

    def test_moderate_negative_is_bearish(self):
        from momentum_radar.services.sentiment_engine import _classify_regime
        assert _classify_regime(-0.4) == "Bearish"

    def test_strong_negative_is_risk_off(self):
        from momentum_radar.services.sentiment_engine import _classify_regime
        assert _classify_regime(-0.7) == "Risk-Off"

    def test_boundary_values(self):
        from momentum_radar.services.sentiment_engine import _classify_regime
        assert _classify_regime(0.6) == "Strong Bullish"
        assert _classify_regime(0.2) == "Bullish"
        assert _classify_regime(-0.2) == "Neutral"
        assert _classify_regime(-0.6) == "Bearish"


# ---------------------------------------------------------------------------
# _compute_confidence
# ---------------------------------------------------------------------------

class TestComputeConfidence:
    def test_perfect_agreement_gives_high_confidence(self):
        from momentum_radar.services.sentiment_engine import _compute_confidence
        # All signals agree (same value)
        scores = {"news": 1.0, "spy": 1.0, "qqq": 1.0, "vix": 1.0, "breadth": 1.0}
        conf = _compute_confidence(scores)
        assert conf >= 90.0

    def test_maximum_disagreement_gives_lower_confidence(self):
        from momentum_radar.services.sentiment_engine import _compute_confidence
        # Alternating extremes
        scores = {"a": 1.0, "b": -1.0, "c": 1.0, "d": -1.0}
        conf = _compute_confidence(scores)
        assert conf < 75.0

    def test_returns_value_between_0_and_100(self):
        from momentum_radar.services.sentiment_engine import _compute_confidence
        scores = {"a": 0.5, "b": -0.5, "c": 0.0}
        conf = _compute_confidence(scores)
        assert 0.0 <= conf <= 100.0

    def test_empty_scores_returns_50(self):
        from momentum_radar.services.sentiment_engine import _compute_confidence
        conf = _compute_confidence({})
        assert conf == 50.0


# ---------------------------------------------------------------------------
# get_market_sentiment
# ---------------------------------------------------------------------------

class TestGetMarketSentiment:
    def test_returns_required_keys(self):
        from momentum_radar.services.sentiment_engine import get_market_sentiment
        fetcher = _make_fetcher()
        with patch("momentum_radar.services.sentiment_engine._score_news_sentiment", return_value=0.2):
            result = get_market_sentiment(fetcher)
        assert "market_regime" in result
        assert "confidence_pct" in result
        assert "weighted_score" in result
        assert "components" in result
        assert "timestamp" in result

    def test_bullish_market_gives_positive_regime(self):
        from momentum_radar.services.sentiment_engine import get_market_sentiment
        fetcher = _make_fetcher(spy_pct=1.8, qqq_pct=2.0, vix=13.0)
        with patch("momentum_radar.services.sentiment_engine._score_news_sentiment", return_value=0.5):
            result = get_market_sentiment(fetcher)
        assert result["market_regime"] in ("Strong Bullish", "Bullish")

    def test_bearish_market_gives_negative_regime(self):
        from momentum_radar.services.sentiment_engine import get_market_sentiment
        fetcher = _make_fetcher(spy_pct=-2.0, qqq_pct=-2.5, vix=35.0)
        with patch("momentum_radar.services.sentiment_engine._score_news_sentiment", return_value=-0.5):
            # Also make breadth negative
            fetcher_bearish = MagicMock()

            def get_quote_bearish(sym):
                if sym == "SPY":
                    return {"price": 390.0, "prev_close": 400.0}
                if sym == "QQQ":
                    return {"price": 340.0, "prev_close": 350.0}
                if sym == "VIX":
                    return {"price": 35.0, "prev_close": 30.0}
                return {"price": 48.0, "prev_close": 50.0}

            fetcher_bearish.get_quote.side_effect = get_quote_bearish
            result = get_market_sentiment(fetcher_bearish)
        assert result["market_regime"] in ("Bearish", "Risk-Off", "Neutral")

    def test_confidence_between_0_and_100(self):
        from momentum_radar.services.sentiment_engine import get_market_sentiment
        fetcher = _make_fetcher()
        with patch("momentum_radar.services.sentiment_engine._score_news_sentiment", return_value=0.0):
            result = get_market_sentiment(fetcher)
        assert 0.0 <= result["confidence_pct"] <= 100.0

    def test_fetcher_error_degrades_gracefully(self):
        from momentum_radar.services.sentiment_engine import get_market_sentiment
        fetcher = MagicMock()
        fetcher.get_quote.side_effect = Exception("API down")
        with patch("momentum_radar.services.sentiment_engine._score_news_sentiment", return_value=0.0):
            result = get_market_sentiment(fetcher)
        # Should still return a valid result
        assert "market_regime" in result
        assert result["market_regime"] == "Neutral"


# ---------------------------------------------------------------------------
# format_sentiment_report
# ---------------------------------------------------------------------------

class TestFormatSentimentReport:
    def _get_result(self, spy_pct=0.5, qqq_pct=0.3, vix=18.0, news_score=0.2):
        from momentum_radar.services.sentiment_engine import get_market_sentiment
        fetcher = _make_fetcher(spy_pct=spy_pct, qqq_pct=qqq_pct, vix=vix)
        with patch("momentum_radar.services.sentiment_engine._score_news_sentiment", return_value=news_score):
            return get_market_sentiment(fetcher)

    def test_contains_market_regime_header(self):
        from momentum_radar.services.sentiment_engine import format_sentiment_report
        result = self._get_result()
        report = format_sentiment_report(result)
        assert "MARKET REGIME" in report

    def test_contains_signal_breakdown(self):
        from momentum_radar.services.sentiment_engine import format_sentiment_report
        result = self._get_result()
        report = format_sentiment_report(result)
        assert "SIGNAL BREAKDOWN" in report

    def test_contains_confidence(self):
        from momentum_radar.services.sentiment_engine import format_sentiment_report
        result = self._get_result()
        report = format_sentiment_report(result)
        assert "Confidence:" in report

    def test_contains_regime_name(self):
        from momentum_radar.services.sentiment_engine import format_sentiment_report
        result = self._get_result()
        report = format_sentiment_report(result)
        assert result["market_regime"] in report

    def test_returns_string(self):
        from momentum_radar.services.sentiment_engine import format_sentiment_report
        result = self._get_result()
        report = format_sentiment_report(result)
        assert isinstance(report, str)
