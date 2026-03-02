"""
tests/test_squeeze_engine_service.py – Unit tests for the squeeze engine service facade.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock


def _make_daily(n: int = 65) -> pd.DataFrame:
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    np.random.seed(99)
    closes = 100 + np.arange(n) * 0.5 + np.random.randn(n) * 0.1
    return pd.DataFrame(
        {
            "open": closes - 0.3,
            "high": closes + 0.8,
            "low": closes - 0.8,
            "close": closes,
            "volume": np.random.randint(1_000_000, 3_000_000, size=n).astype(float),
        },
        index=rng,
    )


def _make_fetcher():
    fetcher = MagicMock()
    fetcher.get_daily_bars.return_value = _make_daily()
    fetcher.get_fundamentals.return_value = {
        "float_shares": 20_000_000,
        "short_percent_of_float": 0.28,
        "short_ratio": 4.0,
        "shares_outstanding": 25_000_000,
    }
    fetcher.get_options_volume.return_value = {
        "call_volume": 8_000,
        "put_volume": 2_000,
        "avg_call_volume": 2_000,
        "avg_put_volume": 1_000,
    }
    fetcher.get_quote.return_value = {"price": 55.0, "prev_close": 52.0}
    fetcher.get_intraday_bars.return_value = None
    return fetcher


# ---------------------------------------------------------------------------
# score_ticker
# ---------------------------------------------------------------------------

class TestScoreTicker:
    def test_returns_report(self):
        from momentum_radar.services.squeeze_engine import score_ticker

        fetcher = _make_fetcher()
        report = score_ticker("GME", fetcher)
        assert report is not None
        assert report["ticker"] == "GME"
        assert "squeeze_score" in report

    def test_returns_none_on_no_data(self):
        from momentum_radar.services.squeeze_engine import score_ticker

        fetcher = MagicMock()
        fetcher.get_daily_bars.return_value = None
        fetcher.get_fundamentals.return_value = None
        fetcher.get_options_volume.return_value = None
        fetcher.get_quote.return_value = None
        fetcher.get_intraday_bars.return_value = None

        report = score_ticker("NONE", fetcher)
        assert report is None


# ---------------------------------------------------------------------------
# scan_universe
# ---------------------------------------------------------------------------

class TestScanUniverse:
    def test_returns_list(self):
        from momentum_radar.services.squeeze_engine import scan_universe

        fetcher = _make_fetcher()
        results = scan_universe(["A", "B"], fetcher, min_score=0, top_n=5)
        assert isinstance(results, list)

    def test_sorted_by_score(self):
        from momentum_radar.services.squeeze_engine import scan_universe

        fetcher = _make_fetcher()
        results = scan_universe(["A", "B", "C"], fetcher, min_score=0, top_n=10)
        scores = [r["squeeze_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_min_score_filter_applied(self):
        from momentum_radar.services.squeeze_engine import scan_universe

        # Make a fetcher that produces near-zero scores
        fetcher = MagicMock()
        fetcher.get_daily_bars.return_value = _make_daily()
        fetcher.get_fundamentals.return_value = {
            "float_shares": 1_000_000_000,
            "short_percent_of_float": 0.01,
            "short_ratio": 0.5,
        }
        fetcher.get_options_volume.return_value = {
            "call_volume": 100, "put_volume": 100,
            "avg_call_volume": 100, "avg_put_volume": 100,
        }
        fetcher.get_quote.return_value = {"price": 50.0, "prev_close": 50.0}
        fetcher.get_intraday_bars.return_value = None

        results = scan_universe(["LOW"], fetcher, min_score=90, top_n=5)
        assert all(r["squeeze_score"] >= 90 for r in results)


# ---------------------------------------------------------------------------
# format_alert_text
# ---------------------------------------------------------------------------

class TestFormatAlertText:
    def _make_report(self, score: int = 80) -> dict:
        return {
            "ticker": "TSLA",
            "squeeze_score": score,
            "short_interest_pct": 0.25,
            "days_to_cover": 3.5,
            "float_shares": 30_000_000,
            "float_str": "30M",
            "rvol": 3.4,
            "cp_ratio": 2.8,
            "borrow_fee_estimate": 0.10,
            "breakout_level": 14.50,
            "resistance": 17.80,
            "bull_target1": 18.20,
            "bull_target2": 20.00,
            "bear_target": 13.70,
            "atr": 1.20,
            "current_price": 15.00,
        }

    def test_high_score_uses_fire_header(self):
        from momentum_radar.services.squeeze_engine import format_alert_text

        text = format_alert_text(self._make_report(score=80))
        assert "🔥" in text or "🚨" in text

    def test_three_confirmations_uses_siren_header(self):
        from momentum_radar.services.squeeze_engine import format_alert_text

        text = format_alert_text(
            self._make_report(score=80),
            confirmations=["Volume 3.4x average", "Breakout", "Call flow spike"],
        )
        assert "🚨" in text

    def test_two_confirmations_uses_fire_header(self):
        from momentum_radar.services.squeeze_engine import format_alert_text

        text = format_alert_text(
            self._make_report(score=60),
            confirmations=["Volume 2.1x average", "Breakout"],
        )
        assert "🔥" in text

    def test_confirmations_shown_as_checkmarks(self):
        from momentum_radar.services.squeeze_engine import format_alert_text

        text = format_alert_text(
            self._make_report(),
            confirmations=["Volume spike", "Breakout"],
        )
        assert "✔ Volume spike" in text
        assert "✔ Breakout" in text

    def test_risk_warning_included(self):
        from momentum_radar.services.squeeze_engine import format_alert_text

        text = format_alert_text(self._make_report())
        assert "risk management" in text.lower() or "not investment advice" in text.lower()

    def test_contains_key_fields(self):
        from momentum_radar.services.squeeze_engine import format_alert_text

        text = format_alert_text(self._make_report())
        assert "Short Interest" in text
        assert "Float" in text
        assert "Call/Put Ratio" in text
        assert "Invalidation" in text
