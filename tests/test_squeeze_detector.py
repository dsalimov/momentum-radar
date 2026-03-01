"""
test_squeeze_detector.py – Unit tests for the short squeeze detection engine.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock


def _make_daily(n: int = 65, trend: str = "flat") -> pd.DataFrame:
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    np.random.seed(42)
    if trend == "up":
        closes = 100 + np.arange(n) * 0.5
    else:
        closes = 100 + np.zeros(n)
    closes = closes + np.random.randn(n) * 0.1
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


def _make_fetcher(daily=None, fundamentals=None, options=None, quote=None):
    fetcher = MagicMock()
    fetcher.get_daily_bars.return_value = daily if daily is not None else _make_daily()
    fetcher.get_fundamentals.return_value = fundamentals if fundamentals is not None else {
        "float_shares": 30_000_000,
        "short_percent_of_float": 0.25,
        "short_ratio": 4.5,
        "shares_outstanding": 40_000_000,
    }
    fetcher.get_options_volume.return_value = options if options is not None else {
        "call_volume": 6_000,
        "put_volume": 2_000,
        "avg_call_volume": 2_000,
        "avg_put_volume": 1_000,
    }
    fetcher.get_quote.return_value = quote if quote is not None else {
        "price": 55.0,
        "prev_close": 52.0,
    }
    fetcher.get_intraday_bars.return_value = None
    return fetcher


# ---------------------------------------------------------------------------
# compute_squeeze_score
# ---------------------------------------------------------------------------

class TestComputeSqueezeScore:
    def test_max_score_all_factors(self):
        from momentum_radar.premarket.squeeze_detector import compute_squeeze_score

        result = compute_squeeze_score(
            short_pct=0.30,       # > 20% → +20
            days_to_cover=5.0,    # > 3  → +15
            float_shares=20e6,    # < 50M → +15
            call_spike=3.0,       # > 2x → +15
            rvol=4.0,             # > 3x → +10
            cp_ratio=3.0,         # ≥ 2  → +10
            breakout=True,        # → +15
        )
        assert result["score"] == 100
        assert result["probability_pct"] == 100

    def test_zero_score_no_factors(self):
        from momentum_radar.premarket.squeeze_detector import compute_squeeze_score

        result = compute_squeeze_score(
            short_pct=0.05,
            days_to_cover=1.0,
            float_shares=500e6,
            call_spike=0.5,
            rvol=0.8,
            cp_ratio=0.5,
            breakout=False,
        )
        assert result["score"] == 0
        assert result["label"] == "NONE – Insufficient criteria"

    def test_partial_score(self):
        from momentum_radar.premarket.squeeze_detector import compute_squeeze_score

        # Only short_interest and days_to_cover trigger → 20 + 15 = 35
        result = compute_squeeze_score(
            short_pct=0.25,
            days_to_cover=4.0,
            float_shares=None,
            call_spike=None,
            rvol=None,
            cp_ratio=None,
            breakout=False,
        )
        assert result["score"] == 35

    def test_high_label_above_70(self):
        from momentum_radar.premarket.squeeze_detector import compute_squeeze_score

        result = compute_squeeze_score(
            short_pct=0.30,
            days_to_cover=5.0,
            float_shares=20e6,
            call_spike=3.0,
            rvol=4.0,
            cp_ratio=3.0,
            breakout=True,
        )
        assert "HIGH" in result["label"]

    def test_medium_label(self):
        from momentum_radar.premarket.squeeze_detector import compute_squeeze_score

        # 20 + 15 + 15 = 50 → MEDIUM
        result = compute_squeeze_score(
            short_pct=0.22,
            days_to_cover=3.5,
            float_shares=30e6,  # < 50M → +15
            call_spike=None,
            rvol=None,
            cp_ratio=None,
            breakout=False,
        )
        assert "MEDIUM" in result["label"]

    def test_factors_list_populated(self):
        from momentum_radar.premarket.squeeze_detector import compute_squeeze_score

        result = compute_squeeze_score(
            short_pct=0.25,
            days_to_cover=4.0,
            float_shares=40e6,
            call_spike=None,
            rvol=None,
            cp_ratio=None,
            breakout=False,
        )
        assert len(result["factors"]) >= 3

    def test_none_inputs_do_not_crash(self):
        from momentum_radar.premarket.squeeze_detector import compute_squeeze_score

        result = compute_squeeze_score(
            short_pct=None,
            days_to_cover=None,
            float_shares=None,
            call_spike=None,
            rvol=None,
            cp_ratio=None,
            breakout=False,
        )
        assert isinstance(result["score"], int)
        assert result["score"] == 0


# ---------------------------------------------------------------------------
# build_squeeze_report
# ---------------------------------------------------------------------------

class TestBuildSqueezeReport:
    def test_returns_dict_with_ticker(self):
        from momentum_radar.premarket.squeeze_detector import build_squeeze_report

        fetcher = _make_fetcher()
        report = build_squeeze_report("GME", fetcher)
        assert report is not None
        assert report["ticker"] == "GME"

    def test_returns_none_on_no_data(self):
        from momentum_radar.premarket.squeeze_detector import build_squeeze_report

        fetcher = MagicMock()
        fetcher.get_daily_bars.return_value = None
        fetcher.get_fundamentals.return_value = None
        fetcher.get_options_volume.return_value = None
        fetcher.get_quote.return_value = None
        fetcher.get_intraday_bars.return_value = None

        report = build_squeeze_report("NONE", fetcher)
        assert report is None

    def test_squeeze_score_present(self):
        from momentum_radar.premarket.squeeze_detector import build_squeeze_report

        fetcher = _make_fetcher(
            fundamentals={
                "float_shares": 20e6,
                "short_percent_of_float": 0.30,
                "short_ratio": 5.0,
                "shares_outstanding": 25e6,
            }
        )
        report = build_squeeze_report("AMC", fetcher)
        assert report is not None
        assert 0 <= report["squeeze_score"] <= 100

    def test_required_fields(self):
        from momentum_radar.premarket.squeeze_detector import build_squeeze_report

        fetcher = _make_fetcher()
        report = build_squeeze_report("AAPL", fetcher)
        assert report is not None
        for field in (
            "ticker", "squeeze_score", "squeeze_probability_pct",
            "squeeze_label", "squeeze_factors",
        ):
            assert field in report


# ---------------------------------------------------------------------------
# format_squeeze_report
# ---------------------------------------------------------------------------

class TestFormatSqueezeReport:
    def test_contains_ticker(self):
        from momentum_radar.premarket.squeeze_detector import (
            build_squeeze_report,
            format_squeeze_report,
        )

        fetcher = _make_fetcher()
        report = build_squeeze_report("BBBY", fetcher)
        assert report is not None
        text = format_squeeze_report(report)
        assert "BBBY" in text
        assert "Squeeze Probability" in text

    def test_contains_scenario_sections(self):
        from momentum_radar.premarket.squeeze_detector import (
            build_squeeze_report,
            format_squeeze_report,
        )

        fetcher = _make_fetcher()
        report = build_squeeze_report("GME", fetcher)
        assert report is not None
        text = format_squeeze_report(report)
        assert "Bull Case" in text
        assert "Bear Case" in text


# ---------------------------------------------------------------------------
# scan_squeeze_candidates
# ---------------------------------------------------------------------------

class TestScanSqueezeCandiates:
    def test_returns_sorted_by_score(self):
        from momentum_radar.premarket.squeeze_detector import scan_squeeze_candidates

        fetcher = _make_fetcher(
            fundamentals={
                "float_shares": 20e6,
                "short_percent_of_float": 0.30,
                "short_ratio": 5.0,
                "shares_outstanding": 25e6,
            }
        )
        results = scan_squeeze_candidates(["A", "B"], fetcher, min_score=0, top_n=5)
        if len(results) >= 2:
            assert results[0]["squeeze_score"] >= results[1]["squeeze_score"]

    def test_min_score_filter(self):
        from momentum_radar.premarket.squeeze_detector import scan_squeeze_candidates

        fetcher = _make_fetcher(
            fundamentals={
                "float_shares": 500e6,   # large float → low score
                "short_percent_of_float": 0.02,
                "short_ratio": 1.0,
                "shares_outstanding": 600e6,
            },
            options={
                "call_volume": 100,
                "put_volume": 100,
                "avg_call_volume": 100,
                "avg_put_volume": 100,
            },
        )
        results = scan_squeeze_candidates(["LOW"], fetcher, min_score=50, top_n=5)
        assert all(r["squeeze_score"] >= 50 for r in results)
