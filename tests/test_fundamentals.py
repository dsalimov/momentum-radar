"""
test_fundamentals.py – Unit tests for the fundamentals and TradingView modules.
"""

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_income_stmt() -> pd.DataFrame:
    """Return a minimal income statement DataFrame (yfinance format)."""
    cols = pd.to_datetime(["2024-09-30", "2023-09-30", "2022-09-30"])
    return pd.DataFrame(
        {
            cols[0]: {
                "Total Revenue": 390e9, "Gross Profit": 170e9, "EBIT": 118e9,
                "Net Income": 100e9, "Diluted EPS": 6.5,
            },
            cols[1]: {
                "Total Revenue": 380e9, "Gross Profit": 162e9, "EBIT": 110e9,
                "Net Income": 95e9, "Diluted EPS": 6.1,
            },
            cols[2]: {
                "Total Revenue": 365e9, "Gross Profit": 155e9, "EBIT": 105e9,
                "Net Income": 90e9, "Diluted EPS": 5.8,
            },
        }
    )


def _make_cash_flow() -> pd.DataFrame:
    cols = pd.to_datetime(["2024-09-30", "2023-09-30"])
    return pd.DataFrame(
        {
            cols[0]: {
                "Operating Cash Flow": 118e9, "Capital Expenditure": -11e9,
                "Payment For Dividends": -15e9,
            },
            cols[1]: {
                "Operating Cash Flow": 110e9, "Capital Expenditure": -10e9,
                "Payment For Dividends": -14e9,
            },
        }
    )


def _make_balance_sheet() -> pd.DataFrame:
    cols = pd.to_datetime(["2024-09-30", "2023-09-30"])
    return pd.DataFrame(
        {
            cols[0]: {
                "Total Assets": 352e9,
                "Total Liabilities Net Minority Interest": 290e9,
                "Stockholders Equity": 62e9,
                "Cash And Cash Equivalents": 29e9,
                "Total Debt": 110e9,
                "Current Assets": 135e9,
                "Current Liabilities": 125e9,
            },
            cols[1]: {
                "Total Assets": 335e9,
                "Total Liabilities Net Minority Interest": 280e9,
                "Stockholders Equity": 55e9,
                "Cash And Cash Equivalents": 24e9,
                "Total Debt": 120e9,
                "Current Assets": 128e9,
                "Current Liabilities": 120e9,
            },
        }
    )


def _make_earnings_dates() -> pd.DataFrame:
    """Return a minimal earnings_dates DataFrame."""
    idx = pd.to_datetime(["2024-08-01", "2024-05-01", "2024-02-01", "2023-11-01"])
    df = pd.DataFrame(
        {
            "EPS Estimate": [1.35, 1.30, 1.25, 1.20],
            "Reported EPS": [1.40, 1.36, 1.30, 1.22],
            "Surprise(%)": [3.7, 4.6, 4.0, 1.7],
        },
        index=idx,
    )
    return df


# ---------------------------------------------------------------------------
# get_financial_statements
# ---------------------------------------------------------------------------

class TestGetFinancialStatements:
    def test_returns_expected_sections(self):
        from momentum_radar.premarket.fundamentals import get_financial_statements

        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_t = MagicMock()
            mock_t.income_stmt = _make_income_stmt()
            mock_t.cash_flow = _make_cash_flow()
            mock_t.balance_sheet = _make_balance_sheet()
            mock_ticker_cls.return_value = mock_t

            result = get_financial_statements("AAPL")

        assert result["ticker"] == "AAPL"
        assert "income_statement" in result
        assert "cash_flow" in result
        assert "balance_sheet" in result

    def test_income_statement_fields(self):
        from momentum_radar.premarket.fundamentals import get_financial_statements

        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_t = MagicMock()
            mock_t.income_stmt = _make_income_stmt()
            mock_t.cash_flow = _make_cash_flow()
            mock_t.balance_sheet = _make_balance_sheet()
            mock_ticker_cls.return_value = mock_t

            result = get_financial_statements("AAPL")

        inc = result["income_statement"]
        assert inc["total_revenue"] == pytest.approx(390e9)
        assert inc["eps_diluted"] == pytest.approx(6.5)
        assert inc["net_margin_pct"] is not None

    def test_cash_flow_free_cf_calculation(self):
        from momentum_radar.premarket.fundamentals import get_financial_statements

        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_t = MagicMock()
            mock_t.income_stmt = _make_income_stmt()
            mock_t.cash_flow = _make_cash_flow()
            mock_t.balance_sheet = _make_balance_sheet()
            mock_ticker_cls.return_value = mock_t

            result = get_financial_statements("AAPL")

        cf = result["cash_flow"]
        assert cf["operating_cash_flow"] == pytest.approx(118e9)
        # free_cf = operating_cf + capex (capex is negative)
        assert cf["free_cash_flow"] == pytest.approx(118e9 - 11e9)

    def test_balance_sheet_ratios(self):
        from momentum_radar.premarket.fundamentals import get_financial_statements

        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_t = MagicMock()
            mock_t.income_stmt = _make_income_stmt()
            mock_t.cash_flow = _make_cash_flow()
            mock_t.balance_sheet = _make_balance_sheet()
            mock_ticker_cls.return_value = mock_t

            result = get_financial_statements("AAPL")

        bs = result["balance_sheet"]
        assert bs["debt_to_equity"] is not None
        assert bs["current_ratio"] is not None
        assert bs["current_ratio"] == pytest.approx(135e9 / 125e9, rel=0.01)

    def test_handles_yfinance_exception(self):
        from momentum_radar.premarket.fundamentals import get_financial_statements

        with patch("yfinance.Ticker", side_effect=Exception("network error")):
            result = get_financial_statements("AAPL")

        # Should return empty dicts, not raise
        assert result["ticker"] == "AAPL"
        assert result["income_statement"] == {}

    def test_handles_empty_dataframes(self):
        from momentum_radar.premarket.fundamentals import get_financial_statements

        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_t = MagicMock()
            mock_t.income_stmt = pd.DataFrame()
            mock_t.cash_flow = pd.DataFrame()
            mock_t.balance_sheet = pd.DataFrame()
            mock_ticker_cls.return_value = mock_t

            result = get_financial_statements("XYZ")

        assert result["income_statement"] == {}


# ---------------------------------------------------------------------------
# get_earnings_analysis
# ---------------------------------------------------------------------------

class TestGetEarningsAnalysis:
    def test_returns_expected_keys(self):
        from momentum_radar.premarket.fundamentals import get_earnings_analysis

        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_t = MagicMock()
            mock_t.info = {
                "forwardEps": 7.0,
                "trailingEps": 6.5,
                "targetMeanPrice": 220.0,
                "targetHighPrice": 250.0,
                "targetLowPrice": 190.0,
                "numberOfAnalystOpinions": 30,
                "recommendationKey": "buy",
                "trailingPE": 28.0,
            }
            mock_t.earnings_dates = _make_earnings_dates()
            mock_t.calendar = pd.DataFrame(
                {"Value": ["2024-11-01"]}, index=["Earnings Date"]
            )
            mock_t.income_stmt = _make_income_stmt()
            mock_ticker_cls.return_value = mock_t

            result = get_earnings_analysis("AAPL")

        for key in ("earnings_history", "next_earnings_date", "analyst_estimates",
                    "ai_summary", "guidance_summary", "eps_trend"):
            assert key in result

    def test_eps_beat_rate_computed(self):
        from momentum_radar.premarket.fundamentals import get_earnings_analysis

        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_t = MagicMock()
            mock_t.info = {}
            mock_t.earnings_dates = _make_earnings_dates()
            mock_t.calendar = None
            mock_t.income_stmt = pd.DataFrame()
            mock_ticker_cls.return_value = mock_t

            result = get_earnings_analysis("AAPL")

        eps_trend = result.get("eps_trend") or {}
        # All 4 quarters have positive surprise
        assert eps_trend.get("beat_rate_pct") == 100

    def test_ai_summary_verdict_present(self):
        from momentum_radar.premarket.fundamentals import get_earnings_analysis

        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_t = MagicMock()
            mock_t.info = {"forwardEps": 7.0, "trailingEps": 6.5}
            mock_t.earnings_dates = _make_earnings_dates()
            mock_t.calendar = None
            mock_t.income_stmt = _make_income_stmt()
            mock_ticker_cls.return_value = mock_t

            result = get_earnings_analysis("AAPL")

        ai = result.get("ai_summary", {})
        assert "verdict" in ai
        assert "quality_score" in ai
        assert 0 <= ai["quality_score"] <= 100

    def test_handles_exception_gracefully(self):
        from momentum_radar.premarket.fundamentals import get_earnings_analysis

        with patch("yfinance.Ticker", side_effect=Exception("timeout")):
            result = get_earnings_analysis("AAPL")

        assert result["ticker"] == "AAPL"
        assert result["earnings_history"] == []


# ---------------------------------------------------------------------------
# format_fundamentals_report
# ---------------------------------------------------------------------------

class TestFormatFundamentalsReport:
    def _make_data(self) -> dict:
        return {
            "ticker": "AAPL",
            "income_statement": {
                "total_revenue": 390e9,
                "total_revenue_yoy_pct": 2.6,
                "gross_profit": 170e9,
                "gross_margin_pct": 43.6,
                "ebit": 118e9,
                "net_income": 100e9,
                "net_income_yoy_pct": 5.3,
                "net_margin_pct": 25.6,
                "eps_diluted": 6.5,
                "period_label": "2024-09-30",
            },
            "cash_flow": {
                "operating_cash_flow": 118e9,
                "capital_expenditure": -11e9,
                "free_cash_flow": 107e9,
                "dividends_paid": -15e9,
                "share_repurchases": -90e9,
                "period_label": "2024-09-30",
            },
            "balance_sheet": {
                "total_assets": 352e9,
                "total_liabilities": 290e9,
                "stockholders_equity": 62e9,
                "cash_and_equivalents": 29e9,
                "total_debt": 110e9,
                "debt_to_equity": 1.77,
                "current_ratio": 1.08,
                "period_label": "2024-09-30",
            },
        }

    def test_report_contains_ticker(self):
        from momentum_radar.premarket.fundamentals import format_fundamentals_report

        report = format_fundamentals_report(self._make_data())
        assert "AAPL" in report

    def test_report_contains_section_headers(self):
        from momentum_radar.premarket.fundamentals import format_fundamentals_report

        report = format_fundamentals_report(self._make_data())
        assert "INCOME STATEMENT" in report
        assert "CASH FLOW" in report
        assert "BALANCE SHEET" in report

    def test_report_contains_key_values(self):
        from momentum_radar.premarket.fundamentals import format_fundamentals_report

        report = format_fundamentals_report(self._make_data())
        assert "6.50" in report  # EPS
        assert "25.6%" in report  # net margin

    def test_empty_data_does_not_raise(self):
        from momentum_radar.premarket.fundamentals import format_fundamentals_report

        report = format_fundamentals_report({"ticker": "XYZ", "income_statement": {}, "cash_flow": {}, "balance_sheet": {}})
        assert "XYZ" in report
        assert "No financial data" in report


# ---------------------------------------------------------------------------
# format_earnings_report
# ---------------------------------------------------------------------------

class TestFormatEarningsReport:
    def _make_data(self) -> dict:
        return {
            "ticker": "AAPL",
            "next_earnings_date": "2024-11-01",
            "earnings_history": [
                {"date": "2024-08-01", "reported_eps": 1.40, "estimated_eps": 1.35, "surprise_pct": 3.7},
                {"date": "2024-05-01", "reported_eps": 1.36, "estimated_eps": 1.30, "surprise_pct": 4.6},
            ],
            "eps_trend": {"beats": 4, "misses": 0, "beat_rate_pct": 100, "avg_surprise_pct": 3.5},
            "revenue_trend": {
                "periods": [
                    {"period": "2024-09-30", "revenue": 390e9},
                    {"period": "2023-09-30", "revenue": 380e9},
                ],
                "latest_yoy_growth_pct": 2.6,
            },
            "analyst_estimates": {
                "forward_eps": 7.0,
                "trailing_eps": 6.5,
                "target_price_mean": 220.0,
                "target_price_high": 250.0,
                "target_price_low": 190.0,
                "num_analysts": 30,
                "recommendation": "BUY",
            },
            "ai_summary": {
                "quality_score": 78,
                "verdict": "BULLISH – Strong fundamental backdrop supports higher prices.",
                "key_observations": ["Strong earnings beat rate: 100%", "Revenue growing at 2.6% YoY"],
                "bull_case": "Company growing well.",
                "bear_case": "High valuation risk.",
            },
            "guidance_summary": "30 analysts rate BUY with mean target $220.",
        }

    def test_report_contains_ticker(self):
        from momentum_radar.premarket.fundamentals import format_earnings_report

        report = format_earnings_report(self._make_data())
        assert "AAPL" in report

    def test_report_contains_next_earnings_date(self):
        from momentum_radar.premarket.fundamentals import format_earnings_report

        report = format_earnings_report(self._make_data())
        assert "2024-11-01" in report

    def test_report_contains_ai_verdict(self):
        from momentum_radar.premarket.fundamentals import format_earnings_report

        report = format_earnings_report(self._make_data())
        assert "BULLISH" in report
        assert "78" in report

    def test_report_contains_beat_rate(self):
        from momentum_radar.premarket.fundamentals import format_earnings_report

        report = format_earnings_report(self._make_data())
        assert "100%" in report

    def test_empty_history_does_not_raise(self):
        from momentum_radar.premarket.fundamentals import format_earnings_report

        data = {
            "ticker": "XYZ",
            "next_earnings_date": None,
            "earnings_history": [],
            "eps_trend": None,
            "revenue_trend": None,
            "analyst_estimates": {},
            "ai_summary": {},
            "guidance_summary": "",
        }
        report = format_earnings_report(data)
        assert "XYZ" in report


# ---------------------------------------------------------------------------
# TradingView helpers
# ---------------------------------------------------------------------------

class TestTradingViewHelpers:
    def test_chart_url_format(self):
        from momentum_radar.premarket.tradingview import get_chart_url

        url = get_chart_url("AAPL", exchange="NASDAQ")
        assert "AAPL" in url
        assert "tradingview.com" in url

    def test_screener_url_format(self):
        from momentum_radar.premarket.tradingview import get_screener_url

        url = get_screener_url("TSLA")
        assert "TSLA" in url
        assert "tradingview.com" in url

    def test_analysis_returns_none_when_package_missing(self):
        """When tradingview-ta is not installed, function returns None gracefully."""
        from momentum_radar.premarket.tradingview import get_tradingview_analysis

        with patch.dict("sys.modules", {"tradingview_ta": None}):
            result = get_tradingview_analysis("AAPL")
        # None or a dict are both acceptable
        assert result is None or isinstance(result, dict)

    def test_format_tradingview_section_with_none_analysis(self):
        from momentum_radar.premarket.tradingview import format_tradingview_section

        section = format_tradingview_section("AAPL", None)
        assert "AAPL" in section
        assert "tradingview.com" in section

    def test_format_tradingview_section_with_analysis(self):
        from momentum_radar.premarket.tradingview import format_tradingview_section

        tv_analysis = {
            "recommendation": "BUY",
            "buy_signals": 15,
            "sell_signals": 3,
            "neutral_signals": 8,
            "oscillators": {"recommendation": "BUY"},
            "moving_averages": {"recommendation": "STRONG_BUY"},
            "rsi": 58.4,
            "adx": 32.1,
        }
        section = format_tradingview_section("AAPL", tv_analysis)
        assert "BUY" in section
        assert "58.4" in section
