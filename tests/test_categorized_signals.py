"""
tests/test_categorized_signals.py – Tests for categorized signal alerts
and the per-(ticker, alert_type) database cooldown added to support
separate chart-pattern, candlestick, options-flow, and momentum signals.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def in_memory_db():
    """Initialise an in-memory SQLite database for every test."""
    from momentum_radar.storage.database import init_db
    import momentum_radar.storage.database as db_module

    init_db("sqlite:///:memory:")
    yield
    db_module._ENGINE = None
    db_module._SessionLocal = None


# ---------------------------------------------------------------------------
# should_send_signal_alert / record_signal_alert
# ---------------------------------------------------------------------------

class TestSignalCategoryRecord:
    def test_first_alert_always_allowed(self):
        from momentum_radar.storage.database import should_send_signal_alert

        assert should_send_signal_alert("AAPL", "chart_pattern") is True

    def test_recent_alert_same_type_suppressed(self):
        from momentum_radar.storage.database import (
            should_send_signal_alert,
            record_signal_alert,
        )
        record_signal_alert("AAPL", "chart_pattern")
        assert should_send_signal_alert("AAPL", "chart_pattern") is False

    def test_recent_alert_different_type_allowed(self):
        """A candlestick cooldown should NOT suppress an options_flow alert."""
        from momentum_radar.storage.database import (
            should_send_signal_alert,
            record_signal_alert,
        )
        record_signal_alert("NVDA", "candlestick")
        # Different category for same ticker – should still be allowed
        assert should_send_signal_alert("NVDA", "options_flow") is True

    def test_different_tickers_independent(self):
        from momentum_radar.storage.database import (
            should_send_signal_alert,
            record_signal_alert,
        )
        record_signal_alert("TSLA", "chart_pattern")
        # Different ticker, same type – should be allowed
        assert should_send_signal_alert("MSFT", "chart_pattern") is True

    def test_old_alert_allowed_after_cooldown(self):
        from momentum_radar.storage.database import (
            should_send_signal_alert,
            record_signal_alert,
            SignalCategoryRecord,
            _ENGINE,
        )
        from sqlalchemy.orm import Session

        record_signal_alert("GME", "squeeze_momentum")

        # Back-date the record by 5 hours (> default 4h cooldown)
        with Session(_ENGINE) as session:
            rec = session.get(SignalCategoryRecord, ("GME", "squeeze_momentum"))
            rec.last_alerted_at = datetime.utcnow() - timedelta(hours=5)
            session.commit()

        assert should_send_signal_alert("GME", "squeeze_momentum") is True

    def test_custom_cooldown_respected(self):
        from momentum_radar.storage.database import (
            should_send_signal_alert,
            record_signal_alert,
        )
        record_signal_alert("SPY", "general")
        # 1-hour custom cooldown → still suppressed immediately after recording
        assert should_send_signal_alert("SPY", "general", cooldown_hours=1) is False

    def test_record_updates_timestamp(self):
        from momentum_radar.storage.database import (
            record_signal_alert,
            SignalCategoryRecord,
            _ENGINE,
        )
        from sqlalchemy.orm import Session

        record_signal_alert("QQQ", "options_flow")
        t1 = None
        with Session(_ENGINE) as session:
            rec = session.get(SignalCategoryRecord, ("QQQ", "options_flow"))
            t1 = rec.last_alerted_at

        record_signal_alert("QQQ", "options_flow")
        with Session(_ENGINE) as session:
            rec = session.get(SignalCategoryRecord, ("QQQ", "options_flow"))
            assert rec.last_alerted_at >= t1


# ---------------------------------------------------------------------------
# Categorized alert formatters
# ---------------------------------------------------------------------------

def _make_confirmation(name, category, detail, confidence=75.0):
    from momentum_radar.services.signal_engine import Confirmation
    return Confirmation(name=name, category=category, detail=detail, confidence=confidence)


class TestFormatChartPatternAlert:
    def _fmt(self, ticker="AAPL", confs=None, price=None):
        from momentum_radar.services.scheduler import _format_chart_pattern_alert
        if confs is None:
            confs = [_make_confirmation("Double Bottom", "pattern", "Support near $150")]
        return _format_chart_pattern_alert(ticker, confs, price=price)

    def test_header_contains_ticker(self):
        assert "AAPL" in self._fmt()

    def test_header_emoji(self):
        assert "📊" in self._fmt()

    def test_header_label(self):
        assert "CHART PATTERN SIGNAL" in self._fmt()

    def test_confirmation_name_shown(self):
        assert "Double Bottom" in self._fmt()

    def test_confidence_shown(self):
        text = self._fmt(confs=[_make_confirmation("DB", "pattern", "detail", 80.0)])
        assert "80%" in text

    def test_price_included_when_provided(self):
        assert "$175.50" in self._fmt(price=175.50)

    def test_disclaimer_included(self):
        assert "risk management" in self._fmt().lower()


class TestFormatCandlestickAlert:
    def _fmt(self, ticker="TSLA", confs=None, price=None):
        from momentum_radar.services.scheduler import _format_candlestick_alert
        if confs is None:
            confs = [_make_confirmation("Bullish Engulfing", "candlestick", "Enter bullish")]
        return _format_candlestick_alert(ticker, confs, price=price)

    def test_header_contains_ticker(self):
        assert "TSLA" in self._fmt()

    def test_header_emoji(self):
        assert "🕯️" in self._fmt()

    def test_header_label(self):
        assert "CANDLESTICK PATTERN SIGNAL" in self._fmt()

    def test_pattern_name_shown(self):
        assert "Bullish Engulfing" in self._fmt()

    def test_price_omitted_when_none(self):
        text = self._fmt(price=None)
        assert "$" not in text.splitlines()[0]

    def test_disclaimer_included(self):
        assert "risk management" in self._fmt().lower()


class TestFormatOptionsFlowAlert:
    def _fmt(self, ticker="NVDA", confs=None, price=None):
        from momentum_radar.services.scheduler import _format_options_flow_alert
        if confs is None:
            confs = [_make_confirmation("Golden Sweep (Calls)", "options", "5000 contracts at $500")]
        return _format_options_flow_alert(ticker, confs, price=price)

    def test_header_contains_ticker(self):
        assert "NVDA" in self._fmt()

    def test_header_emoji(self):
        assert "📈" in self._fmt()

    def test_header_label(self):
        assert "OPTIONS FLOW SIGNAL" in self._fmt()

    def test_confirmation_detail_shown(self):
        assert "5000 contracts" in self._fmt()

    def test_disclaimer_included(self):
        assert "risk management" in self._fmt().lower()


class TestFormatMomentumAlert:
    def _fmt(self, ticker="AMZN", confs=None, price=None):
        from momentum_radar.services.scheduler import _format_momentum_alert
        if confs is None:
            confs = [
                _make_confirmation("Volume Spike", "volume", "3.4x average"),
                _make_confirmation("HTF Trend Alignment", "htf_trend", "Price > EMA21 > EMA50"),
            ]
        return _format_momentum_alert(ticker, confs, price=price)

    def test_header_contains_ticker(self):
        assert "AMZN" in self._fmt()

    def test_header_emoji(self):
        assert "💹" in self._fmt()

    def test_header_label(self):
        assert "MOMENTUM SIGNAL" in self._fmt()

    def test_all_confirmations_shown(self):
        text = self._fmt()
        assert "Volume Spike" in text
        assert "HTF Trend Alignment" in text

    def test_disclaimer_included(self):
        assert "risk management" in self._fmt().lower()


# ---------------------------------------------------------------------------
# _run_categorized_signal_scan – integration
# ---------------------------------------------------------------------------

def _make_fetcher_returning_none():
    fetcher = MagicMock()
    fetcher.get_intraday_bars.return_value = None
    fetcher.get_daily_bars.return_value = None
    fetcher.get_options_volume.return_value = None
    return fetcher


def _make_signal_result(ticker, priority, confirmations):
    """Build a minimal SignalResult mock."""
    from momentum_radar.services.signal_engine import SignalResult
    r = SignalResult(
        ticker=ticker,
        confirmations=confirmations,
        priority=priority,
        confidence_score=80.0,
    )
    return r


class TestRunCategorizedSignalScan:
    def test_no_signal_tickers_send_nothing(self):
        from momentum_radar.services.scheduler import _run_categorized_signal_scan
        fetcher = _make_fetcher_returning_none()
        sent = []

        with patch(
            "momentum_radar.services.signal_engine.evaluate",
            return_value=_make_signal_result("AAPL", "NO_SIGNAL", []),
        ):
            _run_categorized_signal_scan(["AAPL"], fetcher, sent.append)

        assert sent == []

    def test_chart_pattern_alert_sent(self):
        from momentum_radar.services.scheduler import _run_categorized_signal_scan
        fetcher = _make_fetcher_returning_none()
        sent = []

        confs = [_make_confirmation("Double Bottom", "pattern", "support near $100")]
        sig = _make_signal_result("AAPL", "ALERT", confs)

        with patch("momentum_radar.services.signal_engine.evaluate", return_value=sig):
            _run_categorized_signal_scan(["AAPL"], fetcher, sent.append)

        assert len(sent) == 1
        assert "CHART PATTERN SIGNAL" in sent[0]
        assert "AAPL" in sent[0]

    def test_candlestick_alert_sent(self):
        from momentum_radar.services.scheduler import _run_categorized_signal_scan
        fetcher = _make_fetcher_returning_none()
        sent = []

        confs = [_make_confirmation("Hammer", "candlestick", "Hammer at support")]
        sig = _make_signal_result("TSLA", "ALERT", confs)

        with patch("momentum_radar.services.signal_engine.evaluate", return_value=sig):
            _run_categorized_signal_scan(["TSLA"], fetcher, sent.append)

        assert len(sent) == 1
        assert "CANDLESTICK PATTERN SIGNAL" in sent[0]

    def test_options_flow_alert_sent(self):
        from momentum_radar.services.scheduler import _run_categorized_signal_scan
        fetcher = _make_fetcher_returning_none()
        sent = []

        confs = [_make_confirmation("Golden Sweep (Calls)", "options", "5000 contracts")]
        sig = _make_signal_result("NVDA", "ALERT", confs)

        with patch("momentum_radar.services.signal_engine.evaluate", return_value=sig):
            _run_categorized_signal_scan(["NVDA"], fetcher, sent.append)

        assert len(sent) == 1
        assert "OPTIONS FLOW SIGNAL" in sent[0]

    def test_multiple_categories_same_ticker_separate_alerts(self):
        """If a ticker has both chart pattern and candlestick confirmations,
        two separate alerts should be sent (one per category)."""
        from momentum_radar.services.scheduler import _run_categorized_signal_scan
        fetcher = _make_fetcher_returning_none()
        sent = []

        confs = [
            _make_confirmation("Double Bottom", "pattern", "support near $100"),
            _make_confirmation("Bullish Engulfing", "candlestick", "engulfing candle"),
        ]
        sig = _make_signal_result("MSFT", "HIGH_CONFIDENCE", confs)

        with patch("momentum_radar.services.signal_engine.evaluate", return_value=sig):
            _run_categorized_signal_scan(["MSFT"], fetcher, sent.append)

        # Two separate alerts (chart_pattern + candlestick)
        assert len(sent) == 2
        types_sent = {
            "chart" if "CHART PATTERN" in m else
            "candle" if "CANDLESTICK" in m else
            "other"
            for m in sent
        }
        assert "chart" in types_sent
        assert "candle" in types_sent

    def test_cooldown_prevents_same_category_resend(self):
        """After a chart_pattern alert is sent, the same ticker/type should be suppressed."""
        from momentum_radar.services.scheduler import _run_categorized_signal_scan
        from momentum_radar.storage.database import record_signal_alert
        fetcher = _make_fetcher_returning_none()
        sent = []

        # Pre-record a chart_pattern alert for AAPL
        record_signal_alert("AAPL", "chart_pattern")

        confs = [_make_confirmation("Double Bottom", "pattern", "support near $100")]
        sig = _make_signal_result("AAPL", "ALERT", confs)

        with patch("momentum_radar.services.signal_engine.evaluate", return_value=sig):
            _run_categorized_signal_scan(["AAPL"], fetcher, sent.append)

        assert sent == []

    def test_already_alerted_tickers_skipped(self):
        """Tickers in the already_alerted set should be skipped entirely."""
        from momentum_radar.services.scheduler import _run_categorized_signal_scan
        fetcher = _make_fetcher_returning_none()
        sent = []

        confs = [_make_confirmation("Double Bottom", "pattern", "support near $100")]
        sig = _make_signal_result("SKIP", "ALERT", confs)

        with patch("momentum_radar.services.signal_engine.evaluate", return_value=sig):
            _run_categorized_signal_scan(["SKIP"], fetcher, sent.append, already_alerted={"SKIP"})

        assert sent == []

    def test_per_type_cap_enforced(self):
        """No more than _MAX_CHART_PATTERN_ALERTS chart-pattern alerts per run."""
        from momentum_radar.services.scheduler import (
            _run_categorized_signal_scan,
            _MAX_CHART_PATTERN_ALERTS,
        )
        fetcher = _make_fetcher_returning_none()
        sent = []

        tickers = [f"T{i}" for i in range(20)]

        def make_sig(ticker):
            return _make_signal_result(
                ticker, "ALERT",
                [_make_confirmation("DB", "pattern", "detail")],
            )

        with patch(
            "momentum_radar.services.signal_engine.evaluate",
            side_effect=lambda ticker, **kw: make_sig(ticker),
        ):
            _run_categorized_signal_scan(tickers, fetcher, sent.append)

        chart_alerts = [m for m in sent if "CHART PATTERN" in m]
        assert len(chart_alerts) <= _MAX_CHART_PATTERN_ALERTS

    def test_evaluate_failure_does_not_crash(self):
        """If evaluate() raises, the scan should continue without crashing."""
        from momentum_radar.services.scheduler import _run_categorized_signal_scan
        fetcher = _make_fetcher_returning_none()
        sent = []

        with patch(
            "momentum_radar.services.signal_engine.evaluate",
            side_effect=RuntimeError("network error"),
        ):
            _run_categorized_signal_scan(["ERR"], fetcher, sent.append)

        assert sent == []
