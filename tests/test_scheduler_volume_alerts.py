"""
tests/test_scheduler_volume_alerts.py – Tests for automated volume-spike alerts
added to the hourly scheduler.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, call, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spike(ticker: str, rvol: float, pct_change: float = 1.5,
                today_volume: int = 3_000_000) -> dict:
    """Build a minimal volume-spike dict as returned by scan_volume_spikes."""
    return {
        "ticker": ticker,
        "rvol": rvol,
        "today_volume": today_volume,
        "avg_30d_volume": int(today_volume / rvol),
        "last_close": 50.0,
        "pct_change": pct_change,
    }


# ---------------------------------------------------------------------------
# _format_volume_spike_alert
# ---------------------------------------------------------------------------

class TestFormatVolumeSpike:
    def _fmt(self, **kwargs):
        from momentum_radar.services.scheduler import _format_volume_spike_alert
        spike = _make_spike(**kwargs)
        return _format_volume_spike_alert(spike)

    def test_contains_ticker(self):
        text = self._fmt(ticker="AAPL", rvol=3.5)
        assert "AAPL" in text

    def test_contains_rvol(self):
        text = self._fmt(ticker="GME", rvol=4.2)
        assert "4.2x" in text

    def test_volume_in_millions(self):
        text = self._fmt(ticker="T", rvol=2.5, today_volume=5_000_000)
        assert "5.0M" in text

    def test_volume_in_thousands(self):
        text = self._fmt(ticker="T", rvol=2.5, today_volume=500_000)
        assert "500K" in text

    def test_positive_pct_has_plus(self):
        text = self._fmt(ticker="T", rvol=2.0, pct_change=3.7)
        assert "+3.7%" in text

    def test_negative_pct_no_plus(self):
        text = self._fmt(ticker="T", rvol=2.0, pct_change=-2.1)
        assert "-2.1%" in text
        assert "+-" not in text

    def test_disclaimer_included(self):
        text = self._fmt(ticker="T", rvol=2.0)
        assert "risk management" in text.lower()

    def test_emoji_in_header(self):
        text = self._fmt(ticker="T", rvol=2.0)
        assert "📈" in text


# ---------------------------------------------------------------------------
# _run_hourly_scan – volume spike integration
# ---------------------------------------------------------------------------

class TestRunHourlyScanVolumeAlerts:
    """Unit tests for the volume-spike section of _run_hourly_scan."""

    def _make_fetcher(self):
        fetcher = MagicMock()
        fetcher.get_daily_bars.return_value = None
        fetcher.get_intraday_bars.return_value = None
        fetcher.get_fundamentals.return_value = None
        fetcher.get_options_volume.return_value = None
        fetcher.get_quote.return_value = None
        return fetcher

    def test_volume_spike_alert_is_sent(self):
        """When scan_volume_spikes returns a high-RVOL spike, send_fn is called."""
        from momentum_radar.services.scheduler import _run_hourly_scan
        from momentum_radar.storage.database import init_db

        init_db("sqlite:///:memory:")
        fetcher = self._make_fetcher()
        sent_messages = []

        spike = _make_spike("VOLTEST", rvol=3.5)

        with patch(
            "momentum_radar.services.squeeze_engine.scan_universe",
            return_value=[],
        ), patch(
            "momentum_radar.data.volume_scanner.scan_volume_spikes",
            return_value=[spike],
        ):
            _run_hourly_scan(["VOLTEST"], fetcher, sent_messages.append)

        assert len(sent_messages) == 1
        assert "VOLTEST" in sent_messages[0]
        assert "UNUSUAL VOLUME" in sent_messages[0]

    def test_no_double_alert_for_squeeze_ticker(self):
        """A ticker already alerted via squeeze path should NOT also get a volume alert."""
        from momentum_radar.services.scheduler import _run_hourly_scan
        from momentum_radar.storage.database import init_db

        init_db("sqlite:///:memory:")
        fetcher = self._make_fetcher()
        sent_messages = []

        squeeze_report = {
            "ticker": "BOTH",
            "squeeze_score": 80,
            "short_interest_pct": 0.3,
            "days_to_cover": 5.0,
            "float_shares": 10_000_000,
            "float_str": "10M",
            "rvol": 4.0,
            "cp_ratio": 2.5,
            "borrow_fee_estimate": 0.12,
            "breakout_level": 20.0,
            "resistance": 22.0,
            "bull_target1": 23.0,
            "bull_target2": 25.0,
            "bear_target": 18.0,
            "atr": 1.0,
            "current_price": 20.5,
        }
        vol_spike = _make_spike("BOTH", rvol=4.5)

        with patch(
            "momentum_radar.services.squeeze_engine.scan_universe",
            return_value=[squeeze_report],
        ), patch(
            "momentum_radar.services.signal_engine.evaluate",
            return_value=MagicMock(confirmation_count=0, confirmation_labels=[]),
        ), patch(
            "momentum_radar.data.volume_scanner.scan_volume_spikes",
            return_value=[vol_spike],
        ):
            _run_hourly_scan(["BOTH"], fetcher, sent_messages.append)

        # Only one alert for BOTH (squeeze alert supersedes volume alert)
        tickers_alerted = [
            line for msg in sent_messages for line in msg.splitlines()
            if "BOTH" in line
        ]
        assert len(tickers_alerted) >= 1
        # No "UNUSUAL VOLUME ALERT" for the same ticker
        assert not any("UNUSUAL VOLUME ALERT: BOTH" in m for m in sent_messages)

    def test_volume_spike_cap_enforced(self):
        """At most _MAX_VOLUME_ALERTS_PER_HOUR volume alerts are sent."""
        from momentum_radar.services.scheduler import _run_hourly_scan, _MAX_VOLUME_ALERTS_PER_HOUR
        from momentum_radar.storage.database import init_db

        init_db("sqlite:///:memory:")
        fetcher = self._make_fetcher()
        sent_messages = []

        # Create more spikes than the cap
        spikes = [_make_spike(f"VOL{i}", rvol=2.5 + i * 0.1) for i in range(10)]

        with patch(
            "momentum_radar.services.squeeze_engine.scan_universe",
            return_value=[],
        ), patch(
            "momentum_radar.data.volume_scanner.scan_volume_spikes",
            return_value=spikes,
        ):
            _run_hourly_scan([s["ticker"] for s in spikes], fetcher, sent_messages.append)

        volume_alerts = [m for m in sent_messages if "UNUSUAL VOLUME" in m]
        assert len(volume_alerts) <= _MAX_VOLUME_ALERTS_PER_HOUR

    def test_volume_scanner_failure_does_not_crash(self):
        """If scan_volume_spikes raises, the scan should still complete cleanly."""
        from momentum_radar.services.scheduler import _run_hourly_scan
        from momentum_radar.storage.database import init_db

        init_db("sqlite:///:memory:")
        fetcher = self._make_fetcher()
        sent_messages = []

        with patch(
            "momentum_radar.services.squeeze_engine.scan_universe",
            return_value=[],
        ), patch(
            "momentum_radar.data.volume_scanner.scan_volume_spikes",
            side_effect=RuntimeError("network error"),
        ):
            # Should not raise
            _run_hourly_scan(["AAA"], fetcher, sent_messages.append)

        assert sent_messages == []
