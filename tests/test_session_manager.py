"""
tests/test_session_manager.py – Tests for the session_manager module.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

_ET = ZoneInfo("America/New_York")


def _et(hour, minute=0, weekday=0):
    """Build a timezone-aware datetime in US/Eastern on a weekday.

    weekday: 0=Monday … 6=Sunday
    """
    # Use 2024-01-01 (Monday) and advance by weekday days
    base = datetime(2024, 1, 1, hour, minute, tzinfo=_ET)
    from datetime import timedelta
    return base + timedelta(days=weekday)


class TestGetCurrentSession:
    def test_open_session(self):
        from momentum_radar.services.session_manager import get_current_session

        dt = _et(9, 45)  # 09:45 ET Monday
        assert get_current_session(dt) == "open"

    def test_morning_session(self):
        from momentum_radar.services.session_manager import get_current_session

        dt = _et(10, 30)
        assert get_current_session(dt) == "morning"

    def test_midday_session(self):
        from momentum_radar.services.session_manager import get_current_session

        dt = _et(11, 30)
        assert get_current_session(dt) == "midday"

    def test_afternoon_session(self):
        from momentum_radar.services.session_manager import get_current_session

        dt = _et(14, 0)
        assert get_current_session(dt) == "afternoon"

    def test_afterhours_session(self):
        from momentum_radar.services.session_manager import get_current_session

        dt = _et(17, 0)
        assert get_current_session(dt) == "afterhours"

    def test_premarket_session(self):
        from momentum_radar.services.session_manager import get_current_session

        dt = _et(7, 0)
        assert get_current_session(dt) == "premarket"

    def test_weekend_returns_closed(self):
        from momentum_radar.services.session_manager import get_current_session

        dt = _et(10, 0, weekday=5)  # Saturday
        assert get_current_session(dt) == "closed"

    def test_overnight_returns_closed(self):
        from momentum_radar.services.session_manager import get_current_session

        dt = _et(2, 0)  # 02:00 ET
        assert get_current_session(dt) == "closed"


class TestGetSessionTimeframe:
    def test_open_returns_2m(self):
        from momentum_radar.services.session_manager import get_session_timeframe

        assert get_session_timeframe("open") == "2m"

    def test_morning_returns_5m(self):
        from momentum_radar.services.session_manager import get_session_timeframe

        assert get_session_timeframe("morning") == "5m"

    def test_midday_returns_10m(self):
        from momentum_radar.services.session_manager import get_session_timeframe

        assert get_session_timeframe("midday") == "10m"

    def test_afternoon_returns_10m(self):
        from momentum_radar.services.session_manager import get_session_timeframe

        assert get_session_timeframe("afternoon") == "10m"

    def test_closed_returns_daily(self):
        from momentum_radar.services.session_manager import get_session_timeframe

        assert get_session_timeframe("closed") == "daily"


class TestIsMarketOpen:
    def test_during_session_returns_true(self):
        from momentum_radar.services.session_manager import is_market_open

        dt = _et(10, 0)
        assert is_market_open(dt) is True

    def test_before_open_returns_false(self):
        from momentum_radar.services.session_manager import is_market_open

        dt = _et(9, 0)
        assert is_market_open(dt) is False

    def test_after_close_returns_false(self):
        from momentum_radar.services.session_manager import is_market_open

        dt = _et(16, 30)
        assert is_market_open(dt) is False

    def test_weekend_returns_false(self):
        from momentum_radar.services.session_manager import is_market_open

        dt = _et(10, 0, weekday=6)  # Sunday
        assert is_market_open(dt) is False


class TestShouldSendSignals:
    def test_open_session_active(self):
        from momentum_radar.services.session_manager import should_send_signals

        assert should_send_signals("open") is True

    def test_morning_session_active(self):
        from momentum_radar.services.session_manager import should_send_signals

        assert should_send_signals("morning") is True

    def test_premarket_not_active(self):
        from momentum_radar.services.session_manager import should_send_signals

        assert should_send_signals("premarket") is False

    def test_afterhours_not_active(self):
        from momentum_radar.services.session_manager import should_send_signals

        assert should_send_signals("afterhours") is False


class TestGetSessionInfo:
    def test_returns_all_keys(self):
        from momentum_radar.services.session_manager import get_session_info

        info = get_session_info(_et(10, 30))
        for key in ("session", "timeframe", "is_market_open", "signals_active"):
            assert key in info

    def test_morning_info_consistent(self):
        from momentum_radar.services.session_manager import get_session_info

        info = get_session_info(_et(10, 30))
        assert info["session"] == "morning"
        assert info["timeframe"] == "5m"
        assert info["is_market_open"] is True
        assert info["signals_active"] is True
