"""
test_economic_calendar.py – Unit tests for the economic calendar module.
"""

import pytest
from datetime import date, timedelta
from unittest.mock import patch


# ---------------------------------------------------------------------------
# _get_week_dates
# ---------------------------------------------------------------------------

class TestGetWeekDates:
    def test_returns_five_dates(self):
        from momentum_radar.utils.economic_calendar import _get_week_dates
        dates = _get_week_dates(0)
        assert len(dates) == 5

    def test_current_week_starts_on_monday(self):
        from momentum_radar.utils.economic_calendar import _get_week_dates
        dates = _get_week_dates(0)
        assert dates[0].weekday() == 0  # Monday

    def test_dates_are_consecutive(self):
        from momentum_radar.utils.economic_calendar import _get_week_dates
        dates = _get_week_dates(0)
        for i in range(1, len(dates)):
            assert dates[i] == dates[i - 1] + timedelta(days=1)

    def test_next_week_offset_is_seven_days_ahead(self):
        from momentum_radar.utils.economic_calendar import _get_week_dates
        current = _get_week_dates(0)
        next_week = _get_week_dates(1)
        assert next_week[0] == current[0] + timedelta(days=7)

    def test_last_week_offset_is_seven_days_behind(self):
        from momentum_radar.utils.economic_calendar import _get_week_dates
        current = _get_week_dates(0)
        last_week = _get_week_dates(-1)
        assert last_week[0] == current[0] - timedelta(days=7)


# ---------------------------------------------------------------------------
# get_weekly_calendar
# ---------------------------------------------------------------------------

class TestGetWeeklyCalendar:
    def test_returns_list(self):
        from momentum_radar.utils.economic_calendar import get_weekly_calendar
        with patch("momentum_radar.utils.economic_calendar._fetch_earnings_this_week", return_value=[]):
            events = get_weekly_calendar(0)
        assert isinstance(events, list)

    def test_weekly_recurring_events_always_present(self):
        """Weekly events (jobless claims, oil inventory) should always appear."""
        from momentum_radar.utils.economic_calendar import get_weekly_calendar
        with patch("momentum_radar.utils.economic_calendar._fetch_earnings_this_week", return_value=[]):
            events = get_weekly_calendar(0)
        names = [e["name"] for e in events]
        assert "Initial Jobless Claims" in names
        assert "EIA Crude Oil Inventories" in names

    def test_events_have_required_keys(self):
        from momentum_radar.utils.economic_calendar import get_weekly_calendar
        with patch("momentum_radar.utils.economic_calendar._fetch_earnings_this_week", return_value=[]):
            events = get_weekly_calendar(0)
        for event in events:
            assert "name" in event
            assert "date" in event
            assert "day" in event
            assert "time" in event
            assert "impact" in event
            assert "description" in event
            assert "category" in event

    def test_events_sorted_by_date(self):
        from momentum_radar.utils.economic_calendar import get_weekly_calendar
        with patch("momentum_radar.utils.economic_calendar._fetch_earnings_this_week", return_value=[]):
            events = get_weekly_calendar(0)
        dates = [e["date"] for e in events]
        assert dates == sorted(dates)

    def test_all_dates_within_week(self):
        from momentum_radar.utils.economic_calendar import get_weekly_calendar, _get_week_dates
        week_dates = _get_week_dates(0)
        week_start = week_dates[0]
        week_end = week_dates[-1]
        with patch("momentum_radar.utils.economic_calendar._fetch_earnings_this_week", return_value=[]):
            events = get_weekly_calendar(0)
        for event in events:
            ev_date = event.get("date")
            if isinstance(ev_date, date):
                assert week_start <= ev_date <= week_end

    def test_earnings_events_included(self):
        from momentum_radar.utils.economic_calendar import get_weekly_calendar, _get_week_dates
        week_dates = _get_week_dates(0)
        fake_earnings = [{
            "name": "Earnings: AAPL",
            "date": week_dates[1],
            "time": "Pre/Post Market",
            "impact": "HIGH",
            "description": "AAPL quarterly earnings.",
            "category": "Earnings",
        }]
        with patch("momentum_radar.utils.economic_calendar._fetch_earnings_this_week", return_value=fake_earnings):
            events = get_weekly_calendar(0)
        names = [e["name"] for e in events]
        assert "Earnings: AAPL" in names


# ---------------------------------------------------------------------------
# format_calendar_report
# ---------------------------------------------------------------------------

class TestFormatCalendarReport:
    def _get_events(self, week_offset=0):
        from momentum_radar.utils.economic_calendar import get_weekly_calendar
        with patch("momentum_radar.utils.economic_calendar._fetch_earnings_this_week", return_value=[]):
            return get_weekly_calendar(week_offset)

    def test_returns_string(self):
        from momentum_radar.utils.economic_calendar import format_calendar_report
        events = self._get_events()
        report = format_calendar_report(events)
        assert isinstance(report, str)

    def test_contains_calendar_header(self):
        from momentum_radar.utils.economic_calendar import format_calendar_report
        events = self._get_events()
        report = format_calendar_report(events)
        assert "MARKET ECONOMIC CALENDAR" in report

    def test_contains_this_week_label(self):
        from momentum_radar.utils.economic_calendar import format_calendar_report
        events = self._get_events()
        report = format_calendar_report(events, week_offset=0)
        assert "This Week" in report

    def test_contains_next_week_label_for_offset_1(self):
        from momentum_radar.utils.economic_calendar import format_calendar_report
        events = self._get_events(week_offset=1)
        report = format_calendar_report(events, week_offset=1)
        assert "Next Week" in report

    def test_contains_impact_legend(self):
        from momentum_radar.utils.economic_calendar import format_calendar_report
        events = self._get_events()
        report = format_calendar_report(events)
        assert "Impact:" in report

    def test_contains_day_names(self):
        from momentum_radar.utils.economic_calendar import format_calendar_report
        events = self._get_events()
        report = format_calendar_report(events)
        # At least one weekday should appear in uppercase
        day_names = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY"]
        assert any(day in report for day in day_names)

    def test_empty_events_handled_gracefully(self):
        from momentum_radar.utils.economic_calendar import format_calendar_report
        report = format_calendar_report([])
        assert isinstance(report, str)
