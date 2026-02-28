"""
test_market_hours.py – Unit tests for market hours and lunch-lull logic.
"""

from datetime import datetime

import pytz
import pytest

from momentum_radar.utils.market_hours import (
    is_market_open,
    is_lunch_lull,
    get_market_score_penalty,
)

EST = pytz.timezone("America/New_York")


def _est(hour: int, minute: int) -> datetime:
    return EST.localize(datetime(2024, 1, 15, hour, minute))


class TestIsMarketOpen:
    def test_before_open_returns_false(self):
        assert is_market_open(_est(9, 0)) is False

    def test_at_open_returns_true(self):
        assert is_market_open(_est(9, 35)) is True

    def test_mid_session_returns_true(self):
        assert is_market_open(_est(13, 0)) is True

    def test_at_close_returns_true(self):
        assert is_market_open(_est(15, 45)) is True

    def test_after_close_returns_false(self):
        assert is_market_open(_est(16, 0)) is False

    def test_early_morning_returns_false(self):
        assert is_market_open(_est(4, 30)) is False


class TestIsLunchLull:
    def test_before_lunch_returns_false(self):
        assert is_lunch_lull(_est(11, 59)) is False

    def test_at_lunch_start_returns_true(self):
        assert is_lunch_lull(_est(12, 0)) is True

    def test_mid_lunch_returns_true(self):
        assert is_lunch_lull(_est(12, 45)) is True

    def test_at_lunch_end_returns_true(self):
        assert is_lunch_lull(_est(13, 30)) is True

    def test_after_lunch_returns_false(self):
        assert is_lunch_lull(_est(13, 31)) is False


class TestGetMarketScorePenalty:
    def test_flat_market_returns_penalty_1(self):
        penalty, condition = get_market_score_penalty(0.001, 0.002)
        assert penalty == 1
        assert condition == "flat_market"

    def test_moving_market_no_penalty(self):
        penalty, _ = get_market_score_penalty(0.01, 0.012)
        assert penalty == 0

    def test_none_inputs_no_penalty(self):
        penalty, condition = get_market_score_penalty(None, None)
        assert penalty == 0
        assert condition == "unknown"

    def test_bullish_market(self):
        penalty, condition = get_market_score_penalty(0.005, 0.007)
        assert penalty == 0
        assert condition == "bullish"

    def test_bearish_market(self):
        penalty, condition = get_market_score_penalty(-0.005, -0.007)
        assert penalty == 0
        assert condition == "bearish"
