"""
test_trade_formatter.py – Unit tests for the trade setup alert formatter.
"""

from datetime import datetime

import pytest

from momentum_radar.signals.setup_detector import (
    TradeSetup,
    SetupType,
    SetupDirection,
)
from momentum_radar.alerts.trade_formatter import format_trade_setup, format_trade_setup_list


_FIXED_TS = datetime(2024, 1, 15, 10, 31, 0)


def _make_setup(**kwargs) -> TradeSetup:
    """Build a minimal TradeSetup for testing."""
    defaults = dict(
        ticker="MAR",
        setup_type=SetupType.VWAP_BREAKDOWN,
        direction=SetupDirection.SHORT,
        entry=325.80,
        stop=327.20,
        target=321.50,
        rvol=1.9,
        volume_spike=4.2,
        confidence="High",
        timestamp=_FIXED_TS,
        details="Test details",
    )
    defaults.update(kwargs)
    return TradeSetup(**defaults)


# ---------------------------------------------------------------------------
# format_trade_setup
# ---------------------------------------------------------------------------

def test_format_trade_setup_contains_required_fields():
    """Alert must contain all required fields."""
    msg = format_trade_setup(_make_setup(), timestamp=_FIXED_TS)
    assert "🚨 TRADE SETUP" in msg
    assert "MAR" in msg
    assert "VWAP Breakdown" in msg
    assert "Short" in msg
    assert "325.80" in msg
    assert "327.20" in msg
    assert "321.50" in msg
    assert "4.2x" in msg
    assert "1.9" in msg
    assert "High" in msg


def test_format_trade_setup_risk_reward_shown():
    """Risk/reward ratio is included in the message."""
    setup = _make_setup(entry=100.0, stop=98.0, target=104.0)
    msg = format_trade_setup(setup, timestamp=_FIXED_TS)
    assert "1:2.0" in msg or "2.0" in msg


def test_format_trade_setup_timestamp():
    """Formatted timestamp appears in the output."""
    msg = format_trade_setup(_make_setup(), timestamp=_FIXED_TS)
    # 10:31 AM in 12-hour format
    assert "10:31" in msg


def test_format_trade_setup_long_direction():
    """Long direction appears correctly."""
    setup = _make_setup(
        setup_type=SetupType.VWAP_RECLAIM,
        direction=SetupDirection.LONG,
    )
    msg = format_trade_setup(setup, timestamp=_FIXED_TS)
    assert "Long" in msg
    assert "VWAP Reclaim" in msg


def test_format_trade_setup_medium_confidence():
    """Medium confidence level is reflected in the output."""
    setup = _make_setup(confidence="Medium")
    msg = format_trade_setup(setup, timestamp=_FIXED_TS)
    assert "Medium" in msg


def test_format_trade_setup_details_shown():
    """Details string is included when not empty."""
    setup = _make_setup(details="Prev close 99.00 < VWAP 100.00; current 101.00")
    msg = format_trade_setup(setup, timestamp=_FIXED_TS)
    assert "Details:" in msg
    assert "Prev close" in msg


def test_format_trade_setup_no_details():
    """No details section when details is empty."""
    setup = _make_setup(details="")
    msg = format_trade_setup(setup, timestamp=_FIXED_TS)
    assert "Details:" not in msg


def test_format_trade_setup_all_setup_types():
    """All SetupType values produce a non-empty message."""
    for st in SetupType:
        setup = _make_setup(setup_type=st)
        msg = format_trade_setup(setup, timestamp=_FIXED_TS)
        assert msg.strip() != ""
        assert st.value in msg


# ---------------------------------------------------------------------------
# format_trade_setup_list
# ---------------------------------------------------------------------------

def test_format_trade_setup_list_empty():
    """Empty list returns empty string."""
    assert format_trade_setup_list([]) == ""


def test_format_trade_setup_list_single():
    """Single setup returns a non-empty message without separator."""
    setup = _make_setup()
    msg = format_trade_setup_list([setup], timestamp=_FIXED_TS)
    assert "MAR" in msg
    assert "─" not in msg  # no separator for single item


def test_format_trade_setup_list_multiple():
    """Multiple setups are separated by a divider."""
    s1 = _make_setup(ticker="AAA", setup_type=SetupType.VWAP_RECLAIM, direction=SetupDirection.LONG)
    s2 = _make_setup(ticker="BBB", setup_type=SetupType.SUPPORT_BOUNCE, direction=SetupDirection.LONG)
    msg = format_trade_setup_list([s1, s2], timestamp=_FIXED_TS)
    assert "AAA" in msg
    assert "BBB" in msg
    assert "─" in msg  # separator present
