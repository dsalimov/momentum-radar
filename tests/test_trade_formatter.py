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
    assert "🚨 DAY TRADE" in msg
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


# ---------------------------------------------------------------------------
# format_strategy_signal
# ---------------------------------------------------------------------------

def _make_signal(**kwargs):
    """Build a minimal StrategySignal for testing."""
    from momentum_radar.strategies.base import StrategySignal
    defaults = dict(
        ticker="AMD",
        strategy="scalp",
        entry=178.40,
        stop=177.90,
        target=179.10,
        rr=2.0,
        grade="A",
    )
    defaults.update(kwargs)
    return StrategySignal(**defaults)


class TestFormatStrategySignal:
    def test_scalp_trade_header(self):
        """Scalp strategy is now classified as DAY TRADE (scalp removed)."""
        from momentum_radar.alerts.trade_formatter import format_strategy_signal
        signal = _make_signal(strategy="scalp")
        msg = format_strategy_signal(signal)
        assert "🚨 DAY TRADE" in msg

    def test_day_trade_header_intraday(self):
        """Intraday strategy uses 🚨 DAY TRADE header."""
        from momentum_radar.alerts.trade_formatter import format_strategy_signal
        signal = _make_signal(strategy="intraday")
        msg = format_strategy_signal(signal)
        assert "🚨 DAY TRADE" in msg

    def test_swing_trade_header(self):
        """Swing strategy uses 🚨 SWING TRADE header."""
        from momentum_radar.alerts.trade_formatter import format_strategy_signal
        signal = _make_signal(strategy="swing")
        msg = format_strategy_signal(signal)
        assert "🚨 SWING TRADE" in msg

    def test_chart_pattern_header(self):
        """Chart pattern strategy uses 🚨 SWING TRADE header."""
        from momentum_radar.alerts.trade_formatter import format_strategy_signal
        signal = _make_signal(strategy="chart_pattern")
        msg = format_strategy_signal(signal)
        assert "🚨 SWING TRADE" in msg

    def test_unusual_volume_header(self):
        """Unusual volume strategy uses 🚨 DAY TRADE header."""
        from momentum_radar.alerts.trade_formatter import format_strategy_signal
        signal = _make_signal(strategy="unusual_volume")
        msg = format_strategy_signal(signal)
        assert "🚨 DAY TRADE" in msg

    def test_required_fields_present(self):
        """All required fields are included in the alert."""
        from momentum_radar.alerts.trade_formatter import format_strategy_signal
        signal = _make_signal()
        msg = format_strategy_signal(signal)
        assert "AMD" in msg
        assert "178.40" in msg
        assert "177.90" in msg
        assert "179.10" in msg
        assert "🚨" in msg

    def test_empty_without_trade_structure(self):
        """Returns empty string when entry/stop/target are all zero."""
        from momentum_radar.alerts.trade_formatter import format_strategy_signal
        from momentum_radar.strategies.base import StrategySignal
        signal = StrategySignal(ticker="X", strategy="scalp")
        assert format_strategy_signal(signal) == ""

    def test_single_target_when_target2_zero(self):
        """Single target line when target2 is not set."""
        from momentum_radar.alerts.trade_formatter import format_strategy_signal
        signal = _make_signal(target2=0.0)
        msg = format_strategy_signal(signal)
        assert "Target:" in msg
        assert "Target 1:" not in msg
        assert "Target 2:" not in msg

    def test_two_targets_when_target2_set(self):
        """Target 1 and Target 2 lines shown when target2 > 0."""
        from momentum_radar.alerts.trade_formatter import format_strategy_signal
        signal = _make_signal(strategy="swing", target=930.0, target2=950.0)
        msg = format_strategy_signal(signal)
        assert "Target 1:" in msg
        assert "Target 2:" in msg
        assert "930.00" in msg
        assert "950.00" in msg

    def test_options_flow_label_shown(self):
        """Options flow label is included when set."""
        from momentum_radar.alerts.trade_formatter import format_strategy_signal
        signal = _make_signal(options_flow_label="Weekly Call Sweep")
        msg = format_strategy_signal(signal)
        assert "Options Flow: Weekly Call Sweep" in msg

    def test_options_flow_label_omitted_when_empty(self):
        """No options flow line when label is empty."""
        from momentum_radar.alerts.trade_formatter import format_strategy_signal
        signal = _make_signal(options_flow_label="")
        msg = format_strategy_signal(signal)
        assert "Options Flow:" not in msg

    def test_confirmations_shown(self):
        """Confirmations list is included when not empty."""
        from momentum_radar.alerts.trade_formatter import format_strategy_signal
        signal = _make_signal(confirmations=["Break of Structure", "Volume Spike (3.2x)"])
        msg = format_strategy_signal(signal)
        assert "Confirmations:" in msg
        assert "Break of Structure" in msg

    def test_grade_shown_as_confidence(self):
        """Grade is used as the confidence label."""
        from momentum_radar.alerts.trade_formatter import format_strategy_signal
        signal = _make_signal(grade="A+")
        msg = format_strategy_signal(signal)
        assert "A+" in msg

    def test_rr_shown(self):
        """Risk/reward ratio is included in the alert."""
        from momentum_radar.alerts.trade_formatter import format_strategy_signal
        signal = _make_signal(rr=2.5)
        msg = format_strategy_signal(signal)
        assert "2.5" in msg

    def test_custom_setup_name(self):
        """Custom setup_name overrides the default."""
        from momentum_radar.alerts.trade_formatter import format_strategy_signal
        signal = _make_signal()
        msg = format_strategy_signal(signal, setup_name="VWAP Reclaim")
        assert "VWAP Reclaim" in msg

    def test_direction_long_displayed(self):
        """BUY direction renders as Long."""
        from momentum_radar.alerts.trade_formatter import format_strategy_signal
        signal = _make_signal(direction="BUY")
        msg = format_strategy_signal(signal)
        assert "Long" in msg

    def test_direction_short_displayed(self):
        """SELL direction renders as Short."""
        from momentum_radar.alerts.trade_formatter import format_strategy_signal
        signal = _make_signal(direction="SELL")
        msg = format_strategy_signal(signal)
        assert "Short" in msg


# ---------------------------------------------------------------------------
# TradeSetup: target2 multi-target support
# ---------------------------------------------------------------------------

class TestTradeSetupTarget2:
    def test_single_target_when_target2_zero(self):
        """format_trade_setup shows single target line when target2 is 0."""
        setup = _make_setup(target2=0.0)
        msg = format_trade_setup(setup, timestamp=_FIXED_TS)
        assert "Target:" in msg
        assert "Target 1:" not in msg

    def test_two_targets_when_target2_set(self):
        """format_trade_setup shows Target 1 and Target 2 when target2 > 0."""
        setup = _make_setup(
            setup_type=SetupType.CHART_PATTERN_BREAKOUT,
            direction=SetupDirection.LONG,
            target=320.0,
            target2=340.0,
        )
        msg = format_trade_setup(setup, timestamp=_FIXED_TS)
        assert "Target 1:" in msg
        assert "Target 2:" in msg
        assert "320.00" in msg
        assert "340.00" in msg
