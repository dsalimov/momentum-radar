"""
test_formatter.py – Unit tests for alert formatting.
"""

from datetime import datetime

import pytest

from momentum_radar.alerts.formatter import format_alert
from momentum_radar.signals.scoring import AlertLevel


_FIXED_TS = datetime(2024, 1, 15, 10, 42, 0)


def test_format_alert_high_priority():
    """HIGH_PRIORITY alert includes expected header and key fields."""
    msg = format_alert(
        ticker="XYZ",
        price=42.15,
        pct_change=6.4,
        rvol=2.8,
        score=7,
        alert_level=AlertLevel.HIGH_PRIORITY,
        triggered_modules=["volume_spike", "structure_break", "short_interest"],
        module_details={
            "volume_spike": "5m vol 3.2x avg",
            "structure_break": "Break of prev-day high (strong)",
            "short_interest": "Short 20.0%, DTC 4.5, Float 50M",
        },
        short_interest=0.20,
        float_shares=50_000_000,
        atr_ratio=1.9,
        timestamp=_FIXED_TS,
    )
    assert "HIGH PRIORITY SIGNAL" in msg
    assert "Ticker: XYZ" in msg
    assert "Price: 42.15" in msg
    assert "+6.4%" in msg
    assert "RVOL: 2.8" in msg
    assert "Score: 7" in msg
    assert "Volume Spike" in msg
    assert "Structure Break" in msg
    assert "Short Interest" in msg
    assert "Range vs ATR: 1.9x" in msg
    assert "Float: 50M" in msg


def test_format_alert_strong_momentum():
    """STRONG_MOMENTUM alert has correct emoji and label."""
    msg = format_alert(
        ticker="ABC",
        price=10.0,
        pct_change=-2.5,
        rvol=3.5,
        score=9,
        alert_level=AlertLevel.STRONG_MOMENTUM,
        triggered_modules=[],
        module_details={},
        timestamp=_FIXED_TS,
    )
    assert "STRONG MOMENTUM SIGNAL" in msg
    assert "🔥" in msg
    assert "-2.5%" in msg


def test_format_alert_watchlist():
    """WATCHLIST alert uses the watchlist label."""
    msg = format_alert(
        ticker="DEF",
        price=25.00,
        pct_change=0.0,
        rvol=1.0,
        score=4,
        alert_level=AlertLevel.WATCHLIST,
        triggered_modules=["relative_volume"],
        module_details={"relative_volume": "RVOL 2.1x"},
        timestamp=_FIXED_TS,
    )
    assert "WATCHLIST" in msg
    assert "Ticker: DEF" in msg


def test_format_alert_no_optional_fields():
    """format_alert works without optional fields (short_interest, etc.)."""
    msg = format_alert(
        ticker="GHI",
        price=55.0,
        pct_change=1.2,
        rvol=1.5,
        score=5,
        alert_level=AlertLevel.HIGH_PRIORITY,
        triggered_modules=["volume_spike"],
        module_details={},
        timestamp=_FIXED_TS,
    )
    assert "Ticker: GHI" in msg
    # Optional fields should be absent
    assert "Short Interest" not in msg
    assert "Float" not in msg
    assert "Range vs ATR" not in msg


def test_format_alert_timestamp_format():
    """Timestamp is formatted as 12-hour clock with AM/PM."""
    msg = format_alert(
        ticker="TST",
        price=100.0,
        pct_change=0.5,
        rvol=1.0,
        score=5,
        alert_level=AlertLevel.HIGH_PRIORITY,
        triggered_modules=[],
        module_details={},
        timestamp=datetime(2024, 3, 1, 14, 30),
    )
    assert "02:30 PM EST" in msg
