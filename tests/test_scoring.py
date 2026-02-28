"""
test_scoring.py – Unit tests for signal scoring and alert level assignment.
"""

import pytest
from momentum_radar.signals.scoring import score_to_alert_level, AlertLevel


@pytest.mark.parametrize(
    "score,expected_level",
    [
        (0, AlertLevel.IGNORE),
        (2, AlertLevel.IGNORE),
        (3, AlertLevel.IGNORE),
        (4, AlertLevel.WATCHLIST),
        (5, AlertLevel.WATCHLIST),
        (6, AlertLevel.HIGH_PRIORITY),
        (7, AlertLevel.HIGH_PRIORITY),
        (8, AlertLevel.STRONG_MOMENTUM),
        (10, AlertLevel.STRONG_MOMENTUM),
        (100, AlertLevel.STRONG_MOMENTUM),
    ],
)
def test_score_to_alert_level(score: int, expected_level: AlertLevel) -> None:
    """score_to_alert_level maps scores to the correct AlertLevel."""
    assert score_to_alert_level(score) == expected_level


def test_alert_level_enum_values() -> None:
    """AlertLevel enum contains expected string values."""
    assert AlertLevel.IGNORE.value == "ignore"
    assert AlertLevel.WATCHLIST.value == "watchlist"
    assert AlertLevel.HIGH_PRIORITY.value == "high_priority"
    assert AlertLevel.STRONG_MOMENTUM.value == "strong_momentum"


def test_compute_score_no_data() -> None:
    """compute_score handles None inputs without crashing."""
    # Import signal modules to populate the registry
    import momentum_radar.signals.volume  # noqa: F401
    import momentum_radar.signals.volatility  # noqa: F401
    import momentum_radar.signals.structure  # noqa: F401
    import momentum_radar.signals.short_interest  # noqa: F401
    import momentum_radar.signals.options_flow  # noqa: F401

    from momentum_radar.signals.scoring import compute_score

    result = compute_score(
        ticker="TEST",
        bars=None,
        daily=None,
        fundamentals=None,
        options=None,
    )
    assert isinstance(result["score"], int)
    assert result["score"] >= 0
    assert isinstance(result["alert_level"], AlertLevel)
    assert isinstance(result["triggered_modules"], list)


def test_compute_score_with_data(
    sample_intraday_bars, sample_daily_bars, sample_fundamentals, sample_options
) -> None:
    """compute_score returns a valid result with realistic data."""
    import momentum_radar.signals.volume  # noqa: F401
    import momentum_radar.signals.volatility  # noqa: F401
    import momentum_radar.signals.structure  # noqa: F401
    import momentum_radar.signals.short_interest  # noqa: F401
    import momentum_radar.signals.options_flow  # noqa: F401

    from momentum_radar.signals.scoring import compute_score

    result = compute_score(
        ticker="AAPL",
        bars=sample_intraday_bars,
        daily=sample_daily_bars,
        fundamentals=sample_fundamentals,
        options=sample_options,
    )
    assert "score" in result
    assert "alert_level" in result
    assert "triggered_modules" in result
    assert result["score"] >= 0


def test_market_penalty_reduces_score() -> None:
    """A market penalty reduces the total score (floor 0)."""
    import momentum_radar.signals.scoring as scoring

    # Temporarily add a predictable signal
    from momentum_radar.signals.base import SignalResult

    @scoring.register_signal("_test_constant_signal")
    def _constant(ticker, bars, daily, fundamentals, options, **kw):
        return SignalResult(triggered=True, score=5, details="test")

    result = scoring.compute_score(
        ticker="X",
        bars=None,
        daily=None,
        fundamentals=None,
        options=None,
        market_score_penalty=3,
    )
    assert result["score"] == 2  # 5 - 3

    result_floor = scoring.compute_score(
        ticker="X",
        bars=None,
        daily=None,
        fundamentals=None,
        options=None,
        market_score_penalty=10,
    )
    assert result_floor["score"] == 0  # floored at 0

    # Clean up the test signal
    del scoring._SIGNAL_REGISTRY["_test_constant_signal"]
