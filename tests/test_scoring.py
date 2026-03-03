"""
test_scoring.py – Unit tests for signal scoring and alert level assignment.
"""

import pytest
import numpy as np
import pandas as pd

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


def test_compute_score_new_fields() -> None:
    """compute_score returns weighted_score, confirmation_count, module_scores, chop_suppressed."""
    import momentum_radar.signals.volume  # noqa: F401
    from momentum_radar.signals.scoring import compute_score

    result = compute_score(
        ticker="NEW",
        bars=None,
        daily=None,
        fundamentals=None,
        options=None,
    )
    assert "weighted_score" in result
    assert "confirmation_count" in result
    assert "module_scores" in result
    assert "chop_suppressed" in result
    assert isinstance(result["weighted_score"], int)
    assert isinstance(result["confirmation_count"], int)
    assert isinstance(result["module_scores"], dict)
    assert isinstance(result["chop_suppressed"], bool)


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


# ---------------------------------------------------------------------------
# Weighted score tests
# ---------------------------------------------------------------------------

class TestComputeWeightedScore:
    def test_empty_scores_returns_zero(self) -> None:
        from momentum_radar.signals.scoring import _compute_weighted_score
        assert _compute_weighted_score({}) == 0

    def test_untriggered_modules_contribute_zero(self) -> None:
        from momentum_radar.signals.scoring import _compute_weighted_score
        # score=0 means not triggered
        result = _compute_weighted_score({"ema_trend": 0, "rsi_macd": 0})
        assert result == 0

    def test_known_module_full_strength(self) -> None:
        from momentum_radar.signals.scoring import _compute_weighted_score, _MODULE_WEIGHTS
        # ema_trend weight=30, score=2 → 30 pts
        result = _compute_weighted_score({"ema_trend": 2})
        assert result == _MODULE_WEIGHTS["ema_trend"]

    def test_known_module_half_strength(self) -> None:
        from momentum_radar.signals.scoring import _compute_weighted_score, _MODULE_WEIGHTS
        # ema_trend weight=30, score=1 → 15 pts
        result = _compute_weighted_score({"ema_trend": 1})
        assert result == _MODULE_WEIGHTS["ema_trend"] // 2

    def test_unknown_module_uses_default_weight(self) -> None:
        from momentum_radar.signals.scoring import _compute_weighted_score, _DEFAULT_MODULE_WEIGHT
        result = _compute_weighted_score({"unknown_signal_xyz": 2})
        assert result == _DEFAULT_MODULE_WEIGHT

    def test_three_top_tier_modules_exceed_threshold(self) -> None:
        from momentum_radar.signals.scoring import _compute_weighted_score
        from momentum_radar.config import config
        # ema_trend(30) + rsi_macd(25) + volume_spike(25) = 80 > 75
        result = _compute_weighted_score({
            "ema_trend": 2,
            "rsi_macd": 2,
            "volume_spike": 2,
        })
        assert result >= config.scores.signal_score_minimum

    def test_two_modules_below_threshold(self) -> None:
        from momentum_radar.signals.scoring import _compute_weighted_score
        from momentum_radar.config import config
        # ema_trend(30) + rsi_macd(25) = 55 < 75
        result = _compute_weighted_score({
            "ema_trend": 2,
            "rsi_macd": 2,
        })
        assert result < config.scores.signal_score_minimum


# ---------------------------------------------------------------------------
# Chop filter tests
# ---------------------------------------------------------------------------

def _make_quiet_daily(n: int = 20, atr_mult: float = 0.3) -> pd.DataFrame:
    """Daily bars where the last bar's range is atr_mult × ATR (choppy)."""
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = np.full(n, 100.0)
    # Give ATR some body (1-point range on prior bars)
    highs = closes + 1.0
    lows = closes - 1.0
    # Last bar has a tiny range
    highs[-1] = 100.0 + atr_mult * 0.5
    lows[-1] = 100.0 - atr_mult * 0.5
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.full(n, 1_000_000.0),
        },
        index=rng,
    )


def _make_active_daily(n: int = 20) -> pd.DataFrame:
    """Daily bars where the last bar has a large range (> 1× ATR)."""
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = np.full(n, 100.0)
    highs = closes + 1.0
    lows = closes - 1.0
    # Last bar: 2× ATR range
    highs[-1] = 103.0
    lows[-1] = 97.0
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.full(n, 1_000_000.0),
        },
        index=rng,
    )


class TestChopFilter:
    def test_choppy_daily_returns_true(self) -> None:
        from momentum_radar.signals.scoring import _is_choppy_market
        daily = _make_quiet_daily(atr_mult=0.1)  # range=0.1 × ATR → way below 0.5
        assert _is_choppy_market(None, daily) is True

    def test_active_daily_returns_false(self) -> None:
        from momentum_radar.signals.scoring import _is_choppy_market
        daily = _make_active_daily()  # range=6 → well above ATR≈2
        assert _is_choppy_market(None, daily) is False

    def test_none_daily_returns_false(self) -> None:
        from momentum_radar.signals.scoring import _is_choppy_market
        assert _is_choppy_market(None, None) is False

    def test_choppy_market_suppresses_weighted_score(self) -> None:
        """compute_score sets weighted_score=0 and chop_suppressed=True in chop."""
        import momentum_radar.signals.volume  # noqa: F401
        from momentum_radar.signals.scoring import compute_score
        daily = _make_quiet_daily(atr_mult=0.1)
        result = compute_score(
            ticker="CHOP",
            bars=None,
            daily=daily,
            fundamentals=None,
            options=None,
        )
        assert result["chop_suppressed"] is True
        assert result["weighted_score"] == 0

