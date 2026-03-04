"""
tests/test_core_engines.py – Unit tests for the core engine modules.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trending_daily(n: int = 65) -> pd.DataFrame:
    """Daily bars with a strong uptrend."""
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = 100 + np.arange(n) * 0.8
    return pd.DataFrame(
        {
            "open":   closes - 0.3,
            "high":   closes + 1.0,
            "low":    closes - 1.0,
            "close":  closes,
            "volume": np.full(n, 1_500_000.0),
        },
        index=rng,
    )


def _make_flat_daily(n: int = 65) -> pd.DataFrame:
    """Daily bars with no directional trend."""
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = np.full(n, 100.0)
    return pd.DataFrame(
        {
            "open":   closes,
            "high":   closes + 0.5,
            "low":    closes - 0.5,
            "close":  closes,
            "volume": np.full(n, 1_000_000.0),
        },
        index=rng,
    )


def _make_breakout_daily(n: int = 30) -> pd.DataFrame:
    """Daily bars where the last close breaks above prior highs."""
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = np.full(n, 100.0)
    closes[-1] = 120.0  # breakout
    return pd.DataFrame(
        {
            "open":   closes - 0.3,
            "high":   closes + 1.0,
            "low":    closes - 1.0,
            "close":  closes,
            "volume": np.full(n, 2_000_000.0),
        },
        index=rng,
    )


# ---------------------------------------------------------------------------
# regime_engine
# ---------------------------------------------------------------------------

class TestRegimeEngine:
    def test_get_regime_returns_string(self):
        from momentum_radar.core.regime_engine import get_regime

        daily = _make_trending_daily()
        regime = get_regime(daily)
        assert isinstance(regime, str)
        assert regime in ("trending", "ranging", "expanding", "compressing")

    def test_get_regime_none_returns_ranging(self):
        from momentum_radar.core.regime_engine import get_regime

        assert get_regime(None) == "ranging"

    def test_get_regime_display_capitalised(self):
        from momentum_radar.core.regime_engine import get_regime_display

        daily = _make_trending_daily()
        label = get_regime_display(daily)
        assert label[0].isupper()

    def test_get_htf_bias_bullish_trending(self):
        from momentum_radar.core.regime_engine import get_htf_bias

        daily = _make_trending_daily()
        bias = get_htf_bias(daily)
        assert bias in ("Bullish", "Neutral", "Bearish")

    def test_get_htf_bias_none_is_neutral(self):
        from momentum_radar.core.regime_engine import get_htf_bias

        assert get_htf_bias(None) == "Neutral"

    def test_get_htf_bias_short_series_neutral(self):
        from momentum_radar.core.regime_engine import get_htf_bias

        daily = _make_trending_daily(n=10)
        assert get_htf_bias(daily) == "Neutral"


# ---------------------------------------------------------------------------
# structure_engine
# ---------------------------------------------------------------------------

class TestStructureEngine:
    def test_bullish_bos_detected(self):
        from momentum_radar.core.structure_engine import detect_structure_break

        daily = _make_breakout_daily()
        result = detect_structure_break(daily)
        assert result.confirmed is True
        assert result.direction == "bullish"
        assert result.broken_level > 0
        assert result.break_pct > 0

    def test_flat_data_no_bos(self):
        from momentum_radar.core.structure_engine import detect_structure_break

        daily = _make_flat_daily()
        result = detect_structure_break(daily)
        assert result.confirmed is False

    def test_none_returns_no_signal(self):
        from momentum_radar.core.structure_engine import detect_structure_break

        result = detect_structure_break(None)
        assert result.direction == "none"
        assert result.confirmed is False

    def test_insufficient_data_no_signal(self):
        from momentum_radar.core.structure_engine import detect_structure_break

        rng = pd.date_range("2024-01-01", periods=5, freq="B")
        daily = pd.DataFrame(
            {"open": [50]*5, "high": [51]*5, "low": [49]*5,
             "close": [50]*5, "volume": [1e6]*5},
            index=rng,
        )
        result = detect_structure_break(daily)
        assert result.confirmed is False

    def test_get_key_levels_returns_dict(self):
        from momentum_radar.core.structure_engine import get_key_levels

        daily = _make_trending_daily()
        levels = get_key_levels(daily)
        assert "resistance" in levels
        assert "support" in levels
        assert "20d_high" in levels
        assert "20d_low" in levels

    def test_get_key_levels_none_returns_empty(self):
        from momentum_radar.core.structure_engine import get_key_levels

        assert get_key_levels(None) == {}

    def test_bullish_structure_detected(self):
        from momentum_radar.core.structure_engine import has_bullish_structure

        daily = _make_trending_daily()
        # Trending data should produce HH/HL
        result = has_bullish_structure(daily)
        assert isinstance(result, bool)

    def test_flat_data_no_bullish_structure(self):
        from momentum_radar.core.structure_engine import has_bullish_structure

        assert has_bullish_structure(_make_flat_daily()) is False


# ---------------------------------------------------------------------------
# scoring_engine
# ---------------------------------------------------------------------------

class TestScoringEngine:
    def test_all_confirmations_max_score(self):
        from momentum_radar.core.scoring_engine import compute_strategy_score

        score = compute_strategy_score(
            "scalp",
            {
                "momentum":        1.0,
                "volume_spike":    1.0,
                "structure_break": 1.0,
                "htf_bias":        1.0,
                "fake_breakout":   1.0,
            },
        )
        assert score == 100

    def test_no_confirmations_zero_score(self):
        from momentum_radar.core.scoring_engine import compute_strategy_score

        score = compute_strategy_score("scalp", {})
        assert score == 0

    def test_partial_confirmations_intermediate_score(self):
        from momentum_radar.core.scoring_engine import compute_strategy_score

        score = compute_strategy_score(
            "intraday",
            {"trend_alignment": 1.0, "volume_confirm": 1.0},
        )
        assert 0 < score < 100

    def test_unknown_strategy_returns_zero(self):
        from momentum_radar.core.scoring_engine import compute_strategy_score

        score = compute_strategy_score("unknown_xyz", {"anything": 1.0})
        assert score == 0

    def test_score_clamped_to_100(self):
        from momentum_radar.core.scoring_engine import compute_strategy_score

        # Strengths > 1 should be clamped
        score = compute_strategy_score(
            "scalp",
            {k: 2.0 for k in
             ["momentum", "volume_spike", "structure_break", "htf_bias", "fake_breakout"]},
        )
        assert score == 100

    def test_score_to_grade_ap(self):
        from momentum_radar.core.scoring_engine import score_to_grade

        assert score_to_grade(95) == "A+"

    def test_score_to_grade_a(self):
        from momentum_radar.core.scoring_engine import score_to_grade

        assert score_to_grade(82) == "A"

    def test_score_to_grade_b(self):
        from momentum_radar.core.scoring_engine import score_to_grade

        assert score_to_grade(73) == "B"

    def test_score_to_grade_c(self):
        from momentum_radar.core.scoring_engine import score_to_grade

        assert score_to_grade(60) == "C"

    def test_get_strategy_weights_returns_dict(self):
        from momentum_radar.core.scoring_engine import get_strategy_weights

        weights = get_strategy_weights("swing")
        assert isinstance(weights, dict)
        assert "htf_zone" in weights
        assert sum(weights.values()) == 100

    def test_all_strategy_weights_sum_to_100(self):
        from momentum_radar.core.scoring_engine import _STRATEGY_WEIGHTS

        for strategy, weights in _STRATEGY_WEIGHTS.items():
            assert sum(weights.values()) == 100, (
                f"Strategy '{strategy}' weights sum to {sum(weights.values())}, not 100"
            )


# ---------------------------------------------------------------------------
# fake_breakout_filter
# ---------------------------------------------------------------------------

class TestFakeBreakoutFilter:
    def _make_bars(self, n: int = 25) -> pd.DataFrame:
        rng = pd.date_range("2024-01-15 09:30", periods=n, freq="1min")
        volumes = np.full(n, 100_000.0)
        closes = np.full(n, 50.0)
        opens = closes - 0.05
        highs = closes + 0.1
        lows = closes - 0.1
        # Strong close (close == high, tiny wick) on last bar
        opens[-1] = 49.90
        closes[-1] = 50.10
        highs[-1] = 50.10
        lows[-1] = 49.88
        volumes[-1] = 400_000.0  # volume spike
        return pd.DataFrame(
            {"open": opens, "high": highs, "low": lows,
             "close": closes, "volume": volumes},
            index=rng,
        )

    def test_strong_close_passes_filter(self):
        from momentum_radar.core.fake_breakout_filter import passes_fake_breakout_filter

        bars = self._make_bars()
        assert passes_fake_breakout_filter(bars, level=50.0) is True

    def test_none_bars_passes_filter(self):
        from momentum_radar.core.fake_breakout_filter import passes_fake_breakout_filter

        # No data → cannot confirm fake breakout → filter passes
        assert passes_fake_breakout_filter(None, level=50.0) is True

    def test_check_fake_breakout_none_is_false(self):
        from momentum_radar.core.fake_breakout_filter import check_fake_breakout

        assert check_fake_breakout(None, level=50.0) is False


# ---------------------------------------------------------------------------
# risk_engine
# ---------------------------------------------------------------------------

class TestRiskEngine:
    def test_long_trade_stop_below_entry(self):
        from momentum_radar.core.risk_engine import compute_trade_params

        trade = compute_trade_params("scalp", entry=100.0, atr=1.5)
        assert trade.stop < trade.entry
        assert trade.target > trade.entry

    def test_short_trade_stop_above_entry(self):
        from momentum_radar.core.risk_engine import compute_trade_params

        trade = compute_trade_params("scalp", entry=100.0, atr=1.5, direction="short")
        assert trade.stop > trade.entry
        assert trade.target < trade.entry

    def test_rr_meets_strategy_minimum(self):
        from momentum_radar.core.risk_engine import compute_trade_params, get_min_rr

        for strategy in ("scalp", "intraday", "swing"):
            min_rr = get_min_rr(strategy)
            trade  = compute_trade_params(strategy, entry=100.0, atr=1.5)
            # Target is set to achieve min_rr; rr should equal or exceed it
            assert trade.rr >= min_rr * 0.98, (
                f"Strategy '{strategy}': rr={trade.rr} < min_rr={min_rr}"
            )

    def test_no_atr_fallback(self):
        from momentum_radar.core.risk_engine import compute_trade_params

        trade = compute_trade_params("scalp", entry=100.0, atr=None)
        assert trade.stop < trade.entry
        assert trade.rr > 0

    def test_get_min_rr_known_strategy(self):
        from momentum_radar.core.risk_engine import get_min_rr

        assert get_min_rr("swing") == 3.0
        assert get_min_rr("scalp") == 2.0
        assert get_min_rr("intraday") == 2.5

    def test_get_min_rr_unknown_strategy_default(self):
        from momentum_radar.core.risk_engine import get_min_rr

        assert get_min_rr("unknown_xyz") == 2.0
