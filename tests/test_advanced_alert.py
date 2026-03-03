"""
tests/test_advanced_alert.py – Unit tests for advanced alert formatting,
probability scoring (confidence %, risk grade, setup strength), and
alert_stats win-rate utilities.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# format_advanced_alert
# ---------------------------------------------------------------------------


class TestFormatAdvancedAlert:
    def test_output_contains_required_fields(self):
        from momentum_radar.alerts.formatter import format_advanced_alert

        text = format_advanced_alert(
            ticker="XYZ",
            setup_type="Resistance Breakout",
            confidence_pct=82.0,
            entry=12.45,
            stop=11.90,
            target=14.20,
            rvol=2.4,
            market_regime="Risk-On",
            confirmation_count=3,
            win_rate_pct=63.0,
            options={"call_volume": 8000, "put_volume": 2000,
                     "avg_call_volume": 2000, "avg_put_volume": 2000},
        )
        for field in ["XYZ", "Resistance Breakout", "82%", "12.45", "11.90",
                      "14.20", "Risk-On", "63%"]:
            assert field in text, f"Expected '{field}' in alert text"

    def test_bullish_options_flow_label(self):
        from momentum_radar.alerts.formatter import format_advanced_alert

        text = format_advanced_alert(
            ticker="T",
            setup_type="Breakout",
            confidence_pct=75.0,
            entry=50.0,
            stop=48.0,
            target=55.0,
            rvol=2.0,
            market_regime="Neutral",
            options={"call_volume": 9000, "put_volume": 1000,
                     "avg_call_volume": 3000, "avg_put_volume": 3000},
        )
        assert "Bullish" in text

    def test_bearish_options_flow_label(self):
        from momentum_radar.alerts.formatter import format_advanced_alert

        text = format_advanced_alert(
            ticker="T",
            setup_type="Breakdown",
            confidence_pct=72.0,
            entry=50.0,
            stop=52.0,
            target=44.0,
            rvol=1.8,
            market_regime="Risk-Off",
            options={"call_volume": 1000, "put_volume": 9000,
                     "avg_call_volume": 2000, "avg_put_volume": 2000},
        )
        assert "Bearish" in text

    def test_no_options_shows_na(self):
        from momentum_radar.alerts.formatter import format_advanced_alert

        text = format_advanced_alert(
            ticker="T",
            setup_type="Squeeze",
            confidence_pct=70.0,
            entry=100.0,
            stop=97.0,
            target=107.0,
            rvol=1.5,
            market_regime="Neutral",
            options=None,
        )
        assert "N/A" in text

    def test_high_confidence_uses_fire_emoji(self):
        from momentum_radar.alerts.formatter import format_advanced_alert

        text = format_advanced_alert(
            ticker="T",
            setup_type="Setup",
            confidence_pct=85.0,
            entry=100.0,
            stop=97.0,
            target=107.0,
            rvol=2.0,
            market_regime="Risk-On",
        )
        assert "🔥" in text

    def test_medium_confidence_uses_alert_emoji(self):
        from momentum_radar.alerts.formatter import format_advanced_alert

        text = format_advanced_alert(
            ticker="T",
            setup_type="Setup",
            confidence_pct=73.0,
            entry=100.0,
            stop=97.0,
            target=107.0,
            rvol=2.0,
            market_regime="Neutral",
        )
        assert "🚨" in text

    def test_rr_computed_correctly(self):
        from momentum_radar.alerts.formatter import format_advanced_alert

        # Entry 50, stop 48 (risk=2), target 54 (reward=4) → R:R = 2.0
        text = format_advanced_alert(
            ticker="T",
            setup_type="Breakout",
            confidence_pct=75.0,
            entry=50.0,
            stop=48.0,
            target=54.0,
            rvol=1.5,
            market_regime="Neutral",
        )
        assert "2.0" in text

    def test_triggered_modules_listed(self):
        from momentum_radar.alerts.formatter import format_advanced_alert

        text = format_advanced_alert(
            ticker="T",
            setup_type="Multi",
            confidence_pct=75.0,
            entry=50.0,
            stop=48.0,
            target=55.0,
            rvol=2.0,
            market_regime="Risk-On",
            triggered_modules=["volume_spike", "options_flow"],
        )
        assert "Volume Spike" in text
        assert "Options Flow" in text


# ---------------------------------------------------------------------------
# Risk-grade and setup-strength helpers
# ---------------------------------------------------------------------------


class TestRiskGrade:
    def test_high_confidence_low_risk(self):
        from momentum_radar.alerts.formatter import _risk_grade

        assert _risk_grade(85.0) == "Low"

    def test_medium_confidence_medium_risk(self):
        from momentum_radar.alerts.formatter import _risk_grade

        assert _risk_grade(72.0) == "Medium"

    def test_low_confidence_high_risk(self):
        from momentum_radar.alerts.formatter import _risk_grade

        assert _risk_grade(55.0) == "High"


class TestSetupStrength:
    def test_three_confirmations_high_confidence_gives_aplus(self):
        from momentum_radar.alerts.formatter import _setup_strength

        assert _setup_strength(3, 82.0) == "A+"

    def test_three_confirmations_moderate_confidence_gives_a(self):
        from momentum_radar.alerts.formatter import _setup_strength

        assert _setup_strength(3, 70.0) == "A"

    def test_two_confirmations_gives_b(self):
        from momentum_radar.alerts.formatter import _setup_strength

        assert _setup_strength(2, 68.0) == "B"

    def test_one_confirmation_gives_c(self):
        from momentum_radar.alerts.formatter import _setup_strength

        assert _setup_strength(1, 60.0) == "C"


# ---------------------------------------------------------------------------
# Alert stats – win rates
# ---------------------------------------------------------------------------


class TestAlertStats:
    def test_known_setup_returns_nonzero_rate(self):
        from momentum_radar.storage.alert_stats import get_win_rate

        rate = get_win_rate("third_touch_support")
        assert rate > 0.0
        assert rate <= 100.0

    def test_unknown_setup_returns_default(self):
        from momentum_radar.storage.alert_stats import get_win_rate, _DEFAULT_WIN_RATE

        rate = get_win_rate("completely_unknown_setup_xyz123")
        assert rate == _DEFAULT_WIN_RATE

    def test_partial_match_resolves(self):
        from momentum_radar.storage.alert_stats import get_win_rate

        # "Bull trap: broke above" should match "Bull trap" key
        rate = get_win_rate("Bull trap: broke above $102.00")
        assert rate > 0.0

    def test_get_best_win_rate_empty(self):
        from momentum_radar.storage.alert_stats import get_best_win_rate, _DEFAULT_WIN_RATE

        rate = get_best_win_rate([])
        assert rate == _DEFAULT_WIN_RATE

    def test_get_best_win_rate_picks_highest(self):
        from momentum_radar.storage.alert_stats import get_best_win_rate

        rate = get_best_win_rate(["volume_spike", "third_touch_support", "options_flow"])
        # third_touch_support should be the highest at 67%
        assert rate >= 60.0


# ---------------------------------------------------------------------------
# Signal engine – new fields
# ---------------------------------------------------------------------------


class TestSignalEngineNewFields:
    def _make_breakout_daily(self, n: int = 65) -> pd.DataFrame:
        rng = pd.date_range("2024-01-01", periods=n, freq="B")
        closes = np.full(n, 100.0)
        closes[-1] = 120.0
        return pd.DataFrame(
            {
                "open": closes - 0.3,
                "high": closes + 1.0,
                "low": closes - 1.0,
                "close": closes,
                "volume": np.full(n, 2_000_000.0),
            },
            index=rng,
        )

    def _options_call_spike(self) -> dict:
        return {
            "call_volume": 10_000,
            "put_volume": 2_000,
            "avg_call_volume": 2_000,
            "avg_put_volume": 2_000,
        }

    def test_result_has_risk_grade(self):
        from momentum_radar.services.signal_engine import evaluate

        daily = self._make_breakout_daily()
        options = self._options_call_spike()
        result = evaluate("TEST", daily=daily, options=options)
        assert hasattr(result, "risk_grade")
        assert result.risk_grade in ("Low", "Medium", "High")

    def test_result_has_setup_strength(self):
        from momentum_radar.services.signal_engine import evaluate

        daily = self._make_breakout_daily()
        options = self._options_call_spike()
        result = evaluate("TEST", daily=daily, options=options)
        assert hasattr(result, "setup_strength")
        assert result.setup_strength in ("A+", "A", "B", "C")

    def test_high_confidence_gives_low_risk(self):
        from momentum_radar.services.signal_engine import evaluate

        daily = self._make_breakout_daily()
        options = self._options_call_spike()
        result = evaluate("TEST", daily=daily, options=options)
        # With 2+ confirmations confidence should be ≥65% → risk "Medium" or "Low"
        if result.confidence_score >= 80:
            assert result.risk_grade == "Low"
        elif result.confidence_score >= 65:
            assert result.risk_grade == "Medium"

    def test_no_signal_gives_c_grade(self):
        from momentum_radar.services.signal_engine import evaluate

        result = evaluate("EMPTY")
        assert result.priority == "NO_SIGNAL"
        assert result.setup_strength == "C"

    def test_confidence_score_in_valid_range(self):
        from momentum_radar.services.signal_engine import evaluate

        daily = self._make_breakout_daily()
        result = evaluate("TEST", daily=daily)
        assert 0.0 <= result.confidence_score <= 100.0


# ---------------------------------------------------------------------------
# Config – new S&D thresholds
# ---------------------------------------------------------------------------


class TestSDConfig:
    def test_default_sd_zone_min_score(self):
        from momentum_radar.config import config

        assert config.signals.sd_zone_min_score == pytest.approx(50.0)

    def test_default_sd_impulse_atr_mult(self):
        from momentum_radar.config import config

        assert config.signals.sd_impulse_atr_mult == pytest.approx(1.20)

    def test_default_sd_impulse_vol_mult(self):
        from momentum_radar.config import config

        assert config.signals.sd_impulse_vol_mult == pytest.approx(1.30)

    def test_default_sr_touch_tolerance(self):
        from momentum_radar.config import config

        assert config.signals.sr_touch_tolerance == pytest.approx(0.02)

    def test_default_min_confidence_pct(self):
        from momentum_radar.config import config

        assert config.scores.min_confidence_pct == pytest.approx(70.0)
