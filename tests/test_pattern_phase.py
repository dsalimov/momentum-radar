"""
test_pattern_phase.py – Tests for the PatternPhase enum,
compute_pattern_confidence_score, and the premarket scanner's new phase /
confidence_score fields (architecture spec §§ 3, 4, 8).
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_daily(
    n: int = 60,
    trend: str = "up",
    last_vol_mult: float = 1.0,
) -> pd.DataFrame:
    """Build a synthetic daily OHLCV DataFrame."""
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    np.random.seed(42)
    if trend == "up":
        closes = 100.0 + np.arange(n) * 0.5
    elif trend == "down":
        closes = 100.0 - np.arange(n) * 0.5
    else:
        closes = np.full(n, 100.0)
    volumes = np.full(n, 1_000_000.0)
    volumes[-1] *= last_vol_mult
    return pd.DataFrame(
        {
            "open": closes - 0.3,
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
            "volume": volumes,
        },
        index=rng,
    )


def _make_double_bottom_daily(n: int = 60) -> pd.DataFrame:
    """Daily OHLCV shaped like a double bottom."""
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = np.full(n, 100.0)
    closes[10:20] = 90.0
    closes[25:35] = 105.0
    closes[40:50] = 91.0
    closes[50:] = 106.0
    volumes = np.full(n, 1_000_000.0)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + 2.0,
            "low": closes - 2.0,
            "close": closes,
            "volume": volumes,
        },
        index=rng,
    )


# ---------------------------------------------------------------------------
# PatternPhase enum
# ---------------------------------------------------------------------------

class TestPatternPhaseEnum:
    def test_phase_values_exist(self):
        from momentum_radar.patterns.detector import PatternPhase

        assert PatternPhase.FORMATION.value == "forming"
        assert PatternPhase.CONFIRMATION.value == "confirming"
        assert PatternPhase.BREAKOUT.value == "breakout"

    def test_phase_is_distinct_from_state(self):
        from momentum_radar.patterns.detector import PatternPhase, PatternState

        assert PatternPhase is not PatternState
        assert set(p.value for p in PatternPhase) != set(s.value for s in PatternState)


# ---------------------------------------------------------------------------
# compute_pattern_confidence_score
# ---------------------------------------------------------------------------

class TestComputePatternConfidenceScore:
    def test_empty_result_returns_zero(self):
        from momentum_radar.patterns.detector import compute_pattern_confidence_score

        assert compute_pattern_confidence_score({}) == 0
        assert compute_pattern_confidence_score(None) == 0

    def test_returns_int_in_range(self):
        from momentum_radar.patterns.detector import compute_pattern_confidence_score

        result = {
            "confidence": 75.0,
            "volume_trend": "Declining (confirming)",
            "pattern": "bull flag",
            "key_points": [("2024-01-10", 100.0, "high"), ("2024-01-20", 98.0, "low")],
        }
        score = compute_pattern_confidence_score(result)
        assert isinstance(score, int)
        assert 0 <= score <= 100

    def test_score_increases_with_confidence(self):
        from momentum_radar.patterns.detector import compute_pattern_confidence_score

        low_conf = {"confidence": 60.0, "volume_trend": "", "pattern": "unknown", "key_points": []}
        high_conf = {"confidence": 95.0, "volume_trend": "", "pattern": "unknown", "key_points": []}

        assert compute_pattern_confidence_score(high_conf) >= compute_pattern_confidence_score(low_conf)

    def test_rising_volume_scores_higher_than_flat(self):
        from momentum_radar.patterns.detector import compute_pattern_confidence_score

        base = {"confidence": 75.0, "pattern": "bull flag", "key_points": []}
        flat = {**base, "volume_trend": "Flat/neutral"}
        rising = {**base, "volume_trend": "Increasing"}

        assert compute_pattern_confidence_score(rising) >= compute_pattern_confidence_score(flat)

    def test_trend_alignment_boosts_bullish_pattern(self):
        from momentum_radar.patterns.detector import compute_pattern_confidence_score

        df_up = _make_daily(n=60, trend="up")  # price > MA20 > MA50
        df_down = _make_daily(n=60, trend="down")

        result = {"confidence": 75.0, "volume_trend": "", "pattern": "bull flag", "key_points": []}
        score_aligned = compute_pattern_confidence_score(result, df_up)
        score_misaligned = compute_pattern_confidence_score(result, df_down)

        assert score_aligned >= score_misaligned

    def test_more_key_points_increases_level_strength(self):
        from momentum_radar.patterns.detector import compute_pattern_confidence_score

        base = {"confidence": 75.0, "volume_trend": "", "pattern": "unknown"}
        few_touches = {**base, "key_points": [("t", 100.0, "l"), ("t2", 99.0, "l2")]}
        many_touches = {**base, "key_points": [
            ("t1", 100.0, "l"), ("t2", 99.5, "l"), ("t3", 100.2, "l"), ("t4", 99.8, "l"),
        ]}

        assert compute_pattern_confidence_score(many_touches) >= compute_pattern_confidence_score(few_touches)

    def test_maximum_score_capped_at_100(self):
        from momentum_radar.patterns.detector import compute_pattern_confidence_score

        df = _make_daily(n=60, trend="up")
        result = {
            "confidence": 100.0,
            "volume_trend": "Increasing",
            "pattern": "bull flag",
            "key_points": [("t1", 1.0, "l")] * 5,
        }
        score = compute_pattern_confidence_score(result, df)
        assert score <= 100


# ---------------------------------------------------------------------------
# _compute_pattern_phase
# ---------------------------------------------------------------------------

class TestComputePatternPhase:
    def test_near_break_state_maps_to_confirmation(self):
        from momentum_radar.patterns.detector import (
            _compute_pattern_phase,
            PatternPhase,
            PatternState,
        )

        result = {
            "state": PatternState.NEAR_BREAK,
            "breakout_level_upper": 110.0,
            "breakout_level_lower": 90.0,
        }
        df = _make_daily(n=30)  # current price ~114, but stays inside upper
        # Ensure price is inside bounds for NEAR_BREAK
        df_inside = _make_daily(n=30, trend="neutral")
        # Current price ~100, upper=110, lower=90 → inside bounds → CONFIRMATION
        assert _compute_pattern_phase(result, df_inside) == PatternPhase.CONFIRMATION

    def test_forming_state_maps_to_formation(self):
        from momentum_radar.patterns.detector import (
            _compute_pattern_phase,
            PatternPhase,
            PatternState,
        )

        result = {
            "state": PatternState.FORMING,
            "breakout_level_upper": 200.0,
            "breakout_level_lower": 50.0,
        }
        df = _make_daily(n=30, trend="neutral")
        assert _compute_pattern_phase(result, df) == PatternPhase.FORMATION

    def test_breakout_detected_when_price_beyond_level_with_volume(self):
        from momentum_radar.patterns.detector import (
            _compute_pattern_phase,
            PatternPhase,
            PatternState,
        )

        # df where last close > breakout_level_upper AND volume surge
        df = _make_daily(n=30, trend="up", last_vol_mult=2.0)
        last_price = float(df["close"].iloc[-1])  # ~114.5

        result = {
            "state": PatternState.NEAR_BREAK,
            "breakout_level_upper": last_price - 5.0,  # price is above this
            "breakout_level_lower": 50.0,
        }
        phase = _compute_pattern_phase(result, df)
        assert phase == PatternPhase.BREAKOUT

    def test_no_df_returns_formation_for_forming_state(self):
        from momentum_radar.patterns.detector import (
            _compute_pattern_phase,
            PatternPhase,
            PatternState,
        )

        result = {"state": PatternState.FORMING}
        assert _compute_pattern_phase(result, None) == PatternPhase.FORMATION


# ---------------------------------------------------------------------------
# detect_pattern enrichment (phase + confidence_score in result)
# ---------------------------------------------------------------------------

class TestDetectPatternEnrichment:
    def test_result_contains_phase_key(self):
        from momentum_radar.patterns.detector import detect_pattern, PatternPhase

        df = _make_double_bottom_daily(n=90)
        result = detect_pattern("double bottom", df)
        if result is None:
            pytest.skip("No double bottom detected in synthetic data")
        assert "phase" in result
        assert isinstance(result["phase"], PatternPhase)

    def test_result_contains_confidence_score_key(self):
        from momentum_radar.patterns.detector import detect_pattern

        df = _make_double_bottom_daily(n=90)
        result = detect_pattern("double bottom", df)
        if result is None:
            pytest.skip("No double bottom detected in synthetic data")
        assert "confidence_score" in result
        assert isinstance(result["confidence_score"], int)
        assert 0 <= result["confidence_score"] <= 100

    def test_none_result_unchanged(self):
        from momentum_radar.patterns.detector import detect_pattern

        tiny = pd.DataFrame(
            {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [1e6]},
            index=pd.date_range("2024-01-01", periods=1, freq="B"),
        )
        result = detect_pattern("bull flag", tiny)
        assert result is None


# ---------------------------------------------------------------------------
# scan_swing_trade_setups – new phase and confidence_score fields
# ---------------------------------------------------------------------------

class TestScanSwingTradeSetupsNewFields:
    def _make_fetcher(self, daily):
        f = MagicMock()
        f.get_daily_bars.return_value = daily
        f.get_fundamentals.return_value = None
        return f

    def test_result_contains_phase_field(self):
        from momentum_radar.premarket.scanner import scan_swing_trade_setups

        fetcher = self._make_fetcher(_make_double_bottom_daily(n=60))
        results = scan_swing_trade_setups(["AAPL"], fetcher, min_confidence=0)
        if results:
            assert "phase" in results[0]
            assert results[0]["phase"] in ("forming", "confirming", "breakout")

    def test_result_contains_confidence_score_field(self):
        from momentum_radar.premarket.scanner import scan_swing_trade_setups

        fetcher = self._make_fetcher(_make_double_bottom_daily(n=60))
        results = scan_swing_trade_setups(["AAPL"], fetcher, min_confidence=0)
        if results:
            assert "confidence_score" in results[0]
            score = results[0]["confidence_score"]
            assert isinstance(score, int)
            assert 0 <= score <= 100

    def test_top_n_capped_at_15(self):
        from momentum_radar.premarket.scanner import scan_swing_trade_setups

        fetcher = MagicMock()
        fetcher.get_daily_bars.return_value = _make_double_bottom_daily(n=60)
        tickers = [f"T{i}" for i in range(30)]
        # Requesting 20 should be silently capped to 15
        results = scan_swing_trade_setups(tickers, fetcher, top_n=20, min_confidence=0)
        assert len(results) <= 15

    def test_default_top_n_still_returns_at_most_10(self):
        from momentum_radar.premarket.scanner import scan_swing_trade_setups

        fetcher = MagicMock()
        fetcher.get_daily_bars.return_value = _make_double_bottom_daily(n=60)
        tickers = [f"T{i}" for i in range(30)]
        results = scan_swing_trade_setups(tickers, fetcher, min_confidence=0)
        assert len(results) <= 10


# ---------------------------------------------------------------------------
# _run_categorized_signal_scan – blocked_alert_types parameter
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def in_memory_db():
    from momentum_radar.storage.database import init_db
    import momentum_radar.storage.database as db_module

    init_db("sqlite:///:memory:")
    yield
    db_module._ENGINE = None
    db_module._SessionLocal = None


def _make_signal_result(ticker, priority, confirmations):
    from momentum_radar.services.signal_engine import SignalResult

    return SignalResult(
        ticker=ticker,
        confirmations=confirmations,
        priority=priority,
        confidence_score=80.0,
    )


def _make_confirmation(name, category, detail, confidence=75.0):
    from momentum_radar.services.signal_engine import Confirmation

    return Confirmation(name=name, category=category, detail=detail, confidence=confidence)


class TestBlockedAlertTypes:
    def test_chart_pattern_blocked_when_specified(self):
        from unittest.mock import patch
        from momentum_radar.services.scheduler import _run_categorized_signal_scan

        fetcher = MagicMock()
        fetcher.get_intraday_bars.return_value = None
        fetcher.get_daily_bars.return_value = None
        fetcher.get_options_volume.return_value = None
        sent = []

        confs = [_make_confirmation("Double Bottom", "pattern", "support near $100")]
        sig = _make_signal_result("AAPL", "ALERT", confs)

        with patch("momentum_radar.services.signal_engine.evaluate", return_value=sig):
            _run_categorized_signal_scan(
                ["AAPL"], fetcher, sent.append,
                blocked_alert_types={"chart_pattern"},
            )

        # chart_pattern is blocked → no alert sent
        assert sent == []

    def test_non_blocked_type_still_sent(self):
        from unittest.mock import patch
        from momentum_radar.services.scheduler import _run_categorized_signal_scan

        fetcher = MagicMock()
        fetcher.get_intraday_bars.return_value = None
        fetcher.get_daily_bars.return_value = None
        fetcher.get_options_volume.return_value = None
        sent = []

        confs = [_make_confirmation("Golden Sweep", "options", "5000 contracts")]
        sig = _make_signal_result("NVDA", "ALERT", confs)

        with patch("momentum_radar.services.signal_engine.evaluate", return_value=sig):
            _run_categorized_signal_scan(
                ["NVDA"], fetcher, sent.append,
                blocked_alert_types={"chart_pattern"},  # only chart_pattern blocked
            )

        # options_flow is NOT blocked → alert should still be sent
        assert len(sent) == 1
        assert "OPTIONS FLOW SIGNAL" in sent[0]

    def test_no_blocked_types_behaves_as_before(self):
        """Passing blocked_alert_types=None (default) must not affect existing behaviour."""
        from unittest.mock import patch
        from momentum_radar.services.scheduler import _run_categorized_signal_scan

        fetcher = MagicMock()
        fetcher.get_intraday_bars.return_value = None
        fetcher.get_daily_bars.return_value = None
        fetcher.get_options_volume.return_value = None
        sent = []

        confs = [_make_confirmation("Double Bottom", "pattern", "support near $100")]
        sig = _make_signal_result("AAPL", "ALERT", confs)

        with patch("momentum_radar.services.signal_engine.evaluate", return_value=sig):
            _run_categorized_signal_scan(
                ["AAPL"], fetcher, sent.append,
                blocked_alert_types=None,
            )

        assert len(sent) == 1
        assert "CHART PATTERN SIGNAL" in sent[0]

    def test_empty_blocked_set_sends_all_alerts(self):
        from unittest.mock import patch
        from momentum_radar.services.scheduler import _run_categorized_signal_scan

        fetcher = MagicMock()
        fetcher.get_intraday_bars.return_value = None
        fetcher.get_daily_bars.return_value = None
        fetcher.get_options_volume.return_value = None
        sent = []

        confs = [_make_confirmation("Double Bottom", "pattern", "support near $100")]
        sig = _make_signal_result("AAPL", "ALERT", confs)

        with patch("momentum_radar.services.signal_engine.evaluate", return_value=sig):
            _run_categorized_signal_scan(
                ["AAPL"], fetcher, sent.append,
                blocked_alert_types=set(),
            )

        assert len(sent) == 1

    def test_all_chart_pattern_categories_blocked(self):
        """pattern, retest, supply_demand, and liquidity_sweep all map to chart_pattern."""
        from unittest.mock import patch
        from momentum_radar.services.scheduler import _run_categorized_signal_scan

        fetcher = MagicMock()
        fetcher.get_intraday_bars.return_value = None
        fetcher.get_daily_bars.return_value = None
        fetcher.get_options_volume.return_value = None
        sent = []

        confs = [
            _make_confirmation("Double Bottom", "pattern", "support"),
            _make_confirmation("Retest", "retest", "retest of support"),
            _make_confirmation("Supply Zone", "supply_demand", "supply overhead"),
            _make_confirmation("Liquidity Sweep", "liquidity_sweep", "wick below swing"),
        ]
        sig = _make_signal_result("AAPL", "HIGH_CONFIDENCE", confs)

        with patch("momentum_radar.services.signal_engine.evaluate", return_value=sig):
            _run_categorized_signal_scan(
                ["AAPL"], fetcher, sent.append,
                blocked_alert_types={"chart_pattern"},
            )

        # All four categories map to chart_pattern bucket → blocked
        assert sent == []
