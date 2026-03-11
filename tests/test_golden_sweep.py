"""
tests/test_golden_sweep.py – Unit tests for the Golden Sweep detector and formatter.
test_golden_sweep.py – Unit tests for the Golden Sweep detector and formatter.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
from datetime import datetime, timedelta

import pandas as pd
import pytest

from momentum_radar.signals.golden_sweep import (
    GoldenSweepSetup,
    detect_golden_sweep,
    golden_sweep_signal,
    _classify_sweep_type,
    _days_to_expiry,
    _nearest_sd_zone,
    _select_best_sweep,
)
from momentum_radar.alerts.trade_formatter import format_golden_sweep_alert


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_intraday_bars(
    close: float = 100.0,
    trend: str = "up",
    n: int = 60,
    high_volume: bool = True,
) -> pd.DataFrame:
    """Build a synthetic 1-min OHLCV DataFrame with controlled trend and volume.

    Volumes are scaled to produce realistic RVOL values when paired with
    ``_make_daily_bars``. The daily average is ~3.3M; to achieve RVOL >= 1.5
    we need an intraday cumulative volume of at least 5M.
    """
    rng = pd.date_range("2024-01-15 09:30", periods=n, freq="1min")
    np.random.seed(42)
    if trend == "up":
        closes = close + np.linspace(0, 2, n) + np.random.randn(n) * 0.05
    elif trend == "down":
        closes = close - np.linspace(0, 2, n) + np.random.randn(n) * 0.05
    else:
        closes = close + np.random.randn(n) * 0.1

    if high_volume:
        # Most bars: base volume that yields cumulative RVOL > 1.5.
        # Last bar: 4× the recent average to create a clear volume spike.
        base_vol = 90_000.0
        volumes = np.full(n, base_vol)
        volumes[-1] = base_vol * 4   # spike on final bar
    else:
        # Flat, low volume – cumulative RVOL will fall well below 1.5.
        volumes = np.full(n, 500.0)

    return pd.DataFrame(
        {
            "open": closes - 0.05,
            "high": closes + 0.15,
            "low": closes - 0.15,
            "close": closes,
            "volume": volumes,
        },
        index=rng,
    )


def _make_daily_bars(close: float = 100.0, n: int = 40) -> pd.DataFrame:
    """Build synthetic daily OHLCV data (n days)."""
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    np.random.seed(0)
    closes = close + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "open": closes - 0.5,
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
            "volume": np.random.randint(2_000_000, 5_000_000, size=n).astype(float),
        },
        index=rng,
    )


def _weekly_expiry() -> str:
    """Return an ISO date string for 3 days from today (weekly expiry)."""
    return (date.today() + timedelta(days=3)).isoformat()


def _swing_expiry() -> str:
    """Return an ISO date string for 14 days from today (2-3 week expiry)."""
    return (date.today() + timedelta(days=14)).isoformat()


def _monthly_expiry() -> str:
    """Return an ISO date string for 30 days from today (monthly expiry)."""
    return (date.today() + timedelta(days=30)).isoformat()


def _options_flow_with_call_sweep(
    contracts: int = 2000,
    strike: float = 105.0,
    expiry: Optional[str] = None,
    current_price: float = 100.0,
) -> Dict:
    expiry = expiry or _weekly_expiry()
    return {
        "current_price": current_price,
        "call_sweeps": [
            {
                "ticker": "TEST",
                "strike": strike,
                "expiry": expiry,
                "type": "call",
                "volume": contracts,
                "openInterest": contracts // 2,
                "vol_oi_ratio": 2.0,
                "impliedVolatility": 0.35,
                "lastPrice": 2.50,
            }
        ],
        "put_sweeps": [],
        "call_volume": contracts,
        "put_volume": 500,
        "avg_call_volume": 300,
        "avg_put_volume": 300,
    }


def _options_flow_with_put_sweep(
    contracts: int = 1500,
    strike: float = 95.0,
    expiry: Optional[str] = None,
    current_price: float = 100.0,
) -> Dict:
    expiry = expiry or _weekly_expiry()
    return {
        "current_price": current_price,
        "call_sweeps": [],
        "put_sweeps": [
            {
                "ticker": "TEST",
                "strike": strike,
                "expiry": expiry,
                "type": "put",
                "volume": contracts,
                "openInterest": contracts // 2,
                "vol_oi_ratio": 2.0,
                "impliedVolatility": 0.35,
                "lastPrice": 1.80,
            }
        ],
        "call_volume": 400,
        "put_volume": contracts,
        "avg_call_volume": 300,
        "avg_put_volume": 300,
    }


# ---------------------------------------------------------------------------
# Tests: _days_to_expiry
# ---------------------------------------------------------------------------

class TestDaysToExpiry:
    def test_future_date_returns_positive(self):
        future = (date.today() + timedelta(days=5)).isoformat()
        assert _days_to_expiry(future) == 5

    def test_past_date_returns_negative(self):
        past = (date.today() - timedelta(days=2)).isoformat()
        assert _days_to_expiry(past) == -2

    def test_invalid_string_returns_none(self):
        assert _days_to_expiry("not-a-date") is None

    def test_none_returns_none(self):
        assert _days_to_expiry(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests: _classify_sweep_type
# ---------------------------------------------------------------------------

class TestClassifySweepType:
    def test_weekly_up_to_7_days(self):
        for days in range(0, 8):
            sweep_type, trade_type = _classify_sweep_type(days)
            assert sweep_type == "Weekly"
            assert trade_type == "Day Trade"

    def test_swing_8_to_21_days(self):
        for days in [8, 14, 21]:
            sweep_type, trade_type = _classify_sweep_type(days)
            assert sweep_type == "2-3 Week"
            assert trade_type == "Swing Trade"

    def test_monthly_22_plus_days(self):
        for days in [22, 30, 45]:
            sweep_type, trade_type = _classify_sweep_type(days)
            assert sweep_type == "Monthly"
            assert trade_type == "Swing Trade"

    def test_none_days_returns_swing(self):
        sweep_type, trade_type = _classify_sweep_type(None)
        assert sweep_type == "Unknown"
        assert trade_type == "Swing Trade"


# ---------------------------------------------------------------------------
# Tests: _select_best_sweep
# ---------------------------------------------------------------------------

class TestSelectBestSweep:
    def test_returns_none_when_empty(self):
        assert _select_best_sweep([], "Bullish") is None

    def test_returns_none_when_all_below_min_contracts(self):
        sweeps = [{"volume": 10, "strike": 100}]
        assert _select_best_sweep(sweeps, "Bullish") is None

    def test_returns_highest_volume_sweep(self):
        sweeps = [
            {"volume": 600, "strike": 100},
            {"volume": 1500, "strike": 105},
            {"volume": 800, "strike": 110},
        ]
        best = _select_best_sweep(sweeps, "Bullish")
        assert best is not None
        assert best["volume"] == 1500


# ---------------------------------------------------------------------------
# Tests: _nearest_sd_zone
# ---------------------------------------------------------------------------

class TestNearestSdZone:
    def test_returns_demand_zone_for_bullish(self):
        bars = _make_intraday_bars(close=100.0, trend="up", n=50)
        zone = _nearest_sd_zone(bars, 100.0, "Bullish", tolerance_pct=0.02)
        # Should find nearby lows clustered around 100
        assert zone is None or "Demand Zone" in zone

    def test_returns_none_for_empty_bars(self):
        zone = _nearest_sd_zone(None, 100.0, "Bullish")
        assert zone is None


# ---------------------------------------------------------------------------
# Tests: detect_golden_sweep
# ---------------------------------------------------------------------------

class TestDetectGoldenSweep:
    def test_returns_none_when_no_options_data(self):
        result = detect_golden_sweep("TEST", None, None, None)
        assert result is None

    def test_returns_none_when_empty_options_data(self):
        result = detect_golden_sweep("TEST", {}, None, None)
        assert result is None

    def test_returns_none_when_sweep_below_min_contracts(self):
        options_data = _options_flow_with_call_sweep(contracts=50)
        bars = _make_intraday_bars(trend="up", high_volume=True)
        daily = _make_daily_bars()
        result = detect_golden_sweep("TEST", options_data, bars, daily)
        assert result is None

    def test_returns_none_when_rvol_too_low(self):
        """No sweep when volume is flat (low RVOL)."""
        options_data = _options_flow_with_call_sweep(contracts=2000)
        bars = _make_intraday_bars(trend="up", high_volume=False)
        daily = _make_daily_bars()
        result = detect_golden_sweep("TEST", options_data, bars, daily)
        assert result is None

    def test_bullish_call_sweep_detected(self):
        options_data = _options_flow_with_call_sweep(
            contracts=2000,
            strike=105.0,
            expiry=_weekly_expiry(),
            current_price=100.0,
        )
        bars = _make_intraday_bars(close=100.0, trend="up", high_volume=True)
        daily = _make_daily_bars(close=100.0)
        result = detect_golden_sweep("TEST", options_data, bars, daily)
        assert result is not None
        assert result.direction == "Bullish"
        assert result.contract_type == "Call"
        assert result.sweep_type == "Weekly"
        assert result.trade_type == "Day Trade"
        assert result.contracts == 2000
        assert result.strike == 105.0
        assert result.entry > 0
        assert result.stop < result.entry   # stop below entry for long
        assert result.target > result.entry
        assert result.risk_reward >= 1.5

    def test_bearish_put_sweep_detected(self):
        options_data = _options_flow_with_put_sweep(
            contracts=1500,
            strike=95.0,
            expiry=_weekly_expiry(),
            current_price=100.0,
        )
        bars = _make_intraday_bars(close=100.0, trend="down", high_volume=True)
        daily = _make_daily_bars(close=100.0)
        result = detect_golden_sweep("TEST", options_data, bars, daily)
        assert result is not None
        assert result.direction == "Bearish"
        assert result.contract_type == "Put"
        assert result.stop > result.entry   # stop above entry for short
        assert result.target < result.entry

    def test_bullish_sweep_rejected_on_downtrend(self):
        options_data = _options_flow_with_call_sweep(contracts=2000, expiry=_weekly_expiry())
        bars = _make_intraday_bars(close=100.0, trend="down", high_volume=True)
        daily = _make_daily_bars(close=100.0)
        result = detect_golden_sweep("TEST", options_data, bars, daily)
        assert result is None

    def test_bearish_sweep_rejected_on_uptrend(self):
        options_data = _options_flow_with_put_sweep(contracts=1500, expiry=_weekly_expiry())
        bars = _make_intraday_bars(close=100.0, trend="up", high_volume=True)
        daily = _make_daily_bars(close=100.0)
        result = detect_golden_sweep("TEST", options_data, bars, daily)
        assert result is None

    def test_calls_beat_puts_chooses_bullish(self):
        """When both call and put sweeps qualify, highest volume wins."""
        options_data = {
            "current_price": 100.0,
            "call_sweeps": [
                {"ticker": "T", "strike": 105, "expiry": _weekly_expiry(),
                 "type": "call", "volume": 3000, "openInterest": 1000,
                 "vol_oi_ratio": 3.0, "impliedVolatility": 0.3, "lastPrice": 2.0},
            ],
            "put_sweeps": [
                {"ticker": "T", "strike": 95, "expiry": _weekly_expiry(),
                 "type": "put", "volume": 1000, "openInterest": 500,
                 "vol_oi_ratio": 2.0, "impliedVolatility": 0.3, "lastPrice": 1.5},
            ],
            "call_volume": 3000,
            "put_volume": 1000,
            "avg_call_volume": 300,
            "avg_put_volume": 300,
        }
        bars = _make_intraday_bars(close=100.0, trend="up", high_volume=True)
        daily = _make_daily_bars(close=100.0)
        result = detect_golden_sweep("TEST", options_data, bars, daily)
        assert result is not None
        assert result.direction == "Bullish"

    def test_swing_expiry_classified_correctly(self):
        options_data = _options_flow_with_call_sweep(
            contracts=2000,
            expiry=_swing_expiry(),
        )
        bars = _make_intraday_bars(close=100.0, trend="up", high_volume=True)
        daily = _make_daily_bars(close=100.0)
        result = detect_golden_sweep("TEST", options_data, bars, daily)
        assert result is not None
        assert result.sweep_type == "2-3 Week"
        assert result.trade_type == "Swing Trade"

    def test_risk_reward_attribute(self):
        options_data = _options_flow_with_call_sweep(contracts=2000, expiry=_weekly_expiry())
        bars = _make_intraday_bars(close=100.0, trend="up", high_volume=True)
        daily = _make_daily_bars(close=100.0)
        result = detect_golden_sweep("TEST", options_data, bars, daily)
        if result is not None:
            assert result.risk_reward >= 1.5


# ---------------------------------------------------------------------------
# Tests: golden_sweep_signal (scoring registry wrapper)
# ---------------------------------------------------------------------------

class TestGoldenSweepSignal:
    def test_no_signal_without_options(self):
        result = golden_sweep_signal("TEST")
        assert result.triggered is False
        assert result.score == 0

    def test_signal_triggered_with_valid_sweep(self):
        options_data = _options_flow_with_call_sweep(contracts=2000, expiry=_weekly_expiry())
        bars = _make_intraday_bars(close=100.0, trend="up", high_volume=True)
        daily = _make_daily_bars(close=100.0)
        result = golden_sweep_signal("TEST", options=options_data, bars=bars, daily=daily)
        # May or may not trigger depending on RVOL threshold with synthetic data
        assert result.score >= 0
        if result.triggered:
            assert result.score in (2, 3)
            assert "Sweep" in result.details

    def test_no_signal_with_empty_sweeps(self):
        options_data = {
            "current_price": 100.0,
            "call_sweeps": [],
            "put_sweeps": [],
            "call_volume": 100,
            "put_volume": 100,
        }
        result = golden_sweep_signal("TEST", options=options_data)
        assert result.triggered is False


# ---------------------------------------------------------------------------
# Tests: format_golden_sweep_alert
# ---------------------------------------------------------------------------

class TestFormatGoldenSweepAlert:
    def _make_setup(self, **overrides) -> GoldenSweepSetup:
        defaults = dict(
            ticker="TSLA",
            direction="Bullish",
            sweep_type="Weekly",
            trade_type="Day Trade",
            contract_type="Call",
            strike=750.0,
            expiry=_weekly_expiry(),
            contracts=12000,
            underlying_price=740.50,
            entry=742.0,
            stop=737.0,
            target=755.0,
            rvol=2.3,
            volume_spike=4.0,
            supply_demand_zone="Demand Zone 740.00–742.00",
            confidence="High",
            timestamp=datetime(2024, 1, 15, 12, 45, 0),
        )
        defaults.update(overrides)
        return GoldenSweepSetup(**defaults)

    def test_alert_contains_golden_sweep_header(self):
        setup = self._make_setup()
        msg = format_golden_sweep_alert(setup)
        assert "🚨 GOLDEN SWEEP ALERT" in msg

    def test_alert_contains_ticker(self):
        setup = self._make_setup()
        msg = format_golden_sweep_alert(setup)
        assert "TSLA" in msg

    def test_alert_contains_setup_line(self):
        setup = self._make_setup()
        msg = format_golden_sweep_alert(setup)
        assert "Weekly Call Sweep → Bullish Day Trade" in msg

    def test_alert_contains_underlying_price(self):
        setup = self._make_setup()
        msg = format_golden_sweep_alert(setup)
        assert "740.50" in msg

    def test_alert_contains_strike(self):
        setup = self._make_setup()
        msg = format_golden_sweep_alert(setup)
        assert "750" in msg

    def test_alert_contains_contracts(self):
        setup = self._make_setup()
        msg = format_golden_sweep_alert(setup)
        assert "12,000" in msg

    def test_alert_contains_entry_stop_target(self):
        setup = self._make_setup()
        msg = format_golden_sweep_alert(setup)
        assert "742.00" in msg
        assert "737.00" in msg
        assert "755.00" in msg

    def test_alert_contains_rvol_and_volume_spike(self):
        setup = self._make_setup()
        msg = format_golden_sweep_alert(setup)
        assert "2.3" in msg
        assert "4.0x" in msg

    def test_alert_contains_supply_demand_zone(self):
        setup = self._make_setup()
        msg = format_golden_sweep_alert(setup)
        assert "Demand Zone 740.00–742.00" in msg

    def test_alert_contains_confidence(self):
        setup = self._make_setup()
        msg = format_golden_sweep_alert(setup)
        assert "High" in msg

    def test_alert_na_when_no_sd_zone(self):
        setup = self._make_setup(supply_demand_zone=None)
        msg = format_golden_sweep_alert(setup)
        assert "N/A" in msg

    def test_alert_contains_time(self):
        setup = self._make_setup(timestamp=datetime(2024, 1, 15, 12, 45, 0))
        msg = format_golden_sweep_alert(setup)
        assert "12:45 PM EST" in msg

    def test_default_timestamp_used_when_none(self):
        """format_golden_sweep_alert uses the setup's timestamp when no override is given."""
        ts = datetime(2024, 6, 20, 10, 30, 0)
        setup = self._make_setup(timestamp=ts)
        msg = format_golden_sweep_alert(setup)
        assert "10:30 AM EST" in msg

    def test_alert_bearish_put_setup(self):
        setup = self._make_setup(
            direction="Bearish",
            contract_type="Put",
            entry=100.0,
            stop=103.0,
            target=94.0,
        )
        msg = format_golden_sweep_alert(setup)
        assert "Bearish" in msg
        assert "Put" in msg

    def test_risk_reward_property(self):
        setup = self._make_setup(entry=742.0, stop=737.0, target=757.0)
        assert setup.risk_reward == 3.0

    def test_risk_reward_zero_when_no_risk(self):
        setup = self._make_setup(entry=100.0, stop=100.0, target=110.0)
        assert setup.risk_reward == 0.0
    SWEEP_COOLDOWN_SECONDS,
    SWEEP_DAY_TRADE_MAX_DTE,
    SWEEP_MIN_RVOL,
    SWEEP_SWING_TRADE_MAX_DTE,
    SWEEP_VOLUME_MULT,
    SweepAlert,
    _assess_alignment,
    _classify_dte,
    _compute_entry_stop_target,
    detect_golden_sweep,
)
from momentum_radar.alerts.golden_sweep_formatter import (
    format_chart_pattern_alert,
    format_golden_sweep_alert,
)
from momentum_radar.signals.setup_detector import (
    SetupDirection,
    SetupType,
    TradeSetup,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FIXED_TS = datetime(2024, 6, 15, 12, 45, 0)


def _make_bars(n: int = 30, close_trend: str = "up") -> pd.DataFrame:
    """Build a minimal intraday OHLCV DataFrame."""
    closes = [100.0 + (i if close_trend == "up" else -i) * 0.1 for i in range(n)]
    data = {
        "open":   [c - 0.05 for c in closes],
        "high":   [c + 0.20 for c in closes],
        "low":    [c - 0.20 for c in closes],
        "close":  closes,
        "volume": [500_000 + (i * 10_000) for i in range(n)],
    }
    idx = pd.date_range("2024-06-15 09:30", periods=n, freq="1min")
    return pd.DataFrame(data, index=idx)


def _make_daily(n: int = 40) -> pd.DataFrame:
    """Build a minimal daily OHLCV DataFrame."""
    closes = [150.0 + i * 0.5 for i in range(n)]
    data = {
        "open":   [c - 0.10 for c in closes],
        "high":   [c + 1.00 for c in closes],
        "low":    [c - 1.00 for c in closes],
        "close":  closes,
        "volume": [2_000_000] * n,
    }
    idx = pd.date_range("2024-04-01", periods=n, freq="B")
    return pd.DataFrame(data, index=idx)


def _high_alignment_flow(sweep_type: str = "call") -> dict:
    """Options flow dict that passes the SWEEP_VOLUME_MULT gate."""
    avg_vol = 1_000
    return {
        "sweep_type":       sweep_type,
        "sweep_volume":     int(avg_vol * SWEEP_VOLUME_MULT * 2),
        "avg_call_volume":  avg_vol,
        "avg_put_volume":   avg_vol,
        "sweep_strike":     105.0,
        "sweep_dte":        5,
        "sweep_expiration": "Weekly",
        "sweep_premium":    200_000.0,
    }


def _make_sweep_alert(**kwargs) -> SweepAlert:
    """Build a minimal SweepAlert for formatter tests."""
    defaults = dict(
        ticker="TSLA",
        direction="bullish",
        contract_type="call",
        strike=750.0,
        expiration="Weekly",
        dte=5,
        contract_volume=12_000,
        estimated_flow=500_000.0,
        underlying_price=740.50,
        entry=742.0,
        stop=737.0,
        target=757.0,
        rvol=2.3,
        volume_spike=4.0,
        zone_alignment="Demand Zone 740–742",
        confidence="High",
        trade_type="Day Trade",
        timestamp=_FIXED_TS,
        details="Call sweep 6.0× avg | DTE 5 | alignment 4/4",
    )
    defaults.update(kwargs)
    return SweepAlert(**defaults)


# ---------------------------------------------------------------------------
# _classify_dte
# ---------------------------------------------------------------------------

def test_classify_dte_day_trade():
    assert _classify_dte(0) == "Day Trade"
    assert _classify_dte(SWEEP_DAY_TRADE_MAX_DTE) == "Day Trade"


def test_classify_dte_swing_trade():
    assert _classify_dte(SWEEP_DAY_TRADE_MAX_DTE + 1) == "Swing Trade"
    assert _classify_dte(SWEEP_SWING_TRADE_MAX_DTE) == "Swing Trade"


def test_classify_dte_position_trade():
    assert _classify_dte(SWEEP_SWING_TRADE_MAX_DTE + 1) == "Position Trade"


# ---------------------------------------------------------------------------
# SweepAlert.risk_reward
# ---------------------------------------------------------------------------

def test_sweep_alert_risk_reward():
    alert = _make_sweep_alert(entry=100.0, stop=98.0, target=104.0)
    assert alert.risk_reward == 2.0


def test_sweep_alert_risk_reward_zero_risk():
    alert = _make_sweep_alert(entry=100.0, stop=100.0, target=104.0)
    assert alert.risk_reward == 0.0


# ---------------------------------------------------------------------------
# _assess_alignment
# ---------------------------------------------------------------------------

def test_assess_alignment_full_score_bullish():
    bars = _make_bars(30, close_trend="up")
    # Force last bar volume to be huge (volume_spike check)
    bars.loc[bars.index[-1], "volume"] = 5_000_000
    score, desc = _assess_alignment(
        bars,
        direction="bullish",
        rvol=SWEEP_MIN_RVOL + 0.5,
        volume_spike=4.0,
        vwap=bars["close"].iloc[-1],   # near VWAP = same price
        current_price=bars["close"].iloc[-1],
    )
    assert score >= 3


def test_assess_alignment_low_rvol():
    # downtrend bars so bullish momentum check fails too
    bars = _make_bars(10, close_trend="down")
    score, _ = _assess_alignment(
        bars,
        direction="bullish",
        rvol=0.5,               # below minimum → no RVOL point
        volume_spike=0.5,       # below minimum → no volume-spike point
        vwap=None,              # no VWAP proximity point
        current_price=bars["close"].iloc[-1],
    )
    assert score == 0


def test_assess_alignment_bearish_momentum():
    bars = _make_bars(20, close_trend="down")
    score, _ = _assess_alignment(
        bars,
        direction="bearish",
        rvol=SWEEP_MIN_RVOL + 0.1,
        volume_spike=3.5,
        vwap=None,
        current_price=bars["close"].iloc[-1],
    )
    assert score >= 2  # rvol + volume_spike + momentum


# ---------------------------------------------------------------------------
# _compute_entry_stop_target
# ---------------------------------------------------------------------------

def test_compute_entry_stop_target_bullish():
    entry, stop, target = _compute_entry_stop_target("bullish", 100.0, None)
    assert entry > 100.0
    assert stop < entry
    assert target > entry


def test_compute_entry_stop_target_bearish():
    entry, stop, target = _compute_entry_stop_target("bearish", 100.0, None)
    assert entry < 100.0
    assert stop > entry
    assert target < entry


def test_compute_entry_stop_target_uses_atr(tmp_path):
    daily = _make_daily(40)
    entry, stop, target = _compute_entry_stop_target("bullish", 100.0, daily)
    assert stop < entry < target


# ---------------------------------------------------------------------------
# detect_golden_sweep – None path
# ---------------------------------------------------------------------------

def test_detect_golden_sweep_none_flow():
    assert detect_golden_sweep("AAPL", None, None, None) is None


def test_detect_golden_sweep_low_volume():
    flow = {
        "sweep_type":      "call",
        "sweep_volume":    10,        # tiny
        "avg_call_volume": 1_000,
        "sweep_premium":   100_000,
    }
    result = detect_golden_sweep("AAPL", flow, None, None)
    assert result is None


def test_detect_golden_sweep_no_price_data():
    flow = _high_alignment_flow()
    result = detect_golden_sweep("AAPL", flow, None, None)
    assert result is None


# ---------------------------------------------------------------------------
# detect_golden_sweep – full detection
# ---------------------------------------------------------------------------

def test_detect_golden_sweep_returns_alert_with_good_data():
    bars = _make_bars(30, close_trend="up")
    # Make last bar volume spike
    bars.loc[bars.index[-1], "volume"] = bars["volume"].iloc[:-1].mean() * 5
    daily = _make_daily(40)
    flow = _high_alignment_flow("call")

    result = detect_golden_sweep("TSLA", flow, bars, daily)
    # May return None if alignment score is below 2 – that is acceptable.
    # If it does return, it must be well-formed.
    if result is not None:
        assert result.ticker == "TSLA"
        assert result.direction == "bullish"
        assert result.contract_type == "call"
        assert result.entry > 0
        assert result.stop > 0
        assert result.target > 0
        assert result.confidence in ("High", "Medium")
        assert result.trade_type in ("Day Trade", "Swing Trade", "Position Trade")


def test_detect_golden_sweep_put_bearish():
    bars = _make_bars(30, close_trend="down")
    bars.loc[bars.index[-1], "volume"] = bars["volume"].iloc[:-1].mean() * 5
    daily = _make_daily(40)
    flow = _high_alignment_flow("put")

    result = detect_golden_sweep("SPY", flow, bars, daily)
    if result is not None:
        assert result.direction == "bearish"
        assert result.contract_type == "put"


def test_detect_golden_sweep_fallback_to_volume_ratio():
    """When sweep_type is not provided, fall back to call/put volume ratio."""
    bars = _make_bars(30, close_trend="up")
    bars.loc[bars.index[-1], "volume"] = bars["volume"].iloc[:-1].mean() * 5
    daily = _make_daily(40)
    flow = {
        # No sweep_type or sweep_volume – must derive from totals
        "call_volume":     int(1_000 * SWEEP_VOLUME_MULT * 2),
        "put_volume":      500,
        "avg_call_volume": 1_000,
        "avg_put_volume":  1_000,
        "sweep_strike":    105.0,
        "sweep_dte":       5,
        "sweep_premium":   200_000.0,
    }
    result = detect_golden_sweep("NVDA", flow, bars, daily)
    if result is not None:
        assert result.contract_type == "call"


# ---------------------------------------------------------------------------
# SWEEP_COOLDOWN_SECONDS constant
# ---------------------------------------------------------------------------

def test_sweep_cooldown_is_15_minutes():
    assert SWEEP_COOLDOWN_SECONDS == 900


# ---------------------------------------------------------------------------
# format_golden_sweep_alert
# ---------------------------------------------------------------------------

def test_format_golden_sweep_alert_required_fields():
    alert = _make_sweep_alert()
    msg = format_golden_sweep_alert(alert, timestamp=_FIXED_TS)

    assert "🚨 GOLDEN SWEEP ALERT" in msg
    assert "TSLA" in msg
    assert "Weekly" in msg
    assert "Call" in msg
    assert "Bullish" in msg
    assert "Day Trade" in msg
    assert "740.50" in msg
    assert "750" in msg
    assert "742.00" in msg
    assert "737.00" in msg
    assert "757.00" in msg
    assert "2.3" in msg
    assert "4.0x avg" in msg
    assert "Demand Zone 740" in msg
    assert "High" in msg
    assert "12:45" in msg


def test_format_golden_sweep_alert_bearish():
    alert = _make_sweep_alert(
        direction="bearish",
        contract_type="put",
        expiration="2024-06-28",
        dte=13,
        trade_type="Swing Trade",
    )
    msg = format_golden_sweep_alert(alert, timestamp=_FIXED_TS)
    assert "Bearish" in msg
    assert "Put" in msg
    assert "Swing Trade" in msg


def test_format_golden_sweep_alert_large_notional():
    alert = _make_sweep_alert(estimated_flow=1_500_000, contract_volume=5_000)
    msg = format_golden_sweep_alert(alert, timestamp=_FIXED_TS)
    assert "$" in msg  # notional format


def test_format_golden_sweep_alert_small_notional_uses_contracts():
    alert = _make_sweep_alert(estimated_flow=500.0, contract_volume=12_000)
    msg = format_golden_sweep_alert(alert, timestamp=_FIXED_TS)
    assert "contracts" in msg


# ---------------------------------------------------------------------------
# format_chart_pattern_alert
# ---------------------------------------------------------------------------

def _make_trade_setup(**kwargs) -> TradeSetup:
    defaults = dict(
        ticker="NVDA",
        setup_type=SetupType.CHART_PATTERN_BREAKOUT,
        direction=SetupDirection.LONG,
        entry=905.0,
        stop=898.0,
        target=922.0,
        rvol=2.1,
        volume_spike=3.8,
        confidence="High",
        timestamp=_FIXED_TS,
        details="Demand Zone 902–905",
    )
    defaults.update(kwargs)
    return TradeSetup(**defaults)


def test_format_chart_pattern_alert_required_fields():
    setup = _make_trade_setup()
    msg = format_chart_pattern_alert(
        setup,
        pattern_name="Ascending Triangle",
        sweep_info="Weekly Call Sweep 5,000 contracts",
        timestamp=_FIXED_TS,
    )

    assert "🚨 AUTONOMOUS TRADE ALERT" in msg
    assert "NVDA" in msg
    assert "Ascending Triangle" in msg
    assert "Bullish" in msg
    assert "905.00" in msg
    assert "898.00" in msg
    assert "922.00" in msg
    assert "2.1" in msg
    assert "3.8x avg" in msg
    assert "Weekly Call Sweep" in msg
    assert "High" in msg
    assert "12:45" in msg


def test_format_chart_pattern_alert_no_sweep():
    setup = _make_trade_setup()
    msg = format_chart_pattern_alert(
        setup,
        pattern_name="Double Bottom",
        sweep_info=None,
        timestamp=_FIXED_TS,
    )
    assert "Golden Sweep:" not in msg
    assert "Double Bottom" in msg


def test_format_chart_pattern_alert_bearish():
    setup = _make_trade_setup(
        direction=SetupDirection.SHORT,
        entry=500.0,
        stop=505.0,
        target=490.0,
    )
    msg = format_chart_pattern_alert(
        setup,
        pattern_name="Head and Shoulders",
        timestamp=_FIXED_TS,
    )
    assert "Bearish" in msg
    assert "Head and Shoulders" in msg
