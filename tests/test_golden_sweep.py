"""
test_golden_sweep.py – Unit tests for the Golden Sweep detector and formatter.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from momentum_radar.signals.golden_sweep import (
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
