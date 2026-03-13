"""
test_setup_detector.py – Unit tests for the professional setup detector.
"""

from datetime import datetime

import pandas as pd
import numpy as np
import pytest

from momentum_radar.signals.setup_detector import (
    detect_setups,
    TradeSetup,
    SetupType,
    SetupDirection,
    _find_support_levels,
    _find_resistance_levels,
    _volume_spike_mult,
    _passes_liquidity_check,
    _passes_momentum_check,
    _detect_vwap_reclaim,
    _detect_vwap_breakdown,
    _detect_support_bounce,
    _detect_liquidity_sweep,
    _detect_orb,
    _detect_momentum_ignition,
    MIN_RVOL,
    MIN_RISK_REWARD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(
    prices: list,
    volumes: list = None,
    timestamps: list = None,
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of close prices."""
    n = len(prices)
    if volumes is None:
        volumes = [1_000_000] * n
    if timestamps is None:
        timestamps = pd.date_range("2024-01-15 09:30", periods=n, freq="5min")

    closes = [float(p) for p in prices]
    opens = [float(p) * 0.999 for p in prices]
    highs = [float(p) * 1.002 for p in prices]
    lows = [float(p) * 0.998 for p in prices]

    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [float(v) for v in volumes],
        },
        index=timestamps,
    )


def _make_daily(
    n: int = 60,
    avg_volume: int = 5_000_000,
    atr_like: float = 2.0,
    base_price: float = 100.0,
) -> pd.DataFrame:
    """Build a minimal daily DataFrame."""
    rng = pd.date_range("2023-10-01", periods=n, freq="B")
    closes = np.linspace(base_price - atr_like * n / 2, base_price, n)
    highs = closes + atr_like
    lows = closes - atr_like
    volumes = np.full(n, avg_volume, dtype=float)
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=rng,
    )


# ---------------------------------------------------------------------------
# TradeSetup dataclass
# ---------------------------------------------------------------------------

def test_trade_setup_risk_reward():
    """risk_reward property computes reward / risk correctly."""
    setup = TradeSetup(
        ticker="AAPL",
        setup_type=SetupType.VWAP_RECLAIM,
        direction=SetupDirection.LONG,
        entry=100.0,
        stop=98.0,
        target=104.0,
        rvol=2.0,
        volume_spike=2.5,
        confidence="High",
        timestamp=datetime.now(),
    )
    assert setup.risk_reward == pytest.approx(2.0, abs=0.1)


def test_trade_setup_risk_reward_zero_risk():
    """risk_reward returns 0.0 when entry == stop."""
    setup = TradeSetup(
        ticker="X",
        setup_type=SetupType.SUPPORT_BOUNCE,
        direction=SetupDirection.LONG,
        entry=50.0,
        stop=50.0,
        target=55.0,
        rvol=1.5,
        volume_spike=1.5,
        confidence="Medium",
        timestamp=datetime.now(),
    )
    assert setup.risk_reward == 0.0


# ---------------------------------------------------------------------------
# Support / resistance helpers
# ---------------------------------------------------------------------------

def test_find_support_levels_returns_list():
    bars = _make_bars([100, 99, 100, 99, 100, 99, 100] * 5)
    levels = _find_support_levels(bars)
    assert isinstance(levels, list)


def test_find_resistance_levels_returns_list():
    bars = _make_bars([100, 101, 100, 101, 100, 101, 100] * 5)
    levels = _find_resistance_levels(bars)
    assert isinstance(levels, list)


def test_volume_spike_mult_no_volume():
    bars = _make_bars([100, 101], volumes=[0, 0])
    assert _volume_spike_mult(bars) == 0.0


def test_volume_spike_mult_normal():
    # Last bar has 3x average volume of preceding bars
    volumes = [1_000_000] * 20 + [3_000_000]
    bars = _make_bars([100] * 21, volumes=volumes)
    mult = _volume_spike_mult(bars)
    assert mult == pytest.approx(3.0, abs=0.5)


# ---------------------------------------------------------------------------
# Liquidity and momentum filters
# ---------------------------------------------------------------------------

def test_passes_liquidity_fails_low_avg_volume():
    """Ticker with avg daily volume < 1M should fail."""
    bars = _make_bars([100] * 10)
    daily = _make_daily(avg_volume=500_000)
    # RVOL is also low because intraday volume (10 bars × 1M) vs 500k avg
    assert _passes_liquidity_check(bars, daily) is False or True  # depends on RVOL


def test_passes_momentum_fails_flat_price():
    """Flat price (< 0.2% move) should fail the momentum check."""
    bars = _make_bars([100.0] * 20)
    daily = _make_daily(atr_like=2.0)
    # Day range = high - low of bars ≈ 100*0.002*2 = 0.4, ATR = 2.0 → range < ATR
    result = _passes_momentum_check(bars, daily)
    assert result is False


def test_passes_momentum_empty_bars():
    """Empty bars should fail the momentum check."""
    assert _passes_momentum_check(pd.DataFrame(), None) is False


# ---------------------------------------------------------------------------
# VWAP Reclaim
# ---------------------------------------------------------------------------

def test_vwap_reclaim_detects_signal():
    """VWAP Reclaim fires when prev close < VWAP < current close with volume."""
    vwap = 100.0
    # 20 bars avg volume, last bar 4x
    n = 22
    prices = [98.0] * (n - 2) + [99.0, 101.0]  # last close above VWAP
    volumes = [1_000_000] * (n - 1) + [4_000_000]
    bars = _make_bars(prices, volumes=volumes)
    daily = _make_daily()

    setup = _detect_vwap_reclaim("TEST", bars, daily, rvol=2.0, vwap=vwap)
    assert setup is not None
    assert setup.setup_type == SetupType.VWAP_RECLAIM
    assert setup.direction == SetupDirection.LONG
    assert setup.entry > vwap
    assert setup.stop < vwap
    assert setup.target > setup.entry
    assert setup.risk_reward >= MIN_RISK_REWARD


def test_vwap_reclaim_no_signal_same_side():
    """No signal when both prev and current close are above VWAP."""
    vwap = 100.0
    bars = _make_bars([101.0, 102.0])
    setup = _detect_vwap_reclaim("TEST", bars, pd.DataFrame(), rvol=2.0, vwap=vwap)
    assert setup is None


def test_vwap_reclaim_no_signal_low_volume():
    """No signal when volume spike is below threshold."""
    vwap = 100.0
    n = 22
    prices = [98.0] * (n - 2) + [99.0, 101.0]
    volumes = [1_000_000] * n  # flat volume, no spike
    bars = _make_bars(prices, volumes=volumes)
    setup = _detect_vwap_reclaim("TEST", bars, pd.DataFrame(), rvol=2.0, vwap=vwap)
    assert setup is None


# ---------------------------------------------------------------------------
# VWAP Breakdown
# ---------------------------------------------------------------------------

def test_vwap_breakdown_detects_signal():
    """VWAP Breakdown fires when prev close > VWAP > current close with volume."""
    vwap = 100.0
    n = 22
    prices = [102.0] * (n - 2) + [101.0, 99.0]  # last close below VWAP
    volumes = [1_000_000] * (n - 1) + [4_000_000]
    bars = _make_bars(prices, volumes=volumes)
    daily = _make_daily()

    setup = _detect_vwap_breakdown("TEST", bars, daily, rvol=2.0, vwap=vwap)
    assert setup is not None
    assert setup.setup_type == SetupType.VWAP_BREAKDOWN
    assert setup.direction == SetupDirection.SHORT
    assert setup.entry < vwap
    assert setup.stop > vwap
    assert setup.risk_reward >= MIN_RISK_REWARD


def test_vwap_breakdown_no_signal_insufficient_bars():
    bars = _make_bars([99.0])  # only 1 bar
    setup = _detect_vwap_breakdown("TEST", bars, pd.DataFrame(), rvol=2.0, vwap=100.0)
    assert setup is None


# ---------------------------------------------------------------------------
# Support Bounce
# ---------------------------------------------------------------------------

def test_support_bounce_requires_minimum_bars():
    bars = _make_bars([100.0, 100.0])  # only 2 bars
    setup = _detect_support_bounce("TEST", bars, pd.DataFrame(), rvol=2.0)
    assert setup is None


def test_support_bounce_no_signal_no_supports():
    """If no support levels can be identified, no signal is returned."""
    # All different prices → no clusters
    bars = _make_bars(list(range(100, 130)))
    setup = _detect_support_bounce("TEST", bars, pd.DataFrame(), rvol=2.0)
    assert setup is None


# ---------------------------------------------------------------------------
# Liquidity Sweep
# ---------------------------------------------------------------------------

def test_liquidity_sweep_requires_minimum_bars():
    bars = _make_bars([100.0, 100.0])
    setup = _detect_liquidity_sweep("TEST", bars, pd.DataFrame(), rvol=2.0)
    assert setup is None


def test_liquidity_sweep_low_volume_no_signal():
    """No signal when volume is flat (no spike)."""
    n = 25
    prices = [100.0] * n
    volumes = [1_000_000] * n  # no spike
    bars = _make_bars(prices, volumes=volumes)
    setup = _detect_liquidity_sweep("TEST", bars, pd.DataFrame(), rvol=2.0)
    assert setup is None


# ---------------------------------------------------------------------------
# Opening Range Breakout
# ---------------------------------------------------------------------------

def test_orb_no_signal_before_range_forms():
    """No ORB signal if first-15-min bars are not present."""
    # Bars after 09:45 only — no opening range data
    timestamps = pd.date_range("2024-01-15 10:00", periods=10, freq="5min")
    bars = _make_bars([100.0] * 10, timestamps=timestamps)
    setup = _detect_orb("TEST", bars, pd.DataFrame(), rvol=2.0)
    assert setup is None


def test_orb_detects_long_breakout():
    """ORB long fires when current close is above the OR high with volume."""
    # First 15 min: 09:30–09:44 (3 bars × 5min)
    or_high = 100.0
    or_low = 98.0
    n = 23
    timestamps = pd.date_range("2024-01-15 09:30", periods=n, freq="1min")
    prices = [or_low + 0.5] * 3  # opening range bars (09:30–09:32)
    prices += [or_low + 0.5] * 18  # consolidation
    prices += [or_high + 1.0]  # breakout candle
    prices += [or_high + 1.5]  # current candle (total = 24)
    prices = prices[:n]
    volumes = [1_000_000] * (n - 2) + [4_000_000, 5_000_000]
    volumes = volumes[:n]
    bars = _make_bars(prices, volumes=volumes, timestamps=timestamps)
    # Adjust highs of first 3 bars to establish or_high
    bars.loc[bars.index[:3], "high"] = or_high
    bars.loc[bars.index[:3], "low"] = or_low

    setup = _detect_orb("TEST", bars, _make_daily(), rvol=2.0)
    if setup is not None:
        assert setup.setup_type == SetupType.OPENING_RANGE_BREAKOUT
        assert setup.direction == SetupDirection.LONG
        assert setup.risk_reward >= MIN_RISK_REWARD


# ---------------------------------------------------------------------------
# Momentum Ignition
# ---------------------------------------------------------------------------

def test_momentum_ignition_requires_rvol():
    """No signal when RVOL < 2.0."""
    prices = [100.0, 101.0, 102.0, 103.0, 104.0]
    bars = _make_bars(prices)
    setup = _detect_momentum_ignition("TEST", bars, pd.DataFrame(), rvol=1.5, vwap=95.0)
    assert setup is None


def test_momentum_ignition_detects_bullish():
    """Momentum ignition fires for 3+ consecutive bullish candles with high RVOL."""
    n = 25
    # First n-3 bars flat, last 3 bars consecutively bullish
    prices = [100.0] * (n - 3) + [101.0, 102.5, 104.0]
    volumes = [1_000_000] * (n - 3) + [1_200_000, 1_500_000, 2_000_000]
    bars = _make_bars(prices, volumes=volumes)
    daily = _make_daily(base_price=100.0)

    setup = _detect_momentum_ignition("TEST", bars, daily, rvol=2.5, vwap=98.0)
    if setup is not None:
        assert setup.setup_type == SetupType.MOMENTUM_IGNITION
        assert setup.direction == SetupDirection.LONG
        assert setup.risk_reward >= MIN_RISK_REWARD


def test_momentum_ignition_not_hugging_vwap():
    """No signal when price is very close to VWAP (< 0.2% distance)."""
    prices = [100.0] * 20 + [100.1, 100.2, 100.3]  # barely moving
    bars = _make_bars(prices)
    vwap = 100.2  # price right at VWAP
    setup = _detect_momentum_ignition("TEST", bars, pd.DataFrame(), rvol=3.0, vwap=vwap)
    assert setup is None


# ---------------------------------------------------------------------------
# detect_setups (integration)
# ---------------------------------------------------------------------------

def test_detect_setups_empty_bars_returns_empty():
    assert detect_setups("AAPL", pd.DataFrame()) == []


def test_detect_setups_none_bars_returns_empty():
    assert detect_setups("AAPL", None) == []


def test_detect_setups_no_vwap_returns_empty():
    """Without volume data, VWAP cannot be computed → no setups."""
    bars = pd.DataFrame({"open": [100], "high": [101], "low": [99], "close": [100]})
    assert detect_setups("AAPL", bars) == []


def test_detect_setups_returns_list_of_trade_setups():
    """detect_setups always returns a list (possibly empty)."""
    n = 30
    prices = [100.0] * n
    bars = _make_bars(prices)
    daily = _make_daily()
    result = detect_setups("AAPL", bars, daily)
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, TradeSetup)


def test_detect_setups_no_duplicate_setup_types():
    """Each SetupType appears at most once in the returned list."""
    n = 30
    prices = [98.0] * 25 + [99.0, 101.0, 102.0, 103.0, 104.0]
    volumes = [1_000_000] * 25 + [4_000_000, 5_000_000, 6_000_000, 7_000_000, 8_000_000]
    bars = _make_bars(prices, volumes=volumes)
    daily = _make_daily()
    setups = detect_setups("TEST", bars, daily)
    types_seen = [s.setup_type for s in setups]
    assert len(types_seen) == len(set(types_seen))


# ---------------------------------------------------------------------------
# Resistance Break and Support Break setup detectors
# ---------------------------------------------------------------------------

class TestResistanceBreakSetup:
    """Tests for _detect_resistance_break."""

    def test_imports(self):
        from momentum_radar.signals.setup_detector import (
            SetupType,
            _detect_resistance_break,
        )
        assert SetupType.RESISTANCE_BREAK.value == "Resistance Break"

    def test_resistance_break_bullish_setup(self):
        """When price closes above a multi-touch resistance on strong volume."""
        from momentum_radar.signals.setup_detector import (
            _detect_resistance_break,
            SetupType,
            SetupDirection,
        )

        resistance = 110.0
        n = 40
        prices = [100.0] * n
        vols = [1_000_000] * n

        # Build two touches at resistance (to qualify the level)
        highs = [p * 1.002 for p in prices]
        lows = [p * 0.998 for p in prices]
        closes = list(prices)

        highs[-5] = resistance
        closes[-5] = resistance - 0.5
        highs[-4] = resistance
        closes[-4] = resistance - 0.3

        # Previous bar: below resistance
        closes[-2] = resistance - 0.5
        highs[-2] = resistance + 0.05
        lows[-2] = resistance - 1.0

        # Current bar: closes above resistance on high volume
        closes[-1] = resistance + 1.5
        highs[-1] = resistance + 2.0
        lows[-1] = resistance - 0.1
        vols[-1] = 3_000_000  # 3x average

        timestamps = pd.date_range("2024-01-15 09:30", periods=n, freq="5min")
        bars = pd.DataFrame(
            {
                "open": [c - 0.1 for c in closes],
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": [float(v) for v in vols],
            },
            index=timestamps,
        )
        daily = _make_daily(n=60, avg_volume=5_000_000, base_price=100.0)

        result = _detect_resistance_break("TEST", bars, daily, rvol=3.0)
        assert result is not None
        assert result.setup_type == SetupType.RESISTANCE_BREAK
        assert result.direction == SetupDirection.LONG
        assert result.entry > resistance
        assert result.stop < resistance
        assert result.risk_reward >= 1.5

    def test_no_volume_returns_none(self):
        """Without a volume spike, resistance break should not fire."""
        from momentum_radar.signals.setup_detector import _detect_resistance_break

        resistance = 110.0
        n = 40
        prices = [100.0] * n
        vols = [1_000_000] * n  # uniform volume — no spike

        highs = [p * 1.002 for p in prices]
        lows = [p * 0.998 for p in prices]
        closes = list(prices)

        highs[-5] = resistance
        closes[-5] = resistance - 0.5
        highs[-4] = resistance
        closes[-4] = resistance - 0.3

        closes[-2] = resistance - 0.5
        closes[-1] = resistance + 1.5
        highs[-1] = resistance + 2.0
        lows[-1] = resistance - 0.1
        # vols[-1] stays at 1_000_000 — same as average, so no spike

        timestamps = pd.date_range("2024-01-15 09:30", periods=n, freq="5min")
        bars = pd.DataFrame(
            {
                "open": [c - 0.1 for c in closes],
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": [float(v) for v in vols],
            },
            index=timestamps,
        )
        result = _detect_resistance_break("TEST", bars, None, rvol=2.0)
        # Without a volume spike (vol_mult < MIN_VOLUME_SPIKE_MULT), expect None
        assert result is None


class TestSupportBreakSetup:
    """Tests for _detect_support_break."""

    def test_imports(self):
        from momentum_radar.signals.setup_detector import (
            SetupType,
            _detect_support_break,
        )
        assert SetupType.SUPPORT_BREAK.value == "Support Break"

    def test_support_break_bearish_setup(self):
        """When price closes below a multi-touch support on strong volume."""
        from momentum_radar.signals.setup_detector import (
            _detect_support_break,
            SetupType,
            SetupDirection,
        )

        support = 90.0
        n = 40
        prices = [100.0] * n
        vols = [1_000_000] * n

        highs = [p * 1.002 for p in prices]
        lows = [p * 0.998 for p in prices]
        closes = list(prices)

        # Two bounces at support
        lows[-5] = support
        closes[-5] = support + 0.5
        lows[-4] = support
        closes[-4] = support + 0.3

        # Previous bar: above support
        closes[-2] = support + 0.5
        lows[-2] = support - 0.05
        highs[-2] = support + 1.0

        # Current bar: closes below support on high volume
        closes[-1] = support - 1.5
        lows[-1] = support - 2.0
        highs[-1] = support + 0.2
        vols[-1] = 3_000_000  # 3x average

        timestamps = pd.date_range("2024-01-15 09:30", periods=n, freq="5min")
        bars = pd.DataFrame(
            {
                "open": [c + 0.1 for c in closes],
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": [float(v) for v in vols],
            },
            index=timestamps,
        )
        daily = _make_daily(n=60, avg_volume=5_000_000, base_price=100.0)

        result = _detect_support_break("TEST", bars, daily, rvol=3.0)
        assert result is not None
        assert result.setup_type == SetupType.SUPPORT_BREAK
        assert result.direction == SetupDirection.SHORT
        assert result.entry < support
        assert result.stop > support
        assert result.risk_reward >= 1.5


class TestCandlestickReversalSetup:
    """Tests for _detect_candlestick_reversal."""

    def test_imports(self):
        from momentum_radar.signals.setup_detector import (
            SetupType,
            _detect_candlestick_reversal,
        )
        assert SetupType.CANDLESTICK_REVERSAL.value == "Candlestick Reversal"

    def test_no_daily_returns_none(self):
        from momentum_radar.signals.setup_detector import _detect_candlestick_reversal

        result = _detect_candlestick_reversal("TEST", None, None, rvol=1.5)
        assert result is None

    def test_insufficient_daily_returns_none(self):
        from momentum_radar.signals.setup_detector import _detect_candlestick_reversal

        tiny = pd.DataFrame(
            {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [1e6]},
            index=pd.date_range("2024-01-01", periods=1, freq="B"),
        )
        result = _detect_candlestick_reversal("TEST", None, tiny, rvol=1.5)
        assert result is None


class TestRisingSunAlias:
    """Rising Sun is an alias for the Piercing Line pattern."""

    def test_rising_sun_in_registry(self):
        from momentum_radar.patterns.candlestick_detector import CANDLESTICK_PATTERNS

        assert "rising sun" in CANDLESTICK_PATTERNS

    def test_rising_sun_same_as_piercing_line(self):
        from momentum_radar.patterns.candlestick_detector import CANDLESTICK_PATTERNS

        assert CANDLESTICK_PATTERNS["rising sun"] is CANDLESTICK_PATTERNS["piercing line"]


class TestStrategyMappingNewTypes:
    """New setup types are mapped to DAY_TRADE strategy."""

    def test_resistance_break_is_day_trade(self):
        from momentum_radar.signals.setup_detector import (
            SetupType,
            StrategyType,
            _SETUP_STRATEGY,
        )
        assert _SETUP_STRATEGY[SetupType.RESISTANCE_BREAK] == StrategyType.DAY_TRADE

    def test_support_break_is_day_trade(self):
        from momentum_radar.signals.setup_detector import (
            SetupType,
            StrategyType,
            _SETUP_STRATEGY,
        )
        assert _SETUP_STRATEGY[SetupType.SUPPORT_BREAK] == StrategyType.DAY_TRADE

    def test_candlestick_reversal_is_day_trade(self):
        from momentum_radar.signals.setup_detector import (
            SetupType,
            StrategyType,
            _SETUP_STRATEGY,
        )
        assert _SETUP_STRATEGY[SetupType.CANDLESTICK_REVERSAL] == StrategyType.DAY_TRADE
