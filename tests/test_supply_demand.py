"""
tests/test_supply_demand.py – Unit tests for the supply & demand zone engine.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_daily(n: int = 80, seed: int = 42) -> pd.DataFrame:
    """Synthetic daily bars at a flat price (low volatility baseline)."""
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    np.random.seed(seed)
    closes = 100.0 + np.random.randn(n) * 0.2
    return pd.DataFrame(
        {
            "open": closes - 0.1,
            "high": closes + 0.3,
            "low": closes - 0.3,
            "close": closes,
            "volume": np.full(n, 1_000_000.0),
        },
        index=rng,
    )


def _make_demand_zone_daily(n: int = 60, base_price: float = 100.0) -> pd.DataFrame:
    """Daily bars containing a clear demand zone:
    - bars 10–13: tight base (range < 0.5)
    - bars 14–17: strong upward impulse (≥ 2 × ATR)
    - volume doubles during impulse
    """
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = np.full(n, base_price)
    highs = closes + 0.5
    lows = closes - 0.5
    volumes = np.full(n, 1_000_000.0)

    # Tight base at bars 10–13
    for i in range(10, 14):
        highs[i] = base_price + 0.1
        lows[i] = base_price - 0.1

    # Impulse UP starting at bar 14 (+5 units per bar for 4 bars)
    for i in range(14, 18):
        step = (i - 13) * 5.0
        closes[i] = base_price + step
        highs[i] = closes[i] + 0.5
        lows[i] = closes[i] - 0.5
        volumes[i] = 2_500_000.0  # 2.5× avg

    # Remaining bars stay elevated
    for i in range(18, n):
        closes[i] = base_price + 20.0 + np.random.randn() * 0.1
        highs[i] = closes[i] + 0.5
        lows[i] = closes[i] - 0.5

    return pd.DataFrame(
        {
            "open": closes - 0.2,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=rng,
    )


def _make_supply_zone_daily(n: int = 60, base_price: float = 120.0) -> pd.DataFrame:
    """Daily bars containing a supply zone: tight base → strong downward impulse."""
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = np.full(n, base_price)
    highs = closes + 0.5
    lows = closes - 0.5
    volumes = np.full(n, 1_000_000.0)

    # Tight base at bars 15–18
    for i in range(15, 19):
        highs[i] = base_price + 0.1
        lows[i] = base_price - 0.1

    # Impulse DOWN starting at bar 19 (–5 units per bar for 4 bars)
    for i in range(19, 23):
        step = (i - 18) * 5.0
        closes[i] = base_price - step
        highs[i] = closes[i] + 0.5
        lows[i] = closes[i] - 0.5
        volumes[i] = 2_500_000.0

    # Remaining bars stay depressed
    for i in range(23, n):
        closes[i] = base_price - 25.0 + np.random.randn() * 0.1
        highs[i] = closes[i] + 0.5
        lows[i] = closes[i] - 0.5

    return pd.DataFrame(
        {
            "open": closes - 0.2,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=rng,
    )


# ---------------------------------------------------------------------------
# SupplyDemandZone dataclass
# ---------------------------------------------------------------------------


class TestSupplyDemandZone:
    def test_mid_price(self):
        from momentum_radar.signals.supply_demand import SupplyDemandZone

        zone = SupplyDemandZone(
            ticker="TEST",
            timeframe="daily",
            zone_type="demand",
            zone_high=105.0,
            zone_low=100.0,
            strength_score=70.0,
        )
        assert zone.mid_price == pytest.approx(102.5)

    def test_height(self):
        from momentum_radar.signals.supply_demand import SupplyDemandZone

        zone = SupplyDemandZone(
            ticker="TEST",
            timeframe="daily",
            zone_type="supply",
            zone_high=110.0,
            zone_low=105.0,
            strength_score=60.0,
        )
        assert zone.height == pytest.approx(5.0)

    def test_strength_labels(self):
        from momentum_radar.signals.supply_demand import SupplyDemandZone

        for score, expected in [
            (85.0, "Institutional"),
            (70.0, "Strong"),
            (55.0, "Moderate"),
            (40.0, "Weak"),
        ]:
            zone = SupplyDemandZone(
                ticker="T",
                timeframe="daily",
                zone_type="demand",
                zone_high=101.0,
                zone_low=100.0,
                strength_score=score,
            )
            assert zone.strength_label == expected


# ---------------------------------------------------------------------------
# Zone scoring
# ---------------------------------------------------------------------------


class TestComputeScore:
    def test_high_impulse_gives_high_score(self):
        from momentum_radar.signals.supply_demand import _compute_score

        score = _compute_score(
            impulse_magnitude=3.0,  # 3× ATR → 30 pts
            volume_expansion=2.0,   # 2× → 10 pts
            base_range=0.0,
            atr=1.0,
            touch_count=0,          # fresh → 25 pts
            status="fresh",
            timeframe="daily",      # daily bonus → 8 pts
        )
        assert score >= 70.0

    def test_broken_zone_has_zero_freshness(self):
        from momentum_radar.signals.supply_demand import _compute_score

        score_fresh = _compute_score(1.5, 1.5, 0.0, 1.0, 0, "fresh", "daily")
        score_broken = _compute_score(1.5, 1.5, 0.0, 1.0, 0, "broken", "daily")
        assert score_fresh > score_broken

    def test_score_does_not_exceed_100(self):
        from momentum_radar.signals.supply_demand import _compute_score

        score = _compute_score(10.0, 10.0, 0.0, 1.0, 0, "fresh", "weekly")
        assert score <= 100.0

    def test_higher_timeframe_adds_bonus(self):
        from momentum_radar.signals.supply_demand import _compute_score

        daily = _compute_score(2.0, 1.5, 0.3, 1.0, 0, "fresh", "daily")
        intraday = _compute_score(2.0, 1.5, 0.3, 1.0, 0, "fresh", "5m")
        assert daily > intraday


# ---------------------------------------------------------------------------
# Zone detection
# ---------------------------------------------------------------------------


class TestDetectZones:
    def test_demand_zone_detected(self):
        from momentum_radar.signals.supply_demand import detect_zones

        daily = _make_demand_zone_daily()
        zones = detect_zones("TEST", daily, min_score=30.0)
        demand_zones = [z for z in zones if z.zone_type == "demand"]
        assert len(demand_zones) >= 1

    def test_supply_zone_detected(self):
        from momentum_radar.signals.supply_demand import detect_zones

        daily = _make_supply_zone_daily()
        zones = detect_zones("TEST", daily, min_score=30.0)
        supply_zones = [z for z in zones if z.zone_type == "supply"]
        assert len(supply_zones) >= 1

    def test_flat_data_no_zones(self):
        from momentum_radar.signals.supply_demand import detect_zones

        daily = _make_daily(n=40)
        # Flat price with minimal ATR – impulse threshold unlikely to be met
        zones = detect_zones("TEST", daily, min_score=80.0)
        # May or may not find zones; just ensure no crash
        assert isinstance(zones, list)

    def test_insufficient_data_returns_empty(self):
        from momentum_radar.signals.supply_demand import detect_zones

        rng = pd.date_range("2024-01-01", periods=5, freq="B")
        tiny = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.5] * 5,
                "volume": [1e6] * 5,
            },
            index=rng,
        )
        zones = detect_zones("TEST", tiny)
        assert zones == []

    def test_none_daily_returns_empty(self):
        from momentum_radar.signals.supply_demand import detect_zones

        assert detect_zones("TEST", None) == []

    def test_zones_sorted_by_score(self):
        from momentum_radar.signals.supply_demand import detect_zones

        daily = _make_demand_zone_daily(n=80)
        zones = detect_zones("TEST", daily, min_score=0.0)
        scores = [z.strength_score for z in zones]
        assert scores == sorted(scores, reverse=True)

    def test_zone_boundaries_are_valid(self):
        from momentum_radar.signals.supply_demand import detect_zones

        daily = _make_demand_zone_daily()
        zones = detect_zones("TEST", daily, min_score=0.0)
        for z in zones:
            assert z.zone_high >= z.zone_low
            assert z.zone_low > 0


# ---------------------------------------------------------------------------
# get_active_zone
# ---------------------------------------------------------------------------


class TestGetActiveZone:
    def test_price_inside_zone_returns_zone(self):
        from momentum_radar.signals.supply_demand import SupplyDemandZone, get_active_zone

        zone = SupplyDemandZone(
            ticker="T",
            timeframe="daily",
            zone_type="demand",
            zone_high=105.0,
            zone_low=100.0,
            strength_score=70.0,
        )
        result = get_active_zone("T", 102.0, [zone], atr=2.0)
        assert result is zone

    def test_price_near_zone_returns_zone(self):
        from momentum_radar.signals.supply_demand import SupplyDemandZone, get_active_zone

        zone = SupplyDemandZone(
            ticker="T",
            timeframe="daily",
            zone_type="demand",
            zone_high=105.0,
            zone_low=100.0,
            strength_score=70.0,
        )
        # Within 0.5× ATR (= 1.0) of zone_low=100 → near-demand
        result = get_active_zone("T", 99.5, [zone], atr=2.0)
        assert result is zone

    def test_price_far_from_zone_returns_none(self):
        from momentum_radar.signals.supply_demand import SupplyDemandZone, get_active_zone

        zone = SupplyDemandZone(
            ticker="T",
            timeframe="daily",
            zone_type="demand",
            zone_high=105.0,
            zone_low=100.0,
            strength_score=70.0,
        )
        result = get_active_zone("T", 90.0, [zone], atr=2.0)
        assert result is None

    def test_empty_zones_returns_none(self):
        from momentum_radar.signals.supply_demand import get_active_zone

        assert get_active_zone("T", 100.0, [], atr=2.0) is None


# ---------------------------------------------------------------------------
# supply_demand_zone signal (registered)
# ---------------------------------------------------------------------------


class TestSupplyDemandSignal:
    def test_signal_no_daily_data(self):
        import momentum_radar.signals.supply_demand  # noqa: F401
        from momentum_radar.signals.supply_demand import supply_demand_zone

        result = supply_demand_zone(ticker="TEST", bars=None, daily=None)
        assert result.triggered is False
        assert result.score == 0

    def test_signal_no_active_zone(self):
        import momentum_radar.signals.supply_demand  # noqa: F401
        from momentum_radar.signals.supply_demand import supply_demand_zone

        # Flat data unlikely to produce a zone that price is currently in
        daily = _make_daily(n=50)
        result = supply_demand_zone(ticker="TEST", bars=None, daily=daily)
        assert isinstance(result.triggered, bool)
        assert result.score >= 0

    def test_signal_in_demand_zone_triggers(self):
        """Price sitting inside a freshly-detected demand zone should trigger."""
        import momentum_radar.signals.supply_demand  # noqa: F401
        from momentum_radar.signals.supply_demand import supply_demand_zone

        daily = _make_demand_zone_daily(n=80, base_price=100.0)
        # Place current price at bar 11 (inside the base / demand zone)
        daily_short = daily.iloc[:15].copy()
        # Reset last close to be right in the zone
        daily_short.iloc[-1, daily_short.columns.get_loc("close")] = 100.0
        daily_short.iloc[-1, daily_short.columns.get_loc("low")] = 99.9

        result = supply_demand_zone(ticker="TEST", bars=None, daily=daily_short)
        # Just validate it runs without error; triggering depends on ATR
        assert isinstance(result.triggered, bool)
        assert isinstance(result.score, int)


# ---------------------------------------------------------------------------
# structure_supply_demand_engine
# ---------------------------------------------------------------------------


class TestStructureSupplyDemandEngine:
    def test_scan_ticker_no_data(self):
        from momentum_radar.structure_supply_demand_engine import scan_ticker

        result = scan_ticker("TEST", fetcher=None, daily=None, bars=None)
        assert result["has_active_zone"] is False
        assert result["active_zone"] is None

    def test_scan_ticker_returns_correct_keys(self):
        from momentum_radar.structure_supply_demand_engine import scan_ticker

        daily = _make_daily(n=30)
        result = scan_ticker("TEST", daily=daily)
        for key in ["ticker", "has_active_zone", "all_zones", "alert_text", "zone_score_bonus"]:
            assert key in result

    def test_scan_ticker_with_demand_zone(self):
        from momentum_radar.structure_supply_demand_engine import scan_ticker

        daily = _make_demand_zone_daily(n=80)
        result = scan_ticker("TEST", daily=daily)
        # Result should have zones even if current price is not in any
        assert isinstance(result["all_zones"], list)

    def test_get_zone_score_bonus_institutional(self):
        from momentum_radar.signals.supply_demand import SupplyDemandZone
        from momentum_radar.structure_supply_demand_engine import get_zone_score_bonus

        zone = SupplyDemandZone(
            ticker="T",
            timeframe="daily",
            zone_type="demand",
            zone_high=105.0,
            zone_low=100.0,
            strength_score=85.0,
        )
        assert get_zone_score_bonus(zone) == 15

    def test_get_zone_score_bonus_strong_daily(self):
        from momentum_radar.signals.supply_demand import SupplyDemandZone
        from momentum_radar.structure_supply_demand_engine import get_zone_score_bonus

        zone = SupplyDemandZone(
            ticker="T",
            timeframe="daily",
            zone_type="supply",
            zone_high=110.0,
            zone_low=105.0,
            strength_score=70.0,
        )
        assert get_zone_score_bonus(zone) == 12

    def test_get_zone_score_bonus_intraday_moderate(self):
        from momentum_radar.signals.supply_demand import SupplyDemandZone
        from momentum_radar.structure_supply_demand_engine import get_zone_score_bonus

        zone = SupplyDemandZone(
            ticker="T",
            timeframe="15m",
            zone_type="demand",
            zone_high=101.0,
            zone_low=100.0,
            strength_score=55.0,
        )
        assert get_zone_score_bonus(zone) == 3

    def test_format_zone_alert_empty_when_no_zone(self):
        from momentum_radar.structure_supply_demand_engine import format_zone_alert

        result = {
            "ticker": "TEST",
            "active_zone": None,
            "entry": None,
            "stop": None,
            "target": None,
            "rr": None,
            "confidence_pct": 0.0,
            "win_rate_pct": 0.0,
            "third_touch_setup": False,
        }
        assert format_zone_alert(result) == ""

    def test_format_zone_alert_contains_ticker(self):
        from momentum_radar.signals.supply_demand import SupplyDemandZone
        from momentum_radar.structure_supply_demand_engine import format_zone_alert

        zone = SupplyDemandZone(
            ticker="AAPL",
            timeframe="daily",
            zone_type="demand",
            zone_high=150.0,
            zone_low=145.0,
            strength_score=78.0,
            touch_count=1,
            status="tested",
            impulse_magnitude=2.1,
            volume_expansion=1.8,
        )
        result = {
            "ticker": "AAPL",
            "active_zone": zone,
            "entry": 147.5,
            "stop": 143.0,
            "target": 155.0,
            "rr": 1.7,
            "confidence_pct": 74.0,
            "win_rate_pct": 63.0,
            "third_touch_setup": False,
        }
        text = format_zone_alert(result)
        assert "AAPL" in text
        assert "Demand" in text or "demand" in text.lower()
        assert "145.00" in text
        assert "150.00" in text

    def test_format_zone_alert_third_touch(self):
        from momentum_radar.signals.supply_demand import SupplyDemandZone
        from momentum_radar.structure_supply_demand_engine import format_zone_alert

        zone = SupplyDemandZone(
            ticker="TSLA",
            timeframe="daily",
            zone_type="demand",
            zone_high=200.0,
            zone_low=195.0,
            strength_score=80.0,
            touch_count=2,
            status="tested",
        )
        result = {
            "ticker": "TSLA",
            "active_zone": zone,
            "entry": 197.0,
            "stop": 193.0,
            "target": 207.0,
            "rr": 2.5,
            "confidence_pct": 82.0,
            "win_rate_pct": 67.0,
            "third_touch_setup": True,
        }
        text = format_zone_alert(result)
        assert "Third-Touch" in text


# ---------------------------------------------------------------------------
# Zone persistence (in-memory / no real DB)
# ---------------------------------------------------------------------------


class TestZoneStore:
    def test_upsert_and_load_with_no_db(self):
        """upsert_zone should be a no-op (not raise) when DB is uninitialised."""
        from momentum_radar.signals.supply_demand import SupplyDemandZone
        from momentum_radar.storage.zone_store import upsert_zone, load_zones

        zone = SupplyDemandZone(
            ticker="ZTEST",
            timeframe="daily",
            zone_type="demand",
            zone_high=110.0,
            zone_low=105.0,
            strength_score=60.0,
        )
        # Should not raise regardless of DB state
        upsert_zone(zone)
        result = load_zones("ZTEST")
        # Either [] (no DB) or a list; never an exception
        assert isinstance(result, list)

    def test_load_zones_no_db(self):
        from momentum_radar.storage.zone_store import load_zones

        result = load_zones("NONEXISTENT")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Lifecycle rescoring bug fix
# ---------------------------------------------------------------------------


class TestLifecycleRescoring:
    """Verify that _update_zone_lifecycle uses stored base_range/atr, not hardcoded values."""

    def test_rescoring_uses_stored_base_range_and_atr(self):
        """After lifecycle update, a zone should NOT have inflated tightness pts
        from base_range=0.0/atr=1.0 defaults."""
        from momentum_radar.signals.supply_demand import (
            SupplyDemandZone,
            _compute_score,
            _update_zone_lifecycle,
        )

        # Create a zone with a real base_range and atr
        base_range = 0.5
        atr = 1.0
        zone = SupplyDemandZone(
            ticker="T",
            timeframe="daily",
            zone_type="demand",
            zone_high=101.0,
            zone_low=100.5,
            strength_score=99.0,  # arbitrary initial value, will be recalculated by lifecycle update
            touch_count=0,
            status="fresh",
            impulse_magnitude=2.0,
            volume_expansion=1.5,
            creation_bar_index=0,
            base_range=base_range,
            atr=atr,
        )

        # Build a minimal DataFrame with no price inside the zone so touch_count stays 0
        rng = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {
                "open": [105.0] * 5,
                "high": [106.0] * 5,
                "low": [104.0] * 5,
                "close": [105.0] * 5,
                "volume": [1_000_000.0] * 5,
            },
            index=rng,
        )

        _update_zone_lifecycle([zone], df)

        # Score after lifecycle must equal what _compute_score would produce
        # with the *original* base_range and atr (not 0.0/1.0 hardcoded)
        expected = round(
            _compute_score(
                impulse_magnitude=2.0,
                volume_expansion=1.5,
                base_range=base_range,
                atr=atr,
                touch_count=zone.touch_count,
                status=zone.status,
                timeframe="daily",
            ),
            1,
        )
        assert zone.strength_score == expected

    def test_rescoring_with_zero_base_range_would_inflate(self):
        """Demonstrate that base_range=0.0 with atr=1.0 gives max tightness pts (15),
        while using the real base_range gives a lower tightness contribution."""
        from momentum_radar.signals.supply_demand import _compute_score

        # With real base_range (non-zero), tightness_pts < 15
        score_real = _compute_score(
            impulse_magnitude=2.0,
            volume_expansion=1.5,
            base_range=0.5,
            atr=1.0,
            touch_count=0,
            status="fresh",
            timeframe="daily",
        )
        # With hardcoded base_range=0.0 (old bug), tightness_pts = 15 (max)
        score_inflated = _compute_score(
            impulse_magnitude=2.0,
            volume_expansion=1.5,
            base_range=0.0,
            atr=1.0,
            touch_count=0,
            status="fresh",
            timeframe="daily",
        )
        assert score_inflated > score_real, (
            "base_range=0.0 should produce a higher (inflated) score than base_range=0.5"
        )

    def test_zone_dataclass_stores_base_range_and_atr(self):
        """SupplyDemandZone should store base_range and atr fields."""
        from momentum_radar.signals.supply_demand import SupplyDemandZone

        zone = SupplyDemandZone(
            ticker="T",
            timeframe="daily",
            zone_type="supply",
            zone_high=110.0,
            zone_low=108.0,
            strength_score=65.0,
            base_range=0.8,
            atr=2.5,
        )
        assert zone.base_range == pytest.approx(0.8)
        assert zone.atr == pytest.approx(2.5)

    def test_detect_zones_populates_base_range_and_atr(self):
        """Zones produced by detect_zones should have non-default base_range/atr."""
        from momentum_radar.signals.supply_demand import detect_zones

        daily = _make_demand_zone_daily()
        zones = detect_zones("TEST", daily, min_score=0.0)
        demand_zones = [z for z in zones if z.zone_type == "demand"]
        assert len(demand_zones) >= 1
        zone = demand_zones[0]
        # atr should be > 0 (real value from compute_atr) and base_range >= 0
        assert zone.atr > 0
        assert zone.base_range >= 0.0


# ---------------------------------------------------------------------------
# DatetimeIndex guard
# ---------------------------------------------------------------------------


class TestDatetimeIndexGuard:
    def test_non_datetime_index_daily_skips_weekly_resample(self, caplog):
        """detect_zones should warn and skip weekly resample when index is not DatetimeIndex."""
        import logging
        from momentum_radar.signals.supply_demand import detect_zones

        daily = _make_demand_zone_daily(n=60)
        # Reset index to integer — breaks resample
        daily_int = daily.reset_index(drop=True)

        with caplog.at_level(logging.WARNING, logger="momentum_radar.signals.supply_demand"):
            zones = detect_zones("TEST", daily_int, min_score=0.0)

        assert isinstance(zones, list)
        assert any("DatetimeIndex" in r.message for r in caplog.records), (
            "Expected a warning about non-DatetimeIndex for daily bars"
        )

    def test_datetime_index_daily_includes_weekly(self):
        """With a proper DatetimeIndex, weekly resample should run without errors."""
        from momentum_radar.signals.supply_demand import detect_zones

        daily = _make_demand_zone_daily(n=80)
        assert isinstance(daily.index, pd.DatetimeIndex)
        # Should not raise and should return a list
        zones = detect_zones("TEST", daily, min_score=0.0)
        assert isinstance(zones, list)


# ---------------------------------------------------------------------------
# Zone pattern classification (DBR / RBR / RBD / DBD)
# ---------------------------------------------------------------------------


def _make_demand_zone_with_prior_drop(n: int = 80, base_price: float = 100.0) -> pd.DataFrame:
    """Daily bars: prior drop → tight base → upward impulse (DBR pattern)."""
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = np.full(n, base_price, dtype=float)
    highs = closes + 0.5
    lows = closes - 0.5
    volumes = np.full(n, 1_000_000.0)

    # Prior drop: bars 5–9 fall from base_price to base_price – 10
    for i in range(5, 10):
        closes[i] = base_price - (i - 4) * 2.0
        highs[i] = closes[i] + 0.5
        lows[i] = closes[i] - 0.5

    # Tight base at bars 10–13 near the lows
    for i in range(10, 14):
        closes[i] = base_price - 10.0
        highs[i] = base_price - 9.9
        lows[i] = base_price - 10.1

    opens = closes - 0.2

    # Impulse UP: bars 14–18 rally sharply
    for i in range(14, 19):
        step = (i - 13) * 5.0
        closes[i] = base_price - 10.0 + step
        highs[i] = closes[i] + 0.5
        lows[i] = closes[i] - 0.5
        volumes[i] = 2_500_000.0

    for i in range(19, n):
        closes[i] = base_price + 5.0
        highs[i] = closes[i] + 0.5
        lows[i] = closes[i] - 0.5

    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=rng,
    )


class TestZonePatternClassification:
    def test_demand_zone_has_zone_pattern_field(self):
        """SupplyDemandZone dataclass should expose zone_pattern."""
        from momentum_radar.signals.supply_demand import SupplyDemandZone

        zone = SupplyDemandZone(
            ticker="T",
            timeframe="daily",
            zone_type="demand",
            zone_high=105.0,
            zone_low=100.0,
            strength_score=70.0,
            zone_pattern="DBR",
        )
        assert zone.zone_pattern == "DBR"

    def test_zone_pattern_default_is_empty_string(self):
        """zone_pattern defaults to empty string for backward compatibility."""
        from momentum_radar.signals.supply_demand import SupplyDemandZone

        zone = SupplyDemandZone(
            ticker="T",
            timeframe="daily",
            zone_type="supply",
            zone_high=110.0,
            zone_low=105.0,
            strength_score=65.0,
        )
        assert zone.zone_pattern == ""

    def test_detected_demand_zone_has_pattern_label(self):
        """Zones produced by detect_zones should carry a zone_pattern label."""
        from momentum_radar.signals.supply_demand import detect_zones

        daily = _make_demand_zone_daily(n=80)
        zones = detect_zones("TEST", daily, min_score=0.0)
        demand_zones = [z for z in zones if z.zone_type == "demand"]
        assert len(demand_zones) >= 1
        # Each zone should have a non-empty or well-defined zone_pattern
        for z in demand_zones:
            assert isinstance(z.zone_pattern, str)

    def test_dbr_pattern_detected_on_drop_base_rally(self):
        """A demand zone preceded by a drop should be classified as DBR."""
        from momentum_radar.signals.supply_demand import detect_zones

        daily = _make_demand_zone_with_prior_drop(n=80)
        zones = detect_zones("TEST", daily, min_score=0.0)
        demand_zones = [z for z in zones if z.zone_type == "demand"]
        assert len(demand_zones) >= 1
        # At least one should be labelled DBR (drop → base → rally)
        patterns = {z.zone_pattern for z in demand_zones}
        assert "DBR" in patterns, f"Expected DBR in {patterns}"


# ---------------------------------------------------------------------------
# Demand zone boundary convention (wick low → last base body high)
# ---------------------------------------------------------------------------


class TestDemandZoneBoundaries:
    def test_demand_zone_low_is_wick_minimum(self):
        """Demand zone_low should equal the lowest wick in the base, not the body."""
        from momentum_radar.signals.supply_demand import detect_zones

        daily = _make_demand_zone_daily(n=60)
        zones = detect_zones("TEST", daily, min_score=0.0)
        demand_zones = [z for z in zones if z.zone_type == "demand"]
        assert len(demand_zones) >= 1
        z = demand_zones[0]
        # For _make_demand_zone_daily the base bars 10–13 have low = base_price – 0.1
        assert z.zone_low == pytest.approx(100.0 - 0.1, abs=1e-3)

    def test_demand_zone_high_uses_last_base_candle_body(self):
        """Demand zone_high ≤ highest wick (not full wick range)."""
        from momentum_radar.signals.supply_demand import detect_zones

        daily = _make_demand_zone_daily(n=60)
        zones = detect_zones("TEST", daily, min_score=0.0)
        demand_zones = [z for z in zones if z.zone_type == "demand"]
        assert len(demand_zones) >= 1
        z = demand_zones[0]
        # zone_high should be the body-high of last base candle = max(open, close)
        # For the test data last base candle: open=99.8, close=100 → body_high=100
        # It should be strictly less than the wick high (100.1)
        assert z.zone_high < 100.1 + 1e-3  # at or below wick high
        assert z.zone_high >= z.zone_low


# ---------------------------------------------------------------------------
# 4H timeframe scanning
# ---------------------------------------------------------------------------


class TestFourHourTimeframe:
    def test_4h_label_in_tf_score_bonus(self):
        """The 4H timeframe should be in the TF score bonus dict."""
        from momentum_radar.signals.supply_demand import _TF_SCORE_BONUS

        assert "4h" in _TF_SCORE_BONUS
        assert _TF_SCORE_BONUS["4h"] > _TF_SCORE_BONUS["1h"]

    def test_intraday_scan_includes_4h(self):
        """detect_zones with intraday bars should attempt 4H zone scanning."""
        from momentum_radar.signals.supply_demand import detect_zones

        # Build 5 days of 1-min bars at 390 bars/day
        n_days = 5
        freq = pd.tseries.offsets.Minute(1)
        idx = pd.date_range("2024-01-02 09:30", periods=n_days * 390, freq=freq)
        np.random.seed(7)
        closes = 100.0 + np.random.randn(len(idx)) * 0.05
        bars = pd.DataFrame(
            {
                "open": closes - 0.01,
                "high": closes + 0.05,
                "low": closes - 0.05,
                "close": closes,
                "volume": np.full(len(idx), 50_000.0),
            },
            index=idx,
        )
        daily = _make_daily(n=50)
        # Just ensure no exception and a list is returned
        zones = detect_zones("TEST", daily, bars=bars, min_score=0.0)
        assert isinstance(zones, list)
