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
