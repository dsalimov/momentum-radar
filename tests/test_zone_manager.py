"""
tests/test_zone_manager.py – Tests for the ZoneManager class.
"""

import pytest

from momentum_radar.signals.supply_demand import SupplyDemandZone


def _make_zone(zone_type="demand", score=80.0, status="fresh", low=100.0, high=105.0):
    return SupplyDemandZone(
        ticker="TEST",
        timeframe="daily",
        zone_type=zone_type,
        zone_high=high,
        zone_low=low,
        strength_score=score,
        touch_count=0,
        status=status,
    )


class TestZoneManagerBasics:
    def test_update_zones_stores_filtered(self):
        from momentum_radar.signals.zone_manager import ZoneManager

        mgr = ZoneManager(min_score=70.0)
        zones = [
            _make_zone(score=80.0),
            _make_zone(score=50.0),  # below min_score
            _make_zone(score=90.0, status="broken"),  # broken → excluded
        ]
        mgr.update_zones("TEST", zones)
        stored = mgr.get_zones("TEST")
        assert len(stored) == 1
        assert stored[0].strength_score == 80.0

    def test_zones_sorted_by_score_desc(self):
        from momentum_radar.signals.zone_manager import ZoneManager

        mgr = ZoneManager(min_score=0.0)
        zones = [
            _make_zone(score=60.0),
            _make_zone(score=90.0),
            _make_zone(score=75.0),
        ]
        mgr.update_zones("TEST", zones)
        scores = [z.strength_score for z in mgr.get_zones("TEST")]
        assert scores == sorted(scores, reverse=True)

    def test_clear_removes_all_zones(self):
        from momentum_radar.signals.zone_manager import ZoneManager

        mgr = ZoneManager(min_score=0.0)
        mgr.update_zones("TEST", [_make_zone(score=80.0)])
        mgr.clear("TEST")
        assert mgr.get_zones("TEST") == []

    def test_unknown_ticker_returns_empty(self):
        from momentum_radar.signals.zone_manager import ZoneManager

        mgr = ZoneManager()
        assert mgr.get_zones("NONEXISTENT") == []

    def test_zone_count(self):
        from momentum_radar.signals.zone_manager import ZoneManager

        mgr = ZoneManager(min_score=0.0)
        mgr.update_zones("AAPL", [_make_zone(score=80.0), _make_zone(score=75.0)])
        mgr.update_zones("MSFT", [_make_zone(score=85.0)])
        assert mgr.zone_count() == 3
        assert mgr.zone_count("AAPL") == 2
        assert mgr.zone_count("MSFT") == 1

    def test_ticker_count(self):
        from momentum_radar.signals.zone_manager import ZoneManager

        mgr = ZoneManager(min_score=0.0)
        mgr.update_zones("A", [_make_zone()])
        mgr.update_zones("B", [_make_zone()])
        assert mgr.ticker_count() == 2


class TestZoneManagerInvalidate:
    def test_invalidate_zone_removes_it(self):
        from momentum_radar.signals.zone_manager import ZoneManager

        mgr = ZoneManager(min_score=0.0)
        zone = _make_zone(score=80.0)
        mgr.update_zones("TEST", [zone])
        mgr.invalidate_zone("TEST", zone)
        assert zone not in mgr.get_zones("TEST")
        assert zone.status == "broken"

    def test_invalidate_broken_removes_broken_zones(self):
        from momentum_radar.signals.zone_manager import ZoneManager

        mgr = ZoneManager(min_score=0.0)
        good = _make_zone(score=80.0, status="fresh")
        bad = _make_zone(score=75.0, status="broken")
        # Force broken zone past filter by inserting directly
        mgr._zones["TEST"] = [good, bad]
        removed = mgr.invalidate_broken("TEST")
        assert removed == 1
        assert good in mgr.get_zones("TEST")


class TestZoneManagerGetActiveZone:
    def test_price_inside_zone_returns_zone(self):
        from momentum_radar.signals.zone_manager import ZoneManager

        mgr = ZoneManager(min_score=0.0)
        zone = _make_zone(low=100.0, high=105.0)
        mgr.update_zones("TEST", [zone])
        result = mgr.get_active_zone("TEST", current_price=102.0, atr=2.0)
        assert result is zone

    def test_price_near_zone_within_proximity(self):
        from momentum_radar.signals.zone_manager import ZoneManager

        mgr = ZoneManager(min_score=0.0)
        zone = _make_zone(low=100.0, high=105.0)
        mgr.update_zones("TEST", [zone])
        # 0.5 × ATR(2.0) = 1.0 → 99.0 should be within range
        result = mgr.get_active_zone("TEST", current_price=99.5, atr=2.0)
        assert result is zone

    def test_price_far_returns_none(self):
        from momentum_radar.signals.zone_manager import ZoneManager

        mgr = ZoneManager(min_score=0.0)
        zone = _make_zone(low=100.0, high=105.0)
        mgr.update_zones("TEST", [zone])
        result = mgr.get_active_zone("TEST", current_price=90.0, atr=2.0)
        assert result is None

    def test_empty_registry_returns_none(self):
        from momentum_radar.signals.zone_manager import ZoneManager

        mgr = ZoneManager()
        assert mgr.get_active_zone("TEST", 100.0, 2.0) is None

    def test_strongest_zone_returned_first(self):
        from momentum_radar.signals.zone_manager import ZoneManager

        mgr = ZoneManager(min_score=0.0)
        weak = _make_zone(score=70.0, low=100.0, high=106.0)
        strong = _make_zone(score=90.0, low=100.0, high=106.0)
        mgr.update_zones("TEST", [weak, strong])
        result = mgr.get_active_zone("TEST", current_price=103.0, atr=2.0)
        assert result is strong
