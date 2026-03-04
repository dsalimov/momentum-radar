"""
tests/test_zone_scoring.py – Tests for the zone_scoring module.
"""

import pytest


class TestScoreZone:
    def test_high_impulse_high_volume_bos_daily_scores_above_minimum(self):
        from momentum_radar.signals.zone_scoring import score_zone, ZONE_MIN_SCORE

        score = score_zone(
            impulse_ratio=2.5,
            volume_ratio=2.0,
            base_candle_count=4,
            bos_confirmed=True,
            timeframe="daily",
        )
        assert score >= ZONE_MIN_SCORE

    def test_no_bos_reduces_score(self):
        from momentum_radar.signals.zone_scoring import score_zone

        with_bos = score_zone(2.0, 1.8, 4, True, "daily")
        without_bos = score_zone(2.0, 1.8, 4, False, "daily")
        assert with_bos > without_bos

    def test_higher_timeframe_gives_higher_score(self):
        from momentum_radar.signals.zone_scoring import score_zone

        daily_score = score_zone(2.0, 1.8, 4, True, "daily")
        intraday_score = score_zone(2.0, 1.8, 4, True, "5m")
        assert daily_score > intraday_score

    def test_score_capped_at_100(self):
        from momentum_radar.signals.zone_scoring import score_zone

        score = score_zone(10.0, 10.0, 4, True, "weekly")
        assert score <= 100.0

    def test_zero_impulse_low_score(self):
        from momentum_radar.signals.zone_scoring import score_zone

        score = score_zone(0.0, 1.0, 4, False, "5m")
        assert score < 30.0

    def test_optimal_base_candle_count_gives_full_base_pts(self):
        from momentum_radar.signals.zone_scoring import score_zone

        # 3 candles is in the 2-6 optimal range
        score_3 = score_zone(2.0, 2.0, 3, True, "daily")
        score_8 = score_zone(2.0, 2.0, 8, True, "daily")
        assert score_3 > score_8

    def test_weekly_alignment_bonus_highest(self):
        from momentum_radar.signals.zone_scoring import score_zone

        weekly = score_zone(2.0, 2.0, 4, True, "weekly")
        daily = score_zone(2.0, 2.0, 4, True, "daily")
        assert weekly > daily


class TestIsDisplacement:
    def test_strong_candle_is_displacement(self):
        from momentum_radar.signals.zone_scoring import is_displacement

        assert is_displacement(
            candle_range=3.0,
            body_ratio=0.80,
            candle_volume=2_000_000,
            avg_range=1.5,
            avg_volume=1_000_000,
        )

    def test_weak_body_not_displacement(self):
        from momentum_radar.signals.zone_scoring import is_displacement

        assert not is_displacement(
            candle_range=3.0,
            body_ratio=0.40,  # below 0.70 threshold
            candle_volume=2_000_000,
            avg_range=1.5,
            avg_volume=1_000_000,
        )

    def test_low_volume_not_displacement(self):
        from momentum_radar.signals.zone_scoring import is_displacement

        assert not is_displacement(
            candle_range=3.0,
            body_ratio=0.80,
            candle_volume=500_000,  # < 1.5× avg
            avg_range=1.5,
            avg_volume=1_000_000,
        )

    def test_small_range_not_displacement(self):
        from momentum_radar.signals.zone_scoring import is_displacement

        assert not is_displacement(
            candle_range=1.0,  # < 1.5 × avg_range
            body_ratio=0.80,
            candle_volume=2_000_000,
            avg_range=1.5,
            avg_volume=1_000_000,
        )

    def test_zero_avg_range_returns_false(self):
        from momentum_radar.signals.zone_scoring import is_displacement

        assert not is_displacement(3.0, 0.80, 2_000_000, 0.0, 1_000_000)


class TestIsBase:
    def test_tight_2_to_6_candles_is_base(self):
        from momentum_radar.signals.zone_scoring import is_base

        ranges = [0.5, 0.4, 0.6, 0.5]  # 4 candles, all < 1.2 × avg_range=1.0
        assert is_base(ranges, avg_range=1.0)

    def test_too_few_candles_not_base(self):
        from momentum_radar.signals.zone_scoring import is_base

        assert not is_base([0.5], avg_range=1.0)

    def test_too_many_candles_not_base(self):
        from momentum_radar.signals.zone_scoring import is_base

        assert not is_base([0.5] * 7, avg_range=1.0)

    def test_wide_range_candle_not_base(self):
        from momentum_radar.signals.zone_scoring import is_base

        ranges = [0.5, 0.4, 2.5, 0.5]  # one wide candle
        assert not is_base(ranges, avg_range=1.0)

    def test_empty_list_not_base(self):
        from momentum_radar.signals.zone_scoring import is_base

        assert not is_base([], avg_range=1.0)
