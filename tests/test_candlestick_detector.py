"""
test_candlestick_detector.py - Unit tests for candlestick pattern detection.
"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime

from momentum_radar.patterns.candlestick_detector import (
    detect_candlestick_pattern,
    CANDLESTICK_PATTERNS,
    _candle_props,
    _trend_context,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(rows: list) -> pd.DataFrame:
    """Build a OHLCV DataFrame from a list of (open, high, low, close) tuples."""
    rng = pd.date_range("2024-01-02", periods=len(rows), freq="B")
    opens, highs, lows, closes = zip(*rows)
    return pd.DataFrame(
        {
            "open": list(opens),
            "high": list(highs),
            "low": list(lows),
            "close": list(closes),
            "volume": [1_000_000.0] * len(rows),
        },
        index=rng,
    )


def _downtrend_prefix(n: int = 6, start: float = 110.0) -> list:
    """Return n OHLCV rows trending down by 3% total (satisfies downtrend context)."""
    closes = np.linspace(start, start * (1 - 0.03), n)
    return [(c - 0.3, c + 0.3, c - 0.5, c) for c in closes]


def _uptrend_prefix(n: int = 6, start: float = 90.0) -> list:
    """Return n OHLCV rows trending up by 3% total (satisfies uptrend context)."""
    closes = np.linspace(start, start * 1.03, n)
    return [(c - 0.3, c + 0.3, c - 0.5, c) for c in closes]


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


def test_candle_props_bullish():
    body, upper, lower, total, ratio, is_bull, is_bear = _candle_props(100, 106, 98, 104)
    assert body == pytest.approx(4.0)
    assert upper == pytest.approx(2.0)  # 106 - max(100, 104) = 106 - 104
    assert lower == pytest.approx(2.0)  # min(100, 104) - 98 = 100 - 98
    assert total == pytest.approx(8.0)
    assert ratio == pytest.approx(0.5)
    assert is_bull is True
    assert is_bear is False


def test_candle_props_bearish():
    body, upper, lower, total, ratio, is_bull, is_bear = _candle_props(104, 106, 98, 100)
    assert body == pytest.approx(4.0)
    assert upper == pytest.approx(2.0)  # 106 - max(104, 100) = 106 - 104
    assert lower == pytest.approx(2.0)  # min(104, 100) - 98 = 100 - 98
    assert is_bull is False
    assert is_bear is True


def test_trend_context_downtrend():
    closes = np.array([110.0, 109.0, 108.0, 107.5, 107.0, 106.5, 106.0])
    assert _trend_context(closes, len(closes)) == "downtrend"


def test_trend_context_uptrend():
    closes = np.array([90.0, 91.0, 92.0, 92.5, 93.0, 93.5, 94.0])
    assert _trend_context(closes, len(closes)) == "uptrend"


def test_trend_context_neutral():
    closes = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    assert _trend_context(closes, len(closes)) == "neutral"


def test_trend_context_too_short():
    closes = np.array([100.0, 99.0])
    assert _trend_context(closes, len(closes)) == "neutral"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_candlestick_patterns_registry_complete():
    expected = {
        "hammer", "inverted hammer", "hanging man", "shooting star",
        "doji", "dragonfly doji", "gravestone doji",
        "bullish marubozu", "bearish marubozu", "spinning top",
        "bullish engulfing", "bearish engulfing",
        "bullish harami", "bearish harami",
        "tweezer top", "tweezer bottom",
        "piercing line", "dark cloud cover",
        "morning star", "evening star",
        "three white soldiers", "three black crows",
        "three inside up", "three inside down",
    }
    assert set(CANDLESTICK_PATTERNS.keys()) == expected


def test_detect_unknown_pattern_returns_none():
    df = _make_df(_downtrend_prefix(10))
    assert detect_candlestick_pattern("nonexistent pattern", df) is None


def test_detect_with_empty_df_returns_none():
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    assert detect_candlestick_pattern("hammer", df) is None


# ---------------------------------------------------------------------------
# Single candle patterns
# ---------------------------------------------------------------------------


def test_hammer_detected_after_downtrend():
    prefix = _downtrend_prefix(6, start=110.0)
    # Hammer: small body near top, long lower shadow (2x+ body), tiny upper shadow
    # o=104, h=104.2, l=100, c=104 => body=0, shadow_ratio=infinity - skip body=0
    # Let's use: o=104, h=104.3, l=100, c=104.2 => body=0.2, lower=4, upper=0.1
    hammer_candle = [(104.0, 104.3, 100.0, 104.2)]
    df = _make_df(prefix + hammer_candle)
    result = detect_candlestick_pattern("hammer", df)
    assert result is not None
    assert result["pattern"] == "Hammer"
    assert result["bias"] == "bullish"
    assert result["pattern_type"] == "candlestick"
    assert result["confidence"] >= 60
    assert result["state"] == "forming"
    assert len(result["candle_indices"]) == 1


def test_hammer_not_detected_in_uptrend():
    prefix = _uptrend_prefix(6, start=90.0)
    hammer_candle = [(94.0, 94.3, 90.0, 94.2)]
    df = _make_df(prefix + hammer_candle)
    result = detect_candlestick_pattern("hammer", df)
    assert result is None


def test_shooting_star_detected_after_uptrend():
    prefix = _uptrend_prefix(6, start=90.0)
    # Shooting star: open near low, long upper shadow, tiny lower shadow
    # o=93.8, h=98, l=93.7, c=94.0 => body=0.2, upper=4, lower=0.1
    star_candle = [(93.8, 98.0, 93.7, 94.0)]
    df = _make_df(prefix + star_candle)
    result = detect_candlestick_pattern("shooting star", df)
    assert result is not None
    assert result["pattern"] == "Shooting Star"
    assert result["bias"] == "bearish"


def test_hanging_man_detected_after_uptrend():
    prefix = _uptrend_prefix(6, start=90.0)
    # Hanging man: same shape as hammer but after uptrend
    # o=94, h=94.3, l=90, c=94.2 => body=0.2, lower=4, upper=0.1
    hm_candle = [(94.0, 94.3, 90.0, 94.2)]
    df = _make_df(prefix + hm_candle)
    result = detect_candlestick_pattern("hanging man", df)
    assert result is not None
    assert result["pattern"] == "Hanging Man"
    assert result["bias"] == "bearish"


def test_inverted_hammer_detected_after_downtrend():
    prefix = _downtrend_prefix(6, start=110.0)
    # Inverted hammer: small body at bottom, long upper shadow
    # o=104, h=108, l=103.9, c=104.2 => body=0.2, upper=3.8, lower=0.1
    ih_candle = [(104.0, 108.0, 103.9, 104.2)]
    df = _make_df(prefix + ih_candle)
    result = detect_candlestick_pattern("inverted hammer", df)
    assert result is not None
    assert result["pattern"] == "Inverted Hammer"
    assert result["bias"] == "bullish"


def test_doji_detected():
    prefix = _downtrend_prefix(6)
    # Doji: open ≈ close (body < 10% of range)
    # o=100, h=103, l=97, c=100.2 => body=0.2, range=6, ratio=3.3% < 10%
    doji_candle = [(100.0, 103.0, 97.0, 100.2)]
    df = _make_df(prefix + doji_candle)
    result = detect_candlestick_pattern("doji", df)
    assert result is not None
    assert result["pattern_type"] == "candlestick"
    assert result["pattern"] in ("Doji", "Dragonfly Doji", "Gravestone Doji")


def test_bullish_marubozu_detected():
    prefix = _downtrend_prefix(6)
    # Bullish marubozu: large bullish candle, no wicks
    # o=100, h=105.1, l=99.9, c=105 => body=5, upper=0.1, lower=0.1, range=5.2
    # upper+lower = 0.2, range=5.2, each shadow < 5% of range ✓
    marubozu_candle = [(100.0, 105.1, 99.9, 105.0)]
    df = _make_df(prefix + marubozu_candle)
    result = detect_candlestick_pattern("bullish marubozu", df)
    assert result is not None
    assert result["pattern"] == "Bullish Marubozu"
    assert result["bias"] == "bullish"


def test_spinning_top_detected():
    prefix = _downtrend_prefix(6)
    # Spinning top: small body (10-35% of range), both shadows >= 20% of range
    # o=100, h=103, l=97, c=100.8 => body=0.8, range=6, ratio=13%
    # upper=3-100.8=2.2, lower=100-97=3 => both >= 20% of 6 ✓
    spin_candle = [(100.0, 103.0, 97.0, 100.8)]
    df = _make_df(prefix + spin_candle)
    result = detect_candlestick_pattern("spinning top", df)
    assert result is not None
    assert result["pattern"] == "Spinning Top"
    assert result["bias"] == "neutral"


# ---------------------------------------------------------------------------
# Two-candle patterns
# ---------------------------------------------------------------------------


def test_bullish_engulfing_detected():
    prefix = _downtrend_prefix(6, start=110.0)
    # Candle 1 bearish: o=104, h=105, l=102, c=102.5 (body=1.5)
    # Candle 2 bullish: o=101.5, h=106, l=101, c=105.5 (body=4) engulfs C1
    # C2 opens below C1 close (101.5 < 102.5) ✓
    # C2 closes above C1 open (105.5 > 104) ✓
    candle1 = (104.0, 105.0, 102.0, 102.5)
    candle2 = (101.5, 106.0, 101.0, 105.5)
    df = _make_df(prefix + [candle1, candle2])
    result = detect_candlestick_pattern("bullish engulfing", df)
    assert result is not None
    assert result["pattern"] == "Bullish Engulfing"
    assert result["bias"] == "bullish"
    assert len(result["candle_indices"]) == 2


def test_bearish_engulfing_detected():
    # Use start=84 so close[-6] is low enough that even after the bearish
    # engulfing's lower close, the 5-bar % change stays >= +2% (uptrend).
    prefix = _uptrend_prefix(6, start=84.0)  # 84 → 86.52, close[2] ≈ 85.01
    # close[-1] = 88 > 85.01 * 1.02 = 86.71  ✓  → uptrend context passes
    # Candle 1 bullish: o=90, c=90.5 (body=0.5)
    # Candle 2 bearish: opens above c1=90.5 (91.5>90.5), closes below o1=90 (88<90)
    candle1 = (90.0, 91.0, 89.5, 90.5)
    candle2 = (91.5, 92.0, 87.5, 88.0)
    df = _make_df(prefix + [candle1, candle2])
    result = detect_candlestick_pattern("bearish engulfing", df)
    assert result is not None
    assert result["pattern"] == "Bearish Engulfing"
    assert result["bias"] == "bearish"


def test_bullish_harami_detected():
    prefix = _downtrend_prefix(6, start=110.0)
    # Candle 1 large bearish: o=106, h=107, l=101, c=102 (body=4)
    # Candle 2 small bullish: o=103, h=104, l=102.5, c=103.5 (body=0.5, contained)
    candle1 = (106.0, 107.0, 101.0, 102.0)
    candle2 = (103.0, 104.0, 102.5, 103.5)
    df = _make_df(prefix + [candle1, candle2])
    result = detect_candlestick_pattern("bullish harami", df)
    assert result is not None
    assert result["pattern"] == "Bullish Harami"
    assert result["bias"] == "bullish"


def test_tweezer_bottom_detected():
    prefix = _downtrend_prefix(6, start=110.0)
    # Both candles have same low (within 0.1%)
    # Candle 2 is bullish
    candle1 = (105.0, 106.0, 100.0, 104.0)   # bearish
    candle2 = (102.0, 107.0, 100.0, 106.0)   # bullish, same low
    df = _make_df(prefix + [candle1, candle2])
    result = detect_candlestick_pattern("tweezer bottom", df)
    assert result is not None
    assert result["pattern"] == "Tweezer Bottom"
    assert result["bias"] == "bullish"


def test_piercing_line_detected():
    prefix = _downtrend_prefix(6, start=110.0)
    # Candle 1 bearish: o=106, h=107, l=102, c=103 (body=3)
    # Candle 2 bullish: opens below C1 low (101.5 < 102), closes above midpoint ((106+103)/2=104.5)
    # c2=105 > 104.5 ✓, c2=105 < 106=o1 ✓
    candle1 = (106.0, 107.0, 102.0, 103.0)
    candle2 = (101.5, 106.0, 101.0, 105.0)
    df = _make_df(prefix + [candle1, candle2])
    result = detect_candlestick_pattern("piercing line", df)
    assert result is not None
    assert result["pattern"] == "Piercing Line"
    assert result["bias"] == "bullish"


# ---------------------------------------------------------------------------
# Three-candle patterns
# ---------------------------------------------------------------------------


def test_three_white_soldiers_detected():
    prefix = _downtrend_prefix(6, start=110.0)
    # Three consecutive bullish candles, each opening within prior body and closing higher
    c1 = (100.0, 102.5, 99.5, 102.0)  # bullish
    c2 = (101.0, 104.5, 100.5, 104.0)  # opens within C1 body, closes higher
    c3 = (102.5, 106.5, 102.0, 106.0)  # opens within C2 body, closes higher
    df = _make_df(prefix + [c1, c2, c3])
    result = detect_candlestick_pattern("three white soldiers", df)
    assert result is not None
    assert result["pattern"] == "Three White Soldiers"
    assert result["bias"] == "bullish"
    assert len(result["candle_indices"]) == 3


def test_three_black_crows_detected():
    prefix = _uptrend_prefix(6, start=90.0)
    # Three consecutive bearish candles, each opening within prior body and closing lower
    # C1: bearish, o=96, c=94, body in [94,96]
    # C2: opens within [94,96], say o=95, c=93
    # C3: opens within [93,95], say o=94, c=91
    c1 = (96.0, 96.5, 93.5, 94.0)   # bearish
    c2 = (95.0, 95.5, 92.5, 93.0)   # opens within C1 body, closes lower
    c3 = (94.0, 94.5, 90.5, 91.0)   # opens within C2 body, closes lower
    df = _make_df(prefix + [c1, c2, c3])
    result = detect_candlestick_pattern("three black crows", df)
    assert result is not None
    assert result["pattern"] == "Three Black Crows"
    assert result["bias"] == "bearish"


def test_morning_star_detected():
    prefix = _downtrend_prefix(6, start=110.0)
    # C1 large bearish: o=106, c=102, body midpoint=104
    # C2 star: gaps down (o2 < c1=102), small body
    # C3 bullish: closes above midpoint of C1 (104)
    c1 = (106.0, 107.0, 101.5, 102.0)  # large bearish
    c2 = (101.0, 101.5, 100.0, 100.8)  # star: o2=101 < c1=102 ✓, small body ✓
    c3 = (101.5, 106.0, 101.0, 105.0)  # bullish, c3=105 > midpoint(106+102)/2=104 ✓
    df = _make_df(prefix + [c1, c2, c3])
    result = detect_candlestick_pattern("morning star", df)
    assert result is not None
    assert result["pattern"] == "Morning Star"
    assert result["bias"] == "bullish"
    assert len(result["candle_indices"]) == 3


def test_three_inside_up_detected():
    prefix = _downtrend_prefix(6, start=110.0)
    # With 9 total bars: close[-6] = close[3] ≈ 108.02
    # C3 close=104.5 → pct = (104.5-108.02)/108.02 ≈ -3.3% < -2% → downtrend ✓
    c1 = (106.0, 107.0, 101.5, 102.0)   # large bearish, body in [102, 106]
    c2 = (103.0, 104.5, 102.5, 103.5)   # small bullish contained in C1 body
    c3 = (103.5, 105.0, 103.0, 104.5)   # bullish, closes above C2 (103.5) ✓
    df = _make_df(prefix + [c1, c2, c3])
    result = detect_candlestick_pattern("three inside up", df)
    assert result is not None
    assert result["pattern"] == "Three Inside Up"
    assert result["bias"] == "bullish"


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


def test_candlestick_result_has_required_keys():
    prefix = _downtrend_prefix(6, start=110.0)
    hammer_candle = [(104.0, 104.3, 100.0, 104.2)]
    df = _make_df(prefix + hammer_candle)
    result = detect_candlestick_pattern("hammer", df)
    assert result is not None
    required_keys = {
        "pattern", "pattern_type", "confidence", "bias", "key_points",
        "lines", "candle_indices", "description", "state",
        "breakout_level_upper", "breakout_level_lower",
        "compression_ratio", "distance_to_breakout",
    }
    assert required_keys.issubset(result.keys())
    assert result["lines"] == []
    assert result["breakout_level_upper"] is None
    assert result["breakout_level_lower"] is None
    assert result["compression_ratio"] is None
    assert result["distance_to_breakout"] is None


def test_candlestick_confidence_within_bounds():
    prefix = _downtrend_prefix(6, start=110.0)
    hammer_candle = [(104.0, 104.3, 100.0, 104.2)]
    df = _make_df(prefix + hammer_candle)
    result = detect_candlestick_pattern("hammer", df)
    assert result is not None
    assert 0 <= result["confidence"] <= 100


# ---------------------------------------------------------------------------
# Integration: candlestick patterns in detect_pattern
# ---------------------------------------------------------------------------


def test_detect_pattern_finds_hammer():
    """detect_pattern() can detect candlestick patterns via the registry."""
    from momentum_radar.patterns.detector import detect_pattern

    prefix = _downtrend_prefix(20, start=110.0)
    hammer_candle = [(104.0, 104.3, 100.0, 104.2)]
    df = _make_df(prefix + hammer_candle)
    result = detect_pattern("hammer", df)
    assert result is not None
    assert result["pattern"] == "Hammer"
    assert result["pattern_type"] == "candlestick"


def test_detect_pattern_candlestick_in_registry():
    """All candlestick patterns are accessible via the main registry."""
    from momentum_radar.patterns.detector import available_patterns

    patterns = available_patterns()
    assert "hammer" in patterns
    assert "bullish engulfing" in patterns
    assert "morning star" in patterns
    assert "three black crows" in patterns
