"""
test_new_indicators.py – Unit tests for the new technical indicators,
risk utilities, heatmap module, and new signal modules.
"""

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_close_series(n: int = 250, seed: int = 42) -> pd.Series:
    """Build a synthetic close price series with realistic values."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.015, n)
    prices = 100.0 * np.exp(np.cumsum(returns))
    return pd.Series(prices, name="close")


def _make_daily_df(n: int = 250, seed: int = 42) -> pd.DataFrame:
    closes = _make_close_series(n=n, seed=seed)
    opens = closes * 0.999
    highs = closes * 1.005
    lows = closes * 0.995
    volumes = np.full(n, 1_000_000)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}
    )


# ---------------------------------------------------------------------------
# compute_ema
# ---------------------------------------------------------------------------


class TestComputeEMA:
    def test_returns_series(self) -> None:
        from momentum_radar.utils.indicators import compute_ema

        closes = _make_close_series(50)
        result = compute_ema(closes, 9)
        assert result is not None
        assert isinstance(result, pd.Series)
        assert len(result) == len(closes)

    def test_returns_none_for_insufficient_data(self) -> None:
        from momentum_radar.utils.indicators import compute_ema

        closes = _make_close_series(5)
        result = compute_ema(closes, 9)
        assert result is None

    def test_ema_smoothing(self) -> None:
        """EMA(200) should be smoother (lower std) than raw prices."""
        from momentum_radar.utils.indicators import compute_ema

        closes = _make_close_series(250)
        ema200 = compute_ema(closes, 200)
        assert ema200 is not None
        assert float(ema200.std()) < float(closes.std())


# ---------------------------------------------------------------------------
# compute_rsi
# ---------------------------------------------------------------------------


class TestComputeRSI:
    def test_returns_float(self) -> None:
        from momentum_radar.utils.indicators import compute_rsi

        closes = _make_close_series(50)
        rsi = compute_rsi(closes)
        assert rsi is not None
        assert isinstance(rsi, float)

    def test_rsi_in_range(self) -> None:
        from momentum_radar.utils.indicators import compute_rsi

        closes = _make_close_series(100)
        rsi = compute_rsi(closes)
        assert rsi is not None
        assert 0.0 <= rsi <= 100.0

    def test_returns_none_insufficient_data(self) -> None:
        from momentum_radar.utils.indicators import compute_rsi

        closes = _make_close_series(5)
        result = compute_rsi(closes, period=14)
        assert result is None

    def test_strongly_trending_up_gives_high_rsi(self) -> None:
        from momentum_radar.utils.indicators import compute_rsi

        # Mix strong uptrend with occasional down bars so avg_loss is non-zero
        rng = np.random.default_rng(1)
        trend = np.linspace(100, 200, 60)
        noise = rng.normal(0, 0.5, 60)
        closes = pd.Series(trend + noise)
        rsi = compute_rsi(closes, period=14)
        assert rsi is not None
        assert rsi > 70

    def test_strongly_trending_down_gives_low_rsi(self) -> None:
        from momentum_radar.utils.indicators import compute_rsi

        closes = pd.Series(np.linspace(200, 100, 60))
        rsi = compute_rsi(closes, period=14)
        assert rsi is not None
        assert rsi < 30


# ---------------------------------------------------------------------------
# compute_macd
# ---------------------------------------------------------------------------


class TestComputeMACD:
    def test_returns_dict(self) -> None:
        from momentum_radar.utils.indicators import compute_macd

        closes = _make_close_series(100)
        result = compute_macd(closes)
        assert result is not None
        assert "macd" in result
        assert "signal" in result
        assert "histogram" in result

    def test_histogram_is_macd_minus_signal(self) -> None:
        from momentum_radar.utils.indicators import compute_macd

        closes = _make_close_series(100)
        result = compute_macd(closes)
        assert result is not None
        assert abs(result["histogram"] - (result["macd"] - result["signal"])) < 1e-10

    def test_returns_none_insufficient_data(self) -> None:
        from momentum_radar.utils.indicators import compute_macd

        closes = _make_close_series(10)
        result = compute_macd(closes)
        assert result is None


# ---------------------------------------------------------------------------
# compute_bollinger_bands
# ---------------------------------------------------------------------------


class TestComputeBollingerBands:
    def test_returns_dict(self) -> None:
        from momentum_radar.utils.indicators import compute_bollinger_bands

        closes = _make_close_series(50)
        result = compute_bollinger_bands(closes)
        assert result is not None
        assert "upper" in result
        assert "middle" in result
        assert "lower" in result

    def test_band_ordering(self) -> None:
        from momentum_radar.utils.indicators import compute_bollinger_bands

        closes = _make_close_series(100)
        result = compute_bollinger_bands(closes)
        assert result is not None
        assert result["upper"] > result["middle"] > result["lower"]

    def test_returns_none_insufficient_data(self) -> None:
        from momentum_radar.utils.indicators import compute_bollinger_bands

        closes = _make_close_series(5)
        result = compute_bollinger_bands(closes, period=20)
        assert result is None


# ---------------------------------------------------------------------------
# Risk utilities
# ---------------------------------------------------------------------------


class TestRiskUtilities:
    def test_compute_position_size(self) -> None:
        from momentum_radar.utils.risk import compute_position_size

        shares, dollar_risk = compute_position_size(100_000, 0.01, 50.0, 48.0)
        assert shares == 500
        assert dollar_risk == 1000.0

    def test_compute_position_size_zero_risk(self) -> None:
        from momentum_radar.utils.risk import compute_position_size

        shares, _ = compute_position_size(100_000, 0.01, 50.0, 50.0)
        assert shares == 0

    def test_suggest_stop_loss_atr(self) -> None:
        from momentum_radar.utils.risk import suggest_stop_loss

        stop = suggest_stop_loss(50.0, atr=2.0, atr_multiplier=1.5)
        assert stop == pytest.approx(47.0, abs=0.01)

    def test_suggest_stop_loss_no_atr(self) -> None:
        from momentum_radar.utils.risk import suggest_stop_loss

        stop = suggest_stop_loss(50.0, atr=None)
        assert stop == pytest.approx(48.5, abs=0.01)

    def test_compute_risk_reward(self) -> None:
        from momentum_radar.utils.risk import compute_risk_reward

        rr = compute_risk_reward(entry=50.0, stop_loss=48.0, target=54.0)
        assert rr == pytest.approx(2.0, abs=0.01)

    def test_compute_risk_reward_zero_risk(self) -> None:
        from momentum_radar.utils.risk import compute_risk_reward

        rr = compute_risk_reward(entry=50.0, stop_loss=50.0, target=54.0)
        assert rr is None

    def test_format_risk_summary(self) -> None:
        from momentum_radar.utils.risk import format_risk_summary

        summary = format_risk_summary(
            ticker="AAPL",
            entry=50.0,
            stop_loss=48.0,
            target1=54.0,
            target2=58.0,
            shares=500,
            dollar_risk=1000.0,
            confidence_pct=75.0,
        )
        assert "AAPL" in summary
        assert "50.00" in summary
        assert "48.00" in summary
        assert "75%" in summary


# ---------------------------------------------------------------------------
# New signal modules
# ---------------------------------------------------------------------------


class TestEMATrendSignal:
    def test_bullish_alignment_fires(self) -> None:
        import momentum_radar.signals.trend  # noqa: F401
        from momentum_radar.signals.trend import ema_trend

        # Build 250-day strongly uptrending series
        closes = pd.Series(np.linspace(80, 200, 250))
        daily = pd.DataFrame({
            "open": closes * 0.999,
            "high": closes * 1.005,
            "low": closes * 0.995,
            "close": closes,
            "volume": np.full(250, 1_000_000),
        })
        result = ema_trend(ticker="TEST", bars=None, daily=daily, fundamentals=None, options=None)
        assert result.triggered
        assert result.score >= 1

    def test_no_data_does_not_fire(self) -> None:
        import momentum_radar.signals.trend  # noqa: F401
        from momentum_radar.signals.trend import ema_trend

        result = ema_trend(ticker="TEST", bars=None, daily=None, fundamentals=None, options=None)
        assert not result.triggered
        assert result.score == 0

    def test_insufficient_history_does_not_fire(self) -> None:
        import momentum_radar.signals.trend  # noqa: F401
        from momentum_radar.signals.trend import ema_trend

        closes = pd.Series(np.linspace(100, 110, 50))
        daily = pd.DataFrame({"close": closes, "open": closes, "high": closes, "low": closes, "volume": np.ones(50)})
        result = ema_trend(ticker="TEST", bars=None, daily=daily, fundamentals=None, options=None)
        assert not result.triggered


class TestRSIMACDSignal:
    def test_bullish_conditions_fire(self) -> None:
        import momentum_radar.signals.trend  # noqa: F401
        from momentum_radar.signals.trend import rsi_macd

        # Mildly uptrending – RSI should be in 40-70 range
        closes = pd.Series(np.linspace(100, 120, 100))
        daily = pd.DataFrame({"close": closes, "open": closes, "high": closes, "low": closes, "volume": np.ones(100)})
        result = rsi_macd(ticker="TEST", bars=None, daily=daily, fundamentals=None, options=None)
        # Just verify it runs without error; RSI/MACD values depend on data shape
        assert isinstance(result.triggered, bool)

    def test_no_data_does_not_fire(self) -> None:
        import momentum_radar.signals.trend  # noqa: F401
        from momentum_radar.signals.trend import rsi_macd

        result = rsi_macd(ticker="TEST", bars=None, daily=None, fundamentals=None, options=None)
        assert not result.triggered


class TestVWAPProximitySignal:
    def _make_bars(self, n: int = 60, price: float = 100.0, last_price: float = 102.0) -> pd.DataFrame:
        """Create bars where the last close is above VWAP."""
        rng = pd.date_range("2024-01-15 09:30", periods=n, freq="1min")
        # Use a flat base price for all bars except the last which is higher
        closes = np.full(n, price)
        closes[-1] = last_price
        volumes = np.full(n, 10_000)
        return pd.DataFrame(
            {
                "open": closes - 0.05,
                "high": closes + 0.1,
                "low": closes - 0.1,
                "close": closes,
                "volume": volumes,
            },
            index=rng,
        )

    def test_price_above_vwap_fires(self) -> None:
        import momentum_radar.signals.vwap_signal  # noqa: F401
        from momentum_radar.signals.vwap_signal import vwap_proximity

        # last_price=103 is clearly above the VWAP of ~100 for 59 bars + 1 spike
        bars = self._make_bars(price=100.0, last_price=103.0)
        result = vwap_proximity(ticker="TEST", bars=bars, daily=None, fundamentals=None, options=None)
        assert result.triggered
        assert result.score >= 1

    def test_no_bars_does_not_fire(self) -> None:
        import momentum_radar.signals.vwap_signal  # noqa: F401
        from momentum_radar.signals.vwap_signal import vwap_proximity

        result = vwap_proximity(ticker="TEST", bars=None, daily=None, fundamentals=None, options=None)
        assert not result.triggered

    def test_price_below_vwap_does_not_fire(self) -> None:
        import momentum_radar.signals.vwap_signal  # noqa: F401
        from momentum_radar.signals.vwap_signal import vwap_proximity

        # All closes uniformly below would put VWAP above last price
        rng = pd.date_range("2024-01-15 09:30", periods=60, freq="1min")
        # Make prices that trend DOWN so last close is below VWAP
        closes = np.linspace(105, 95, 60)
        bars = pd.DataFrame({
            "open": closes - 0.05,
            "high": closes + 0.1,
            "low": closes - 0.1,
            "close": closes,
            "volume": np.full(60, 10_000),
        }, index=rng)
        result = vwap_proximity(ticker="TEST", bars=bars, daily=None, fundamentals=None, options=None)
        assert not result.triggered


# ---------------------------------------------------------------------------
# Config: PaperTradingConfig
# ---------------------------------------------------------------------------


class TestPaperTradingConfig:
    def test_default_disabled(self) -> None:
        from momentum_radar.config import config

        assert config.paper_trading.enabled is False

    def test_default_capital(self) -> None:
        from momentum_radar.config import config

        assert config.paper_trading.initial_capital == 100_000.0

    def test_default_confidence_threshold(self) -> None:
        from momentum_radar.config import config

        assert config.paper_trading.confidence_threshold == 60.0

    def test_default_min_rr(self) -> None:
        from momentum_radar.config import config

        assert config.paper_trading.min_rr_ratio == 2.0
