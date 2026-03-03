"""
data_fetcher.py – Abstract data provider interface and yfinance/finnhub implementations.

The abstract base class ``BaseDataFetcher`` defines the contract that all data
providers must satisfy.  New providers (Polygon, Alpaca, etc.) should subclass
``BaseDataFetcher`` and implement every abstract method.

The concrete ``YFinanceDataFetcher`` uses *yfinance* as an alternative source.
The concrete ``FinnhubDataFetcher`` uses the *finnhub* REST API as the default source.
"""

import logging
import time
import datetime
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _normalise_yf_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    """Normalise yfinance DataFrame column names to lowercase strings.

    yfinance may return MultiIndex columns (e.g. ``('Close', 'AAPL')``) or
    plain string columns depending on the version.  This helper flattens
    and lowercases them consistently.
    """
    df.columns = [
        c.lower() if isinstance(c, str) else (c[0].lower() if c else "")
        for c in df.columns
    ]
    return df


class BaseDataFetcher(ABC):
    """Abstract base class for all data providers."""

    @abstractmethod
    def get_intraday_bars(
        self,
        ticker: str,
        interval: str = "1m",
        period: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """Return intraday OHLCV bars as a DataFrame indexed by datetime.

        Columns expected: ``open``, ``high``, ``low``, ``close``, ``volume``.

        Args:
            ticker: Stock symbol (e.g. ``"AAPL"``).
            interval: Bar interval string (e.g. ``"1m"``, ``"5m"``).
            period: Look-back period string (e.g. ``"1d"``, ``"5d"``).

        Returns:
            DataFrame or ``None`` if data is unavailable.
        """

    @abstractmethod
    def get_daily_bars(
        self,
        ticker: str,
        period: str = "60d",
    ) -> Optional[pd.DataFrame]:
        """Return daily OHLCV bars.

        Args:
            ticker: Stock symbol.
            period: Look-back period string.

        Returns:
            DataFrame or ``None`` if data is unavailable.
        """

    @abstractmethod
    def get_quote(self, ticker: str) -> Optional[Dict]:
        """Return a snapshot dict with at least ``price`` and ``volume`` keys.

        Args:
            ticker: Stock symbol.

        Returns:
            Dict or ``None`` if unavailable.
        """

    @abstractmethod
    def get_fundamentals(self, ticker: str) -> Optional[Dict]:
        """Return fundamental data including float and short interest.

        Expected keys (where available): ``float_shares``, ``short_ratio``,
        ``short_percent_of_float``, ``shares_outstanding``.

        Args:
            ticker: Stock symbol.

        Returns:
            Dict or ``None`` if unavailable.
        """

    @abstractmethod
    def get_options_volume(self, ticker: str) -> Optional[Dict]:
        """Return options activity summary.

        Expected keys: ``call_volume``, ``put_volume``, ``avg_call_volume``,
        ``avg_put_volume``.

        Args:
            ticker: Stock symbol.

        Returns:
            Dict or ``None`` if unavailable.
        """


# ---------------------------------------------------------------------------
# yfinance implementation
# ---------------------------------------------------------------------------

class YFinanceDataFetcher(BaseDataFetcher):
    """Data fetcher backed by the *yfinance* library (default V1 provider)."""

    def get_intraday_bars(
        self,
        ticker: str,
        interval: str = "1m",
        period: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """Fetch intraday OHLCV bars via yfinance."""
        try:
            import yfinance as yf

            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )
            if df is None or df.empty:
                logger.warning("No intraday data returned for %s", ticker)
                return None
            _normalise_yf_columns(df)
            return df
        except Exception as exc:
            logger.error("Error fetching intraday bars for %s: %s", ticker, exc)
            return None

    def get_daily_bars(
        self,
        ticker: str,
        period: str = "60d",
    ) -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV bars via yfinance."""
        try:
            import yfinance as yf

            df = yf.download(
                ticker,
                period=period,
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if df is None or df.empty:
                logger.warning("No daily data returned for %s", ticker)
                return None
            _normalise_yf_columns(df)
            return df
        except Exception as exc:
            logger.error("Error fetching daily bars for %s: %s", ticker, exc)
            return None

    def get_quote(self, ticker: str) -> Optional[Dict]:
        """Fetch a live quote snapshot via yfinance."""
        try:
            import yfinance as yf

            t = yf.Ticker(ticker)
            info = t.fast_info
            return {
                "price": getattr(info, "last_price", None),
                "volume": getattr(info, "last_volume", None),
                "prev_close": getattr(info, "previous_close", None),
            }
        except Exception as exc:
            logger.error("Error fetching quote for %s: %s", ticker, exc)
            return None

    def get_fundamentals(self, ticker: str) -> Optional[Dict]:
        """Fetch fundamental / short-interest data via yfinance."""
        try:
            import yfinance as yf

            info = yf.Ticker(ticker).info
            return {
                "float_shares": info.get("floatShares"),
                "short_ratio": info.get("shortRatio"),
                "short_percent_of_float": info.get("shortPercentOfFloat"),
                "shares_outstanding": info.get("sharesOutstanding"),
            }
        except Exception as exc:
            logger.error("Error fetching fundamentals for %s: %s", ticker, exc)
            return None

    def get_options_volume(self, ticker: str) -> Optional[Dict]:
        """Estimate options activity from the nearest expiry chain."""
        try:
            import yfinance as yf

            t = yf.Ticker(ticker)
            expirations = t.options
            if not expirations:
                return None
            chain = t.option_chain(expirations[0])
            call_vol = int(chain.calls["volume"].sum()) if not chain.calls.empty else 0
            put_vol = int(chain.puts["volume"].sum()) if not chain.puts.empty else 0
            # Approximate averages using a simple 1.0 baseline (no history)
            return {
                "call_volume": call_vol,
                "put_volume": put_vol,
                "avg_call_volume": max(call_vol / 3, 1),
                "avg_put_volume": max(put_vol / 3, 1),
            }
        except Exception as exc:
            logger.error("Error fetching options volume for %s: %s", ticker, exc)
            return None


# ---------------------------------------------------------------------------
# Finnhub implementation
# ---------------------------------------------------------------------------

class FinnhubDataFetcher(BaseDataFetcher):
    """Data fetcher backed by the *finnhub* REST API (default provider).

    Requires a free API key from https://finnhub.io.  Set ``FINNHUB_API_KEY``
    in your ``.env`` file or environment.  The free tier allows 60 API calls
    per minute — a small sleep is added between calls to stay within limits.
    """

    def __init__(self) -> None:
        import finnhub
        from momentum_radar.config import config

        api_key = config.data.finnhub_api_key or ""
        if not api_key:
            logger.warning(
                "FINNHUB_API_KEY is not set. Finnhub requests will fail. "
                "Get a free key at https://finnhub.io"
            )
        self._client = finnhub.Client(api_key=api_key)

    # Mapping from interval strings to Finnhub resolution codes.
    # Finnhub supports: 1, 5, 15, 30, 60 (minutes), D, W, M.
    _INTERVAL_MAP: Dict[str, str] = {
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "60m": "60",
        "1h": "60",
    }

    def get_intraday_bars(
        self,
        ticker: str,
        interval: str = "1m",
        period: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """Fetch intraday OHLCV bars via Finnhub ``stock_candles``."""
        try:
            resolution = self._INTERVAL_MAP.get(interval, "5")
            now = int(datetime.datetime.now().timestamp())
            # For "1d" look-back use approximately 1 trading day of seconds
            lookback = 86400
            if period.endswith("d"):
                lookback = int(period[:-1]) * 86400
            start = now - lookback
            time.sleep(0.05)
            resp = self._client.stock_candles(ticker, resolution, start, now)
            if resp.get("s") != "ok":
                logger.warning("No intraday data returned for %s", ticker)
                return None
            df = pd.DataFrame({
                "open": resp["o"],
                "high": resp["h"],
                "low": resp["l"],
                "close": resp["c"],
                "volume": resp["v"],
            }, index=pd.to_datetime(resp["t"], unit="s", utc=True))
            return df
        except Exception as exc:
            logger.error("Error fetching intraday bars for %s: %s", ticker, exc)
            return None

    def get_daily_bars(
        self,
        ticker: str,
        period: str = "60d",
    ) -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV bars via Finnhub ``stock_candles`` (resolution ``D``)."""
        try:
            days = 60
            if period.endswith("d"):
                days = int(period[:-1])
            now = int(datetime.datetime.now().timestamp())
            start = now - days * 86400
            time.sleep(0.05)
            resp = self._client.stock_candles(ticker, "D", start, now)
            if resp.get("s") != "ok":
                logger.warning("No daily data returned for %s", ticker)
                return None
            df = pd.DataFrame({
                "open": resp["o"],
                "high": resp["h"],
                "low": resp["l"],
                "close": resp["c"],
                "volume": resp["v"],
            }, index=pd.to_datetime(resp["t"], unit="s", utc=True))
            return df
        except Exception as exc:
            logger.error("Error fetching daily bars for %s: %s", ticker, exc)
            return None

    def get_quote(self, ticker: str) -> Optional[Dict]:
        """Fetch a live quote snapshot via Finnhub ``quote``."""
        try:
            time.sleep(0.05)
            resp = self._client.quote(ticker)
            return {
                "price": resp.get("c"),
                "volume": resp.get("v"),
                "prev_close": resp.get("pc"),
            }
        except Exception as exc:
            logger.error("Error fetching quote for %s: %s", ticker, exc)
            return None

    def get_fundamentals(self, ticker: str) -> Optional[Dict]:
        """Fetch fundamental / short-interest data via Finnhub ``company_basic_financials``."""
        try:
            time.sleep(0.05)
            resp = self._client.company_basic_financials(ticker, "all")
            metric = resp.get("metric", {})
            return {
                "float_shares": metric.get("floatShares") or metric.get("shareFloat"),
                "short_ratio": metric.get("shortRatio"),
                "short_percent_of_float": metric.get("shortPercentOfFloat"),
                "shares_outstanding": metric.get("sharesOutstanding"),
            }
        except Exception as exc:
            logger.error("Error fetching fundamentals for %s: %s", ticker, exc)
            return None

    def get_options_volume(self, ticker: str) -> Optional[Dict]:
        """Return ``None`` — options data is not available on the Finnhub free tier."""
        logger.debug(
            "Options data not available with Finnhub free tier for %s", ticker
        )
        return None



# ---------------------------------------------------------------------------
# Hybrid implementation (yfinance for bars, Finnhub for quotes/fundamentals)
# ---------------------------------------------------------------------------

class HybridDataFetcher(BaseDataFetcher):
    """Hybrid fetcher: yfinance for OHLCV bars, Finnhub for quotes/fundamentals.

    The Finnhub free tier does NOT support ``stock_candles`` (returns 403), so
    bar data is sourced from *yfinance* instead.  Quotes and fundamentals still
    use Finnhub for real-time accuracy.

    A local SQLite cache directory (``.yf_cache``) is created next to the
    package root to work around yfinance timezone-cache issues.
    """

    def __init__(self) -> None:
        import os
        import yfinance as yf
        import finnhub
        from momentum_radar.config import config

        # yfinance SQLite cache fix
        from pathlib import Path
        cache_dir = str(Path(__file__).parent.parent.parent / ".yf_cache")
        import os as _os
        _os.makedirs(cache_dir, exist_ok=True)
        try:
            yf.set_tz_cache_location(cache_dir)
        except Exception:
            pass

        api_key = config.data.finnhub_api_key or ""
        if not api_key:
            logger.warning(
                "FINNHUB_API_KEY is not set. Finnhub requests will fail. "
                "Get a free key at https://finnhub.io"
            )
        self._finnhub_client = finnhub.Client(api_key=api_key)

    # ------------------------------------------------------------------
    # Bar data via yfinance
    # ------------------------------------------------------------------

    def get_intraday_bars(
        self,
        ticker: str,
        interval: str = "1m",
        period: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """Fetch intraday OHLCV bars via yfinance."""
        try:
            import yfinance as yf

            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )
            if df is None or df.empty:
                logger.warning("No intraday data returned for %s", ticker)
                return None
            _normalise_yf_columns(df)
            return df
        except Exception as exc:
            logger.error("Error fetching intraday bars for %s: %s", ticker, exc)
            return None

    def get_daily_bars(
        self,
        ticker: str,
        period: str = "60d",
    ) -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV bars via yfinance."""
        try:
            import yfinance as yf

            df = yf.download(
                ticker,
                period=period,
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if df is None or df.empty:
                logger.warning("No daily data returned for %s", ticker)
                return None
            _normalise_yf_columns(df)
            return df
        except Exception as exc:
            logger.error("Error fetching daily bars for %s: %s", ticker, exc)
            return None

    # ------------------------------------------------------------------
    # Quotes and fundamentals via Finnhub
    # ------------------------------------------------------------------

    def get_quote(self, ticker: str) -> Optional[Dict]:
        """Fetch a live quote snapshot via Finnhub."""
        try:
            time.sleep(0.05)
            resp = self._finnhub_client.quote(ticker)
            return {
                "price": resp.get("c"),
                "volume": resp.get("v"),
                "prev_close": resp.get("pc"),
            }
        except Exception as exc:
            logger.error("Error fetching quote for %s: %s", ticker, exc)
            return None

    def get_fundamentals(self, ticker: str) -> Optional[Dict]:
        """Fetch fundamental data via Finnhub ``company_basic_financials``."""
        try:
            time.sleep(0.05)
            resp = self._finnhub_client.company_basic_financials(ticker, "all")
            metric = resp.get("metric", {})
            return {
                "float_shares": metric.get("floatShares") or metric.get("shareFloat"),
                "short_ratio": metric.get("shortRatio"),
                "short_percent_of_float": metric.get("shortPercentOfFloat"),
                "shares_outstanding": metric.get("sharesOutstanding"),
            }
        except Exception as exc:
            logger.error("Error fetching fundamentals for %s: %s", ticker, exc)
            return None

    def get_options_volume(self, ticker: str) -> Optional[Dict]:
        """Return ``None`` - options data is not available in hybrid mode."""
        logger.debug("Options data not available with HybridDataFetcher for %s", ticker)
        return None


def get_data_fetcher(provider: str = "finnhub") -> BaseDataFetcher:
    """Factory function that returns the appropriate data fetcher.

    Args:
        provider: Provider name string.  Supported values:

                  - ``"finnhub"``      — :class:`HybridDataFetcher` (default)
                    Uses yfinance for OHLCV bars and Finnhub for quotes /
                    fundamentals.  Works with the free Finnhub tier.
                  - ``"yfinance"``     — :class:`YFinanceDataFetcher`
                    Fully yfinance-backed; useful when no Finnhub key is set.
                  - ``"finnhub_paid"`` — :class:`FinnhubDataFetcher`
                    Pure Finnhub implementation for paid-tier subscribers.

    Returns:
        A concrete :class:`BaseDataFetcher` instance.

    Raises:
        ValueError: If the requested provider is not supported.
    """
    from momentum_radar.data.ibkr_fetcher import IBKRDataFetcher  # noqa: PLC0415
    providers: Dict[str, type] = {
        "finnhub": HybridDataFetcher,
        "yfinance": YFinanceDataFetcher,
        "finnhub_paid": FinnhubDataFetcher,
        "ibkr": IBKRDataFetcher,
    }
    if provider not in providers:
        raise ValueError(
            f"Unsupported data provider '{provider}'. "
            f"Available providers: {list(providers.keys())}"
        )
    return providers[provider]()
