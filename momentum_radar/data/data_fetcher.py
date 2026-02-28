"""
data_fetcher.py – Abstract data provider interface and yfinance implementation.

The abstract base class ``BaseDataFetcher`` defines the contract that all data
providers must satisfy.  New providers (Polygon, Alpaca, etc.) should subclass
``BaseDataFetcher`` and implement every abstract method.

The concrete ``YFinanceDataFetcher`` uses *yfinance* as the default V1 source.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


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
            df.columns = [c.lower() for c in df.columns]
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
            df.columns = [c.lower() for c in df.columns]
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


def get_data_fetcher(provider: str = "yfinance") -> BaseDataFetcher:
    """Factory function that returns the appropriate data fetcher.

    Args:
        provider: Provider name string (``"yfinance"`` supported in V1).

    Returns:
        A concrete :class:`BaseDataFetcher` instance.

    Raises:
        ValueError: If the requested provider is not supported.
    """
    providers: Dict[str, type] = {
        "yfinance": YFinanceDataFetcher,
    }
    if provider not in providers:
        raise ValueError(
            f"Unsupported data provider '{provider}'. "
            f"Available providers: {list(providers.keys())}"
        )
    return providers[provider]()
