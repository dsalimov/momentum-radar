"""
ibkr_fetcher.py – Optional Interactive Brokers data fetcher.

Requires ``ib_insync`` to be installed::

    pip install ib_insync

And an active TWS or IB Gateway instance running on localhost (port 7497 for
paper trading, 7496 for live trading).

Set environment variables::

    IBKR_HOST=127.0.0.1
    IBKR_PORT=7497          # 7496 for live, 7497 for paper
    IBKR_CLIENT_ID=1

This fetcher is optional.  When IBKR is unavailable, the system falls back to
the default HybridDataFetcher automatically.
"""

import logging
import time
from typing import Dict, List, Optional

import pandas as pd

from momentum_radar.data.data_fetcher import BaseDataFetcher

logger = logging.getLogger(__name__)


class IBKRDataFetcher(BaseDataFetcher):
    """Data fetcher backed by Interactive Brokers TWS / IB Gateway via ib_insync.

    Features:
    - Real-time market data (level 1)
    - Historical OHLCV bars
    - Borrow fee rates via ``reqShortableShares``
    - Auto-reconnect on disconnect

    If ``ib_insync`` is not installed or IBKR is unreachable, every method
    returns ``None`` (the system uses its fallback provider transparently).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        timeout: float = 4.0,
    ) -> None:
        import os

        self._host = os.getenv("IBKR_HOST", host)
        self._port = int(os.getenv("IBKR_PORT", port))
        self._client_id = int(os.getenv("IBKR_CLIENT_ID", client_id))
        self._timeout = timeout
        self._ib = None
        self._connected = False
        self._connect()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Attempt to connect to IBKR TWS / IB Gateway."""
        try:
            from ib_insync import IB
            self._ib = IB()
            self._ib.connect(
                self._host,
                self._port,
                clientId=self._client_id,
                timeout=self._timeout,
                readonly=True,
            )
            self._connected = True
            logger.info(
                "IBKR connected: %s:%d (client %d)",
                self._host,
                self._port,
                self._client_id,
            )
        except Exception as exc:
            logger.warning("IBKR connection failed: %s – using fallback data.", exc)
            self._connected = False

    def _ensure_connected(self) -> bool:
        """Re-connect if disconnected.  Returns ``True`` if connected."""
        if self._ib is None:
            return False
        if not self._connected or not self._ib.isConnected():
            logger.info("IBKR reconnecting…")
            try:
                self._ib.connect(
                    self._host,
                    self._port,
                    clientId=self._client_id,
                    timeout=self._timeout,
                    readonly=True,
                )
                self._connected = True
            except Exception as exc:
                logger.warning("IBKR reconnect failed: %s", exc)
                self._connected = False
        return self._connected

    # ------------------------------------------------------------------
    # BaseDataFetcher interface
    # ------------------------------------------------------------------

    def _make_contract(self, ticker: str):
        """Build a US stock contract."""
        from ib_insync import Stock
        return Stock(ticker, "SMART", "USD")

    def get_intraday_bars(
        self,
        ticker: str,
        interval: str = "1m",
        period: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """Fetch intraday OHLCV bars from IBKR historical data API."""
        if not self._ensure_connected():
            return None
        try:
            from ib_insync import util

            # Map interval to IBKR barSize string
            bar_map = {
                "1m": "1 min",
                "5m": "5 mins",
                "15m": "15 mins",
                "30m": "30 mins",
                "1h": "1 hour",
                "60m": "1 hour",
            }
            bar_size = bar_map.get(interval, "1 min")

            # Map period to IBKR duration string
            dur_map = {"1d": "1 D", "5d": "5 D", "1w": "1 W", "1mo": "1 M"}
            duration = dur_map.get(period, "1 D")

            contract = self._make_contract(ticker)
            bars = self._ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
            if not bars:
                return None
            df = util.df(bars)
            df.index = pd.to_datetime(df["date"])
            return df[["open", "high", "low", "close", "volume"]]
        except Exception as exc:
            logger.error("IBKR get_intraday_bars(%s): %s", ticker, exc)
            return None

    def get_daily_bars(
        self,
        ticker: str,
        period: str = "60d",
    ) -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV bars from IBKR."""
        if not self._ensure_connected():
            return None
        try:
            from ib_insync import util

            days = 60
            if period.endswith("d"):
                days = int(period[:-1])
            duration = f"{days} D"

            contract = self._make_contract(ticker)
            bars = self._ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
            if not bars:
                return None
            df = util.df(bars)
            df.index = pd.to_datetime(df["date"])
            return df[["open", "high", "low", "close", "volume"]]
        except Exception as exc:
            logger.error("IBKR get_daily_bars(%s): %s", ticker, exc)
            return None

    def get_quote(self, ticker: str) -> Optional[Dict]:
        """Fetch a live quote snapshot from IBKR market data."""
        if not self._ensure_connected():
            return None
        try:
            contract = self._make_contract(ticker)
            ticker_data = self._ib.reqMktData(contract, "", False, False)
            self._ib.sleep(0.5)  # allow market data snapshot to arrive
            price = ticker_data.last or ticker_data.close or None
            prev_close = ticker_data.close or None
            volume = ticker_data.volume or None
            self._ib.cancelMktData(contract)
            return {"price": price, "volume": volume, "prev_close": prev_close}
        except Exception as exc:
            logger.error("IBKR get_quote(%s): %s", ticker, exc)
            return None

    def get_fundamentals(self, ticker: str) -> Optional[Dict]:
        """Fetch fundamental data from IBKR (uses yfinance as a fallback for fundamentals)."""
        # IBKR fundamental data requires a dedicated subscription.
        # Fall back to yfinance for short interest / float data.
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
            logger.error("IBKR get_fundamentals(%s) fallback failed: %s", ticker, exc)
            return None

    def get_options_volume(self, ticker: str) -> Optional[Dict]:
        """Fetch options chain from IBKR."""
        if not self._ensure_connected():
            return None
        try:
            import yfinance as yf
            # IBKR options chains require subscribe calls; use yfinance for simplicity
            t = yf.Ticker(ticker)
            expirations = t.options
            if not expirations:
                return None
            chain = t.option_chain(expirations[0])
            call_vol = int(chain.calls["volume"].sum()) if not chain.calls.empty else 0
            put_vol = int(chain.puts["volume"].sum()) if not chain.puts.empty else 0
            return {
                "call_volume": call_vol,
                "put_volume": put_vol,
                "avg_call_volume": max(call_vol / 3, 1),
                "avg_put_volume": max(put_vol / 3, 1),
            }
        except Exception as exc:
            logger.error("IBKR get_options_volume(%s): %s", ticker, exc)
            return None

    def get_borrow_rate(self, ticker: str) -> Optional[float]:
        """Fetch the estimated borrow fee rate from IBKR shortable shares data.

        Returns the annual borrow rate as a decimal (e.g. 0.15 = 15 %) or
        ``None`` if unavailable.
        """
        if not self._ensure_connected():
            return None
        try:
            contract = self._make_contract(ticker)
            shortable = self._ib.reqShortableShares(contract)
            self._ib.sleep(0.3)
            if shortable is None or shortable <= 0:
                return None
            # IBKR provides shortable share count; map to estimated fee tier
            # Based on typical IBKR borrow fee schedule:
            # > 1M shares available  → easy-to-borrow  (~0.25% p.a.)
            # 100K–1M               → moderate borrow  (~5–15% p.a.)
            # < 100K                → hard-to-borrow   (~25–100% p.a.)
            if shortable > 1_000_000:
                return 0.0025
            elif shortable > 100_000:
                return 0.10
            else:
                return 0.35
        except Exception as exc:
            logger.debug("IBKR get_borrow_rate(%s): %s", ticker, exc)
            return None

    def disconnect(self) -> None:
        """Disconnect from IBKR."""
        if self._ib and self._connected:
            try:
                self._ib.disconnect()
            except Exception:
                pass
            self._connected = False
            logger.info("IBKR disconnected.")
