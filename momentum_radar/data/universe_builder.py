"""
universe_builder.py – Builds and filters the stock scanning universe.

The universe is populated from a curated list of highly liquid US equities.
Filters are applied based on price and average daily volume thresholds defined
in :class:`~momentum_radar.config.UniverseConfig`.
"""

import logging
from typing import List

from momentum_radar.config import config
from momentum_radar.data.data_fetcher import BaseDataFetcher

logger = logging.getLogger(__name__)

# Baseline list of highly-liquid US equities used as the seed universe.
# In a production system this would be fetched dynamically (e.g. from an
# exchange listing or a screener API).
_SEED_UNIVERSE: List[str] = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "TSLA", "BRK-B",
    "UNH", "LLY", "JPM", "V", "AVGO", "XOM", "PG", "MA", "HD", "COST", "JNJ",
    "MRK", "ABBV", "CVX", "CRM", "ORCL", "AMD", "NFLX", "KO", "PEP", "TMO",
    "ACN", "MCD", "BAC", "CSCO", "ADBE", "ABT", "PFE", "DIS", "WMT", "INTC",
    "QCOM", "DHR", "VZ", "NEE", "T", "TXN", "PM", "RTX", "SPGI", "UPS",
    "BMY", "AMGN", "MS", "GS", "CAT", "HON", "INTU", "NOW", "ISRG", "ELV",
    "LOW", "SYK", "BKNG", "VRTX", "GE", "DE", "REGN", "AXP", "ADI", "LRCX",
    "KLAC", "PANW", "AMAT", "GILD", "CI", "ADP", "ZTS", "SCHW", "SLB", "CME",
    "TJX", "CB", "BDX", "MMC", "EOG", "MO", "SBUX", "SO", "DUK", "CL",
    "NOC", "GD", "LMT", "WM", "ICE", "AON", "MDLZ", "CSX", "ITW", "FCX",
    "PLD", "SPG", "PSA", "AMT", "CCI", "EQIX", "O", "WPC", "VICI", "EXR",
    "F", "GM", "RIVN", "LCID", "NIO", "XPEV", "LI", "PLUG", "FCEL", "BLNK",
    "SQ", "PYPL", "COIN", "HOOD", "SOFI", "AFRM", "UPST", "OPEN", "OFFERPAD",
    "MSTR", "RIOT", "MARA", "HUT", "CLSK", "BTBT", "BITF", "CAN", "CIFR",
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLI", "XLP",
    "AAL", "UAL", "DAL", "LUV", "BA", "UBER", "LYFT", "ABNB", "DKNG", "PENN",
    "MGM", "WYNN", "LVS", "RCL", "CCL", "NCLH", "MAR", "HLT", "H", "IHG",
    "WFC", "C", "USB", "PNC", "TFC", "KEY", "RF", "FITB", "HBAN", "CFG",
    "CVS", "WBA", "MCK", "CAH", "COR", "HUM", "MOH", "CNC", "ANTM", "WCG",
    "MRNA", "BNTX", "NVAX", "VXRT", "INO", "OCGN", "ATOS", "SNDL", "ACB",
    "CGC", "TLRY", "CRON", "OGI", "APHA", "HEXO", "KERN", "CARA", "CLVS",
    "AMC", "GME", "BB", "EXPR", "KOSS", "NAKD", "BBBY", "WISH", "CLOV",
]


class UniverseBuilder:
    """Builds the stock scanning universe by applying liquidity and price filters.

    Args:
        fetcher: A :class:`~momentum_radar.data.data_fetcher.BaseDataFetcher`
            instance used to retrieve screening data.
    """

    def __init__(self, fetcher: BaseDataFetcher) -> None:
        self._fetcher = fetcher

    def build(self) -> List[str]:
        """Return a filtered list of ticker symbols ready for scanning.

        Applies the following filters from :attr:`~momentum_radar.config.config`:
        - Price > ``min_price``
        - 30-day average daily volume > ``min_avg_volume``
        - Exclude OTC / pink-sheet stocks (no ``"."`` in symbol)
        - Limit to ``universe_size`` symbols

        Returns:
            List of filtered ticker symbols.
        """
        cfg = config.universe
        candidates = list(dict.fromkeys(_SEED_UNIVERSE))  # deduplicate, preserve order
        filtered: List[str] = []

        for ticker in candidates:
            if len(filtered) >= cfg.universe_size:
                break
            if "." in ticker:
                logger.debug("Skipping OTC ticker %s", ticker)
                continue
            try:
                daily = self._fetcher.get_daily_bars(ticker, period="35d")
                if daily is None or daily.empty:
                    continue
                avg_volume = daily["volume"].mean()
                last_close = float(daily["close"].iloc[-1])
                if last_close < cfg.min_price:
                    logger.debug(
                        "Excluding %s: price %.2f < min %.2f",
                        ticker,
                        last_close,
                        cfg.min_price,
                    )
                    continue
                if avg_volume < cfg.min_avg_volume:
                    logger.debug(
                        "Excluding %s: avg_volume %.0f < min %d",
                        ticker,
                        avg_volume,
                        cfg.min_avg_volume,
                    )
                    continue
                filtered.append(ticker)
            except Exception as exc:
                logger.warning("Error screening %s: %s", ticker, exc)
                continue

        logger.info("Universe built: %d tickers", len(filtered))
        return filtered
