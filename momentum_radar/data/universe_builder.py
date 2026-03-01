"""
universe_builder.py – Builds and filters the stock scanning universe.

The universe is populated from a curated list of highly liquid US equities.
Filters are applied based on price and average daily volume thresholds defined
in :class:`~momentum_radar.config.UniverseConfig`.
"""

import logging
from typing import List

import pandas as pd

from momentum_radar.config import config
from momentum_radar.data.data_fetcher import BaseDataFetcher

logger = logging.getLogger(__name__)

_SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_NASDAQ100_WIKI_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"


def fetch_index_constituents() -> List[str]:
    """Fetch current S&P 500 and NASDAQ-100 tickers from Wikipedia.

    Returns a deduplicated list of ticker symbols.  Falls back to
    :data:`_SEED_UNIVERSE` if the S&P 500 page is unreachable or cannot be
    parsed.  Returns S&P 500 tickers only if the NASDAQ-100 page fails.

    Returns:
        List of ticker symbols from the two indices, or the static seed
        universe on failure.
    """
    tickers: List[str] = []
    try:
        sp500_tables = pd.read_html(_SP500_WIKI_URL, attrs={"id": "constituents"})
        sp500_df = sp500_tables[0]
        # Column is "Symbol" on the S&P 500 Wikipedia page
        sp500_tickers = sp500_df["Symbol"].dropna().tolist()
        # Wikipedia uses "." for BRK.B / BRK.A; yfinance uses "-"
        sp500_tickers = [t.replace(".", "-") for t in sp500_tickers]
        tickers.extend(sp500_tickers)
        logger.info("Fetched %d S&P 500 tickers from Wikipedia", len(sp500_tickers))
    except Exception as exc:
        logger.warning("Failed to fetch S&P 500 constituents from Wikipedia: %s", exc)
        return _SEED_UNIVERSE

    try:
        ndx_tables = pd.read_html(_NASDAQ100_WIKI_URL, attrs={"id": "constituents"})
        ndx_df = ndx_tables[0]
        # Column is "Ticker" on the NASDAQ-100 Wikipedia page
        ndx_tickers = ndx_df["Ticker"].dropna().tolist()
        tickers.extend(ndx_tickers)
        logger.info("Fetched %d NASDAQ-100 tickers from Wikipedia", len(ndx_tickers))
    except Exception as exc:
        logger.warning(
            "Failed to fetch NASDAQ-100 constituents from Wikipedia: %s", exc
        )

    if not tickers:
        logger.warning("No tickers fetched from Wikipedia; using seed universe")
        return _SEED_UNIVERSE

    # Deduplicate while preserving order
    return list(dict.fromkeys(tickers))


# Baseline list of highly-liquid US equities used as the seed universe.
# In a production system this would be fetched dynamically (e.g. from an
# exchange listing or a screener API).
_SEED_UNIVERSE: List[str] = [
    # --- S&P 500 core (mega/large cap) ---
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
    "PLD", "SPG", "PSA", "AMT", "CCI", "EQIX", "O", "VICI", "EXR",
    "WFC", "C", "USB", "PNC", "TFC", "KEY", "RF", "FITB", "HBAN", "CFG",
    "CVS", "MCK", "CAH", "COR", "HUM", "MOH", "CNC",
    "MRNA", "BNTX", "NVAX", "CGC", "TLRY", "CRON", "ACB",
    "AMC", "GME", "BB", "KOSS",
    # --- S&P 500 additional ---
    "APD", "AIZ", "ALB", "ARE", "AES", "AFL", "A", "ATO", "AWK", "AMP",
    "ABC", "AME", "APTV", "ACGL", "ANET", "AIG", "ARW", "AOS", "APA",
    "ALLE", "LNT", "AWK", "AZO", "BKR", "BALL", "BAX", "BDX", "BEN",
    "BBY", "BIO", "TECH", "BIIB", "BLK", "BK", "BA", "BWA", "CDNS",
    "CZR", "CPT", "CPB", "COF", "CF", "CRL", "CTAS", "CTXS", "CLX",
    "CMI", "CTSH", "COO", "ED", "GLW", "CPRT", "COP", "CTRA", "DXCM",
    "DVA", "DVN", "DECK", "DLR", "DFS", "DG", "DLTR", "D", "DOV",
    "DOW", "DTE", "DRE", "DD", "DPZ", "EBAY", "ECL", "EIX", "EA",
    "EMN", "ETN", "ENPH", "EQT", "EFX", "EVRG", "ES", "EXC", "EXPD",
    "EXR", "XOM", "FFIV", "FDS", "FAST", "FRT", "FIS", "FITB", "FLT",
    "FMC", "F", "FOXA", "FOX", "FTV", "FTNT", "GPC", "GRMN", "IT",
    "GIS", "GL", "GPN", "HAL", "HIG", "HAS", "HCA", "PEAK", "HSIC",
    "HES", "HPQ", "HOLX", "HRL", "HST", "HPE", "HUBB", "HII", "IBM",
    "IEX", "IDXX", "ITW", "ILMN", "INCY", "IP", "IPG", "IFF", "IPGP",
    "IRM", "JBHT", "JKHY", "JCI", "JNPR", "KSU", "K", "KMB", "KIM",
    "KMI", "KLAC", "KR", "LH", "LMT", "L", "LNC", "LEN", "LIN",
    "LYV", "LKQ", "LYB", "MTB", "MRO", "MPC", "MKTX", "MAR", "MMC",
    "MLM", "MAS", "MKC", "MXIM", "MET", "MTD", "MGM", "MCHP", "MU",
    "MSCI", "NDAQ", "NTAP", "NWSA", "NWS", "NEE", "NKE", "NI", "NSC",
    "NTRS", "NOC", "NLOK", "NRG", "NUE", "NVDA", "NVR", "NXPI", "ORLY",
    "OXY", "ODFL", "OMC", "ON", "OKE", "OTIS", "PKG", "PARA", "PH",
    "PAYX", "PAYC", "PYPL", "PNR", "PBCT", "PFG", "PCG", "PNW", "PPG",
    "PPL", "PRU", "PEG", "PTC", "PVH", "PWR", "QRVO", "RL", "RJF",
    "REG", "REGN", "RE", "RSG", "ROL", "ROK", "ROP", "RCL", "ROST",
    "RMD", "SWK", "SBAC", "SLB", "STX", "SEE", "SRE", "NOW", "SHW",
    "SIVB", "SNA", "SNPS", "SO", "SPG", "SPGI", "SYK", "SWKS", "SYY",
    "TMUS", "TROW", "TTWO", "TDY", "TPR", "TRMB", "TER", "TEL", "TDC",
    "TXT", "TFX", "HSY", "TRV", "TSCO", "TT", "TYL", "TSN", "UDR",
    "ULTA", "UAL", "UNP", "URI", "VLO", "VTR", "VRSN", "VRSK", "VFC",
    "VTRS", "V", "VNT", "VICI", "VMC", "WRB", "WAB", "WBA", "WMT",
    "WBD", "WM", "WAT", "WEC", "WFC", "WELL", "WST", "WDC", "WRK",
    "WY", "WHR", "WMB", "WYNN", "XEL", "XLNX", "XYL", "YUM", "ZBRA",
    "ZBH", "ZION", "ZTS",
    # --- NASDAQ 100 growth ---
    "ADSK", "ALGN", "ALXN", "ANSS", "ASML", "ATVI", "CDNS", "CERN",
    "CHKP", "CMCSA", "CPRT", "CSGP", "CSX", "CTSH", "DOCU", "DXCM",
    "EA", "EBAY", "FAST", "FISV", "FTNT", "IDXX", "ILMN", "JD", "KDP",
    "KLAC", "LBTYA", "LBTYK", "LCNB", "LRCX", "LULU", "MAR", "MELI",
    "MNST", "MRVL", "MSFT", "MU", "MXIM", "NTAP", "NTES", "NXPI",
    "OKTA", "PAYX", "PCAR", "PDD", "PTON", "QCOM", "REGN", "ROST",
    "SGEN", "SIRI", "SPLK", "SWKS", "TCOM", "TEAM", "TMUS", "TSLA",
    "TTWO", "TXN", "VRSK", "VRSN", "WBA", "WDAY", "XEL", "ZM", "ZS",
    # --- Top volume ETFs ---
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLI", "XLP",
    "XLU", "XLB", "XLRE", "XLC", "GLD", "SLV", "TLT", "HYG", "LQD",
    "EEM", "EFA", "VWO", "VEA", "GDX", "GDXJ", "USO", "UNG", "ARKK",
    "ARKG", "ARKW", "ARKF", "ARKQ", "TQQQ", "SQQQ", "SPXU", "UVXY",
    "VIXY", "SOXL", "SOXS", "LABU", "LABD",
    # --- Airlines / Travel / Leisure ---
    "AAL", "UAL", "DAL", "LUV", "UBER", "LYFT", "ABNB", "DKNG", "PENN",
    "MGM", "WYNN", "LVS", "RCL", "CCL", "NCLH", "MAR", "HLT", "H",
    # --- Fintech / Crypto-related ---
    "SQ", "PYPL", "COIN", "HOOD", "SOFI", "AFRM", "UPST",
    "MSTR", "RIOT", "MARA", "HUT", "CLSK", "BTBT", "BITF", "CAN", "CIFR",
    # --- EVs / Clean Energy ---
    "F", "GM", "RIVN", "LCID", "NIO", "XPEV", "LI", "PLUG", "BLNK",
    "CHPT", "EVGO", "FSR", "GOEV", "NKLA", "REE", "RIDE",
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
        candidates = fetch_index_constituents()  # dynamic; falls back to seed
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
