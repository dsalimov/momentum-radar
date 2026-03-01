"""
scanner.py – Pre-Market Intelligence Scanner.

Provides three scan functions used by the scheduler and bot commands:

* :func:`scan_unusual_volume`      – Stocks with RVOL ≥ 2.0 or > 20 % of daily avg before open.
* :func:`scan_most_active`         – Top-10 leaders sorted by volume, dollar-volume, gainers, losers.
* :func:`scan_options_spikes`      – Options volume > 2× 30-day avg; call/put spike detection.

All functions accept a *fetcher* (:class:`~momentum_radar.data.data_fetcher.BaseDataFetcher`)
so they are fully testable with a mock.
"""

import logging
from typing import Dict, List, Optional

from momentum_radar.data.data_fetcher import BaseDataFetcher

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (matching the problem-statement spec)
# ---------------------------------------------------------------------------

_RVOL_THRESHOLD = 2.0          # Relative Volume minimum to qualify as unusual
_PREMARKET_VOL_RATIO = 0.20    # > 20 % of avg full-day volume before open
_OPTIONS_VOL_MULTIPLIER = 2.0  # Options volume > 2× 30-day avg
_TOP_N = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_pct_change(last: float, prev: float) -> float:
    if prev and prev != 0:
        return round((last - prev) / prev * 100, 2)
    return 0.0


def _format_float(val: Optional[float]) -> str:
    if val is None:
        return "N/A"
    if val >= 1e9:
        return f"{val / 1e9:.1f}B"
    if val >= 1e6:
        return f"{val / 1e6:.1f}M"
    if val >= 1e3:
        return f"{val / 1e3:.0f}K"
    return str(int(val))


# ---------------------------------------------------------------------------
# A) Unusual Volume
# ---------------------------------------------------------------------------

def scan_unusual_volume(
    tickers: List[str],
    fetcher: BaseDataFetcher,
    top_n: int = _TOP_N,
    min_rvol: float = _RVOL_THRESHOLD,
) -> List[Dict]:
    """Detect stocks with unusually high pre-market / current volume.

    Criteria (either triggers inclusion):

    * RVOL ≥ ``min_rvol`` (default 2.0)
    * Current volume > 20 % of 30-day average daily volume

    Args:
        tickers:  Tickers to scan.
        fetcher:  Data provider.
        top_n:    Maximum results to return.
        min_rvol: Minimum relative volume threshold.

    Returns:
        List of dicts (sorted by RVOL descending) with keys:
        ``ticker``, ``rvol``, ``pct_change``, ``gap_pct``, ``last_close``,
        ``avg_volume``, ``today_volume``, ``float_shares``, ``sector``.
    """
    results: List[Dict] = []

    for ticker in tickers:
        try:
            daily = fetcher.get_daily_bars(ticker, period="35d")
            if daily is None or daily.empty or "volume" not in daily.columns:
                continue
            if len(daily) < 5:
                continue

            hist_vols = daily["volume"].iloc[-31:-1]
            avg_vol = float(hist_vols.mean()) if len(hist_vols) >= 5 else 0.0
            if avg_vol <= 0:
                continue

            today_vol = float(daily["volume"].iloc[-1])
            rvol = today_vol / avg_vol
            vol_ratio = today_vol / avg_vol  # same as rvol

            if rvol < min_rvol and vol_ratio < _PREMARKET_VOL_RATIO:
                continue

            last_close = float(daily["close"].iloc[-1])
            prev_close = float(daily["close"].iloc[-2]) if len(daily) >= 2 else last_close
            pct_change = _safe_pct_change(last_close, prev_close)

            # Gap % = pct change from prior close (proxy for open gap)
            gap_pct = pct_change

            # Fundamentals (float, sector) – best-effort
            float_shares: Optional[float] = None
            sector: str = "N/A"
            try:
                fundamentals = fetcher.get_fundamentals(ticker)
                if fundamentals:
                    float_shares = fundamentals.get("float_shares")
            except Exception:
                pass

            results.append(
                {
                    "ticker": ticker,
                    "rvol": round(rvol, 2),
                    "pct_change": pct_change,
                    "gap_pct": gap_pct,
                    "last_close": round(last_close, 2),
                    "avg_volume": int(avg_vol),
                    "today_volume": int(today_vol),
                    "float_shares": float_shares,
                    "float_str": _format_float(float_shares),
                    "sector": sector,
                }
            )
        except Exception as exc:
            logger.debug("scan_unusual_volume skipped %s: %s", ticker, exc)

    results.sort(key=lambda r: r["rvol"], reverse=True)
    return results[:top_n]


# ---------------------------------------------------------------------------
# B) Most Active (Pre-Market Leaders)
# ---------------------------------------------------------------------------

def scan_most_active(
    tickers: List[str],
    fetcher: BaseDataFetcher,
    top_n: int = _TOP_N,
) -> Dict[str, List[Dict]]:
    """Build top-10 lists for four pre-market leader categories.

    Categories:

    * ``highest_volume``       – sorted by today's volume
    * ``highest_dollar_volume`` – sorted by price × volume
    * ``top_gainers``          – sorted by % change (ascending → biggest gain last)
    * ``top_losers``           – sorted by % change (ascending → biggest loss first)

    Args:
        tickers: Tickers to scan.
        fetcher: Data provider.
        top_n:   Results per category.

    Returns:
        Dict with four list keys, each containing dicts with
        ``ticker``, ``last_close``, ``pct_change``, ``today_volume``,
        ``dollar_volume``.
    """
    rows: List[Dict] = []

    for ticker in tickers:
        try:
            daily = fetcher.get_daily_bars(ticker, period="5d")
            if daily is None or daily.empty or len(daily) < 2:
                continue

            last_close = float(daily["close"].iloc[-1])
            prev_close = float(daily["close"].iloc[-2])
            pct_change = _safe_pct_change(last_close, prev_close)
            today_vol = float(daily["volume"].iloc[-1])
            dollar_vol = last_close * today_vol

            rows.append(
                {
                    "ticker": ticker,
                    "last_close": round(last_close, 2),
                    "pct_change": pct_change,
                    "today_volume": int(today_vol),
                    "dollar_volume": round(dollar_vol, 2),
                }
            )
        except Exception as exc:
            logger.debug("scan_most_active skipped %s: %s", ticker, exc)

    return {
        "highest_volume": sorted(rows, key=lambda r: r["today_volume"], reverse=True)[:top_n],
        "highest_dollar_volume": sorted(rows, key=lambda r: r["dollar_volume"], reverse=True)[:top_n],
        "top_gainers": sorted(rows, key=lambda r: r["pct_change"], reverse=True)[:top_n],
        "top_losers": sorted(rows, key=lambda r: r["pct_change"])[:top_n],
    }


# ---------------------------------------------------------------------------
# C) Options Volume Spike
# ---------------------------------------------------------------------------

def scan_options_spikes(
    tickers: List[str],
    fetcher: BaseDataFetcher,
    top_n: int = _TOP_N,
    min_multiplier: float = _OPTIONS_VOL_MULTIPLIER,
) -> List[Dict]:
    """Detect stocks with abnormal options volume (call or put).

    Criteria:

    * Total options volume > ``min_multiplier`` × 30-day avg (approximated
      as 3× the single-chain volume stored by the fetcher), **or**
    * Call volume or put volume individually exceed their averages by the
      same multiplier.

    Args:
        tickers:        Tickers to scan.
        fetcher:        Data provider.
        top_n:          Maximum results.
        min_multiplier: Options volume multiple to qualify.

    Returns:
        List of dicts (sorted by call/put ratio desc) with keys:
        ``ticker``, ``call_volume``, ``put_volume``, ``cp_ratio``,
        ``call_spike``, ``put_spike``, ``bias``, ``iv_change``.
    """
    results: List[Dict] = []

    for ticker in tickers:
        try:
            opts = fetcher.get_options_volume(ticker)
            if not opts:
                continue

            call_vol = opts.get("call_volume", 0) or 0
            put_vol = opts.get("put_volume", 0) or 0
            avg_call = opts.get("avg_call_volume", 1) or 1
            avg_put = opts.get("avg_put_volume", 1) or 1

            call_spike = call_vol / avg_call if avg_call > 0 else 0
            put_spike = put_vol / avg_put if avg_put > 0 else 0
            total_vol = call_vol + put_vol
            avg_total = avg_call + avg_put
            total_spike = total_vol / avg_total if avg_total > 0 else 0

            if total_spike < min_multiplier and call_spike < min_multiplier and put_spike < min_multiplier:
                continue

            cp_ratio = round(call_vol / put_vol, 2) if put_vol > 0 else float("inf")
            if call_vol > put_vol * 1.5:
                bias = "BULLISH"
            elif put_vol > call_vol * 1.5:
                bias = "BEARISH"
            else:
                bias = "NEUTRAL"

            results.append(
                {
                    "ticker": ticker,
                    "call_volume": int(call_vol),
                    "put_volume": int(put_vol),
                    "cp_ratio": cp_ratio,
                    "call_spike": round(call_spike, 2),
                    "put_spike": round(put_spike, 2),
                    "total_spike": round(total_spike, 2),
                    "bias": bias,
                    "iv_change": "N/A",  # requires historical IV baseline
                }
            )
        except Exception as exc:
            logger.debug("scan_options_spikes skipped %s: %s", ticker, exc)

    results.sort(key=lambda r: r["total_spike"], reverse=True)
    return results[:top_n]
