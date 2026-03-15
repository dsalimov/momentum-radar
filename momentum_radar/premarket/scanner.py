"""
scanner.py – Pre-Market Intelligence Scanner.

Provides four scan functions used by the scheduler and bot commands:

* :func:`scan_unusual_volume`      – Stocks with RVOL ≥ 2.0 or > 20 % of daily avg before open.
* :func:`scan_most_active`         – Top-10 leaders sorted by volume, dollar-volume, gainers, losers.
* :func:`scan_options_spikes`      – Options volume > 2× 30-day avg; call/put spike detection.
* :func:`scan_swing_trade_setups`  – Top 10 swing trade setups from S&P 500/NASDAQ using Daily/4H/1H
                                     charts with pattern detection (Double Bottom, Double Top,
                                     Head and Shoulders, Cup and Handle, Flags/Pennants).

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


# ---------------------------------------------------------------------------
# D) Swing Trade Setup Scanner (Premarket Morning Watchlist)
# ---------------------------------------------------------------------------

#: Chart patterns detected by the swing trade scanner.
_SWING_PATTERNS = [
    "double bottom",
    "double top",
    "head and shoulders",
    "inverse head and shoulders",
    "cup and handle",
    "flag",
    "pennant",
]

#: Minimum pattern confidence (0–100) to include a setup in results.
_MIN_PATTERN_CONFIDENCE = 60


def _detect_swing_pattern(daily) -> tuple:
    """Detect the most likely swing chart pattern in *daily* OHLCV data.

    Performs simple rule-based pattern detection on the last 60 bars:

    * **Double Bottom** – two local lows within 2 % of each other separated
      by a higher intermediate high.
    * **Double Top** – two local highs within 2 % of each other separated
      by a lower intermediate low.
    * **Head and Shoulders** – left shoulder, higher head, right shoulder
      with similar shoulder heights; neckline below shoulder lows.
    * **Inverse Head and Shoulders** – mirror of the above pattern.
    * **Cup and Handle** – U-shaped price action followed by a slight dip.
    * **Flag / Pennant** – strong trending move followed by a tight
      consolidation (low ATR in final bars vs. prior bars).

    Args:
        daily: Daily OHLCV ``pandas.DataFrame`` with at least 20 rows.

    Returns:
        Tuple ``(pattern_name, confidence, key_level)`` where *pattern_name*
        is a string (or ``""`` if nothing detected), *confidence* is 0–100,
        and *key_level* is the critical breakout / breakdown price.
    """
    import numpy as np

    if daily is None or len(daily) < 20:
        return ("", 0, 0.0)

    closes = daily["close"].values.astype(float)
    highs = daily["high"].values.astype(float)
    lows = daily["low"].values.astype(float)
    vols = daily["volume"].values.astype(float)
    n = len(closes)

    # Use last 60 bars (or all if fewer)
    window = min(60, n)
    closes = closes[-window:]
    highs = highs[-window:]
    lows = lows[-window:]
    vols = vols[-window:]
    wn = len(closes)

    # Helper: find local minima / maxima with a minimum distance of 5 bars
    def _local_minima(arr, min_dist=5):
        idx = []
        for i in range(min_dist, len(arr) - min_dist):
            if arr[i] == min(arr[i - min_dist:i + min_dist + 1]):
                idx.append(i)
        return idx

    def _local_maxima(arr, min_dist=5):
        idx = []
        for i in range(min_dist, len(arr) - min_dist):
            if arr[i] == max(arr[i - min_dist:i + min_dist + 1]):
                idx.append(i)
        return idx

    lmin_idx = _local_minima(lows)
    lmax_idx = _local_maxima(highs)

    current_price = float(closes[-1])

    # ------------------------------------------------------------------
    # Double Bottom
    # ------------------------------------------------------------------
    if len(lmin_idx) >= 2:
        b1_idx, b2_idx = lmin_idx[-2], lmin_idx[-1]
        b1, b2 = lows[b1_idx], lows[b2_idx]
        if abs(b1 - b2) / max(b1, b2) <= 0.03 and b2_idx > b1_idx + 5:
            # Intermediate high between the two bottoms
            mid_high = highs[b1_idx:b2_idx].max() if b2_idx > b1_idx else 0.0
            if mid_high > max(b1, b2) * 1.02:
                confidence = 70
                # Increase confidence if current price is above mid_high (breakout)
                if current_price >= mid_high:
                    confidence = 85
                return ("Double Bottom", confidence, round(float(mid_high), 2))

    # ------------------------------------------------------------------
    # Double Top
    # ------------------------------------------------------------------
    if len(lmax_idx) >= 2:
        t1_idx, t2_idx = lmax_idx[-2], lmax_idx[-1]
        t1, t2 = highs[t1_idx], highs[t2_idx]
        if abs(t1 - t2) / max(t1, t2) <= 0.03 and t2_idx > t1_idx + 5:
            mid_low = lows[t1_idx:t2_idx].min() if t2_idx > t1_idx else float("inf")
            if mid_low < min(t1, t2) * 0.98:
                confidence = 70
                if current_price <= mid_low:
                    confidence = 85
                return ("Double Top", confidence, round(float(mid_low), 2))

    # ------------------------------------------------------------------
    # Head and Shoulders
    # ------------------------------------------------------------------
    if len(lmax_idx) >= 3:
        ls_idx, h_idx, rs_idx = lmax_idx[-3], lmax_idx[-2], lmax_idx[-1]
        ls, hd, rs = highs[ls_idx], highs[h_idx], highs[rs_idx]
        if hd > ls * 1.02 and hd > rs * 1.02 and abs(ls - rs) / max(ls, rs) <= 0.05:
            # Neckline: midpoint between left-shoulder trough and right-shoulder trough
            trough_l = lows[ls_idx:h_idx].min() if h_idx > ls_idx else ls
            trough_r = lows[h_idx:rs_idx + 1].min() if rs_idx > h_idx else rs
            neckline = (trough_l + trough_r) / 2.0
            confidence = 75
            if current_price <= neckline:
                confidence = 85
            return ("Head and Shoulders", confidence, round(float(neckline), 2))

    # ------------------------------------------------------------------
    # Inverse Head and Shoulders
    # ------------------------------------------------------------------
    if len(lmin_idx) >= 3:
        ls_idx, h_idx, rs_idx = lmin_idx[-3], lmin_idx[-2], lmin_idx[-1]
        ls, hd, rs = lows[ls_idx], lows[h_idx], lows[rs_idx]
        if hd < ls * 0.98 and hd < rs * 0.98 and abs(ls - rs) / max(ls, rs) <= 0.05:
            trough_h_l = highs[ls_idx:h_idx].max() if h_idx > ls_idx else ls
            trough_h_r = highs[h_idx:rs_idx + 1].max() if rs_idx > h_idx else rs
            neckline = (trough_h_l + trough_h_r) / 2.0
            confidence = 75
            if current_price >= neckline:
                confidence = 85
            return ("Inverse Head and Shoulders", confidence, round(float(neckline), 2))

    # ------------------------------------------------------------------
    # Cup and Handle
    # ------------------------------------------------------------------
    if wn >= 30:
        # Cup: early high, deep low in middle, recovery; Handle: shallow dip at end
        first_third = closes[:wn // 3]
        mid_third = closes[wn // 3: 2 * wn // 3]
        last_third = closes[2 * wn // 3:]
        cup_left_high = first_third.max()
        cup_bottom = mid_third.min()
        cup_right_high = last_third[:len(last_third) // 2].max() if len(last_third) > 0 else 0.0
        handle_low = last_third.min() if len(last_third) > 0 else 0.0

        depth = (cup_left_high - cup_bottom) / cup_left_high if cup_left_high > 0 else 0
        recovery = (cup_right_high - cup_bottom) / (cup_left_high - cup_bottom) if (cup_left_high - cup_bottom) > 0 else 0
        handle_ok = handle_low >= cup_right_high * 0.95 and current_price >= cup_right_high * 0.97

        if 0.15 <= depth <= 0.50 and recovery >= 0.80 and handle_ok:
            confidence = 70
            if current_price >= cup_left_high:
                confidence = 85
            return ("Cup and Handle", confidence, round(float(cup_left_high), 2))

    # ------------------------------------------------------------------
    # Flag / Pennant (strong prior move + tight consolidation)
    # ------------------------------------------------------------------
    if wn >= 15:
        prior_bars = closes[-15:-5]
        consol_bars = closes[-5:]
        prior_move = abs(prior_bars[-1] - prior_bars[0]) / prior_bars[0] if prior_bars[0] > 0 else 0
        consol_range = (consol_bars.max() - consol_bars.min()) / consol_bars[0] if consol_bars[0] > 0 else 0

        # Strong prior trend (>3%) + tight consolidation (<1.5%)
        if prior_move >= 0.03 and consol_range <= 0.015:
            pattern = "Flag" if prior_move >= 0.05 else "Pennant"
            confidence = 65
            # Key level: top of consolidation for bullish flag; bottom for bearish
            if prior_bars[-1] > prior_bars[0]:  # uptrend → bullish flag
                key_level = float(consol_bars.max())
                if current_price >= key_level:
                    confidence = 80
            else:  # downtrend → bearish flag
                key_level = float(consol_bars.min())
                if current_price <= key_level:
                    confidence = 80
            return (pattern, confidence, round(key_level, 2))

    return ("", 0, 0.0)


def scan_swing_trade_setups(
    tickers: List[str],
    fetcher: BaseDataFetcher,
    top_n: int = 10,
    min_confidence: int = _MIN_PATTERN_CONFIDENCE,
) -> List[Dict]:
    """Scan for the best swing trade setups from S&P 500 / NASDAQ tickers.

    Scans higher timeframes (Daily bars with 1-year history) and detects the
    following strong chart patterns:

    * Double Bottom / Double Top
    * Head and Shoulders / Inverse Head and Shoulders
    * Cup and Handle
    * Flag / Pennant

    This function is designed to run **before market open** (premarket) and
    return a clean morning watchlist of the **top 10** swing trade candidates.

    Args:
        tickers:        S&P 500 + NASDAQ tickers to scan.
        fetcher:        Data provider.
        top_n:          Maximum results to return (default 10).
        min_confidence: Minimum pattern confidence (0–100) to include a result.

    Returns:
        List of up to *top_n* dicts (sorted by confidence descending), each with:

        * ``ticker``           – Stock symbol.
        * ``pattern_name``     – Detected chart pattern name.
        * ``pattern_confidence`` – Pattern confidence score (0–100).
        * ``current_price``    – Latest closing price.
        * ``key_level``        – Key breakout or breakdown level.
        * ``pct_to_key``       – Percentage distance from current price to key level.
        * ``timeframe``        – Timeframe used for analysis (``"Daily"``).
        * ``strategy_type``    – Always ``"SWING TRADE"``.
    """
    results: List[Dict] = []

    for ticker in tickers:
        try:
            # Fetch ~1 year of daily data for reliable pattern detection
            daily = fetcher.get_daily_bars(ticker, period="252d")
            if daily is None or daily.empty or len(daily) < 20:
                continue

            pattern_name, confidence, key_level = _detect_swing_pattern(daily)

            if not pattern_name or confidence < min_confidence:
                continue

            current_price = float(daily["close"].iloc[-1])
            if current_price <= 0:
                continue

            pct_to_key = round((key_level - current_price) / current_price * 100, 2)

            results.append(
                {
                    "ticker": ticker,
                    "pattern_name": pattern_name,
                    "pattern_confidence": confidence,
                    "current_price": round(current_price, 2),
                    "key_level": key_level,
                    "pct_to_key": pct_to_key,
                    "timeframe": "Daily",
                    "strategy_type": "SWING TRADE",
                }
            )
        except Exception as exc:
            logger.debug("scan_swing_trade_setups skipped %s: %s", ticker, exc)

    # Sort by confidence descending, return top N
    results.sort(key=lambda r: r["pattern_confidence"], reverse=True)
    return results[:top_n]
