"""
squeeze_detector.py – Short Squeeze Detection Engine.

Calculates a ``squeeze_score`` (0–100) and returns a structured full report
for any squeeze candidate.

Score weighting (per problem-statement spec):

+--------------------------------------+--------+
| Factor                               | Points |
+======================================+========+
| Short interest > 20 %                |   +20  |
| Days to cover > 3                    |   +15  |
| Float < 50 M                         |   +15  |
| Call volume spike (> 2× avg)         |   +15  |
| Breakout / bullish structure pattern |   +15  |
| Gamma ramp detected (CP ratio ≥ 2)   |   +10  |
| Volume expansion > 3×                |   +10  |
+--------------------------------------+--------+

Total max = 100.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

from momentum_radar.data.data_fetcher import BaseDataFetcher

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Score weights
# ---------------------------------------------------------------------------

_W_SHORT_INTEREST = 20   # short interest > 20 %
_W_DAYS_TO_COVER = 15    # days-to-cover > 3
_W_LOW_FLOAT = 15        # float < 50 M shares
_W_CALL_SPIKE = 15       # call volume > 2× avg
_W_BREAKOUT = 15         # breakout / bullish technical structure
_W_GAMMA_RAMP = 10       # call/put ratio ≥ 2
_W_VOL_EXPANSION = 10    # RVOL > 3×


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def compute_squeeze_score(
    short_pct: Optional[float],
    days_to_cover: Optional[float],
    float_shares: Optional[float],
    call_spike: Optional[float],
    rvol: Optional[float],
    cp_ratio: Optional[float],
    breakout: bool = False,
) -> Dict:
    """Compute a squeeze probability score (0–100).

    Args:
        short_pct:      Short interest as a decimal (e.g. 0.25 = 25 %).
        days_to_cover:  Days-to-cover (short ratio).
        float_shares:   Float size in shares.
        call_spike:     Call volume / 30-day avg call volume.
        rvol:           Relative volume.
        cp_ratio:       Call/put ratio.
        breakout:       ``True`` if a bullish breakout pattern is detected.

    Returns:
        Dict with keys ``score``, ``factors``, ``probability_pct``, ``label``.
    """
    score = 0
    factors: List[str] = []

    if short_pct is not None and short_pct > 0.20:
        score += _W_SHORT_INTEREST
        factors.append(f"Short interest {short_pct:.1%} > 20% (+{_W_SHORT_INTEREST})")

    if days_to_cover is not None and days_to_cover > 3:
        score += _W_DAYS_TO_COVER
        factors.append(f"Days to cover {days_to_cover:.1f} > 3 (+{_W_DAYS_TO_COVER})")

    if float_shares is not None and float_shares < 50_000_000:
        score += _W_LOW_FLOAT
        factors.append(f"Float {float_shares / 1e6:.1f}M < 50M (+{_W_LOW_FLOAT})")

    if call_spike is not None and call_spike > 2.0:
        score += _W_CALL_SPIKE
        factors.append(f"Call volume spike {call_spike:.1f}x (+{_W_CALL_SPIKE})")

    if breakout:
        score += _W_BREAKOUT
        factors.append(f"Breakout / bullish structure detected (+{_W_BREAKOUT})")

    if cp_ratio is not None and cp_ratio >= 2.0:
        score += _W_GAMMA_RAMP
        factors.append(f"Gamma ramp: C/P ratio {cp_ratio:.1f} ≥ 2 (+{_W_GAMMA_RAMP})")

    if rvol is not None and rvol > 3.0:
        score += _W_VOL_EXPANSION
        factors.append(f"Volume expansion {rvol:.1f}x > 3x (+{_W_VOL_EXPANSION})")

    score = min(score, 100)

    if score >= 70:
        label = "HIGH – Strong squeeze candidate"
    elif score >= 45:
        label = "MEDIUM – Moderate squeeze setup"
    elif score >= 20:
        label = "LOW – Weak squeeze potential"
    else:
        label = "NONE – Insufficient criteria"

    return {
        "score": score,
        "probability_pct": score,
        "label": label,
        "factors": factors,
    }


def build_squeeze_report(ticker: str, fetcher: BaseDataFetcher) -> Optional[Dict]:
    """Fetch all required data for *ticker* and return a full squeeze report.

    Args:
        ticker:  Stock symbol.
        fetcher: Data provider.

    Returns:
        Dict with all squeeze data fields, or ``None`` if data is unavailable.
    """
    try:
        daily = fetcher.get_daily_bars(ticker, period="60d")
        fundamentals = fetcher.get_fundamentals(ticker)
        options = fetcher.get_options_volume(ticker)
        quote = fetcher.get_quote(ticker)
    except Exception as exc:
        logger.error("build_squeeze_report: data fetch failed for %s: %s", ticker, exc)
        return None

    if daily is None or daily.empty:
        return None

    # ---- Price info ----
    current_price: float = 0.0
    prev_close: float = 0.0
    if quote:
        current_price = float(quote.get("price") or 0.0)
        prev_close = float(quote.get("prev_close") or 0.0)

    # ---- Fundamentals ----
    short_pct: Optional[float] = None
    days_to_cover: Optional[float] = None
    float_shares: Optional[float] = None
    if fundamentals:
        raw_si = fundamentals.get("short_percent_of_float")
        short_pct = float(raw_si) if raw_si is not None else None
        raw_dtc = fundamentals.get("short_ratio")
        days_to_cover = float(raw_dtc) if raw_dtc is not None else None
        raw_float = fundamentals.get("float_shares")
        float_shares = float(raw_float) if raw_float is not None else None

    # ---- Volume / RVOL ----
    rvol: Optional[float] = None
    try:
        from momentum_radar.utils.indicators import compute_rvol
        intraday = fetcher.get_intraday_bars(ticker, interval="1m", period="1d")
        rvol = compute_rvol(intraday, daily)
    except Exception:
        pass

    # Fallback RVOL from daily bars
    if rvol is None and len(daily) >= 5:
        lookback = daily["volume"].iloc[-31:-1] if len(daily) >= 31 else daily["volume"].iloc[:-1]
        avg_vol = float(lookback.mean())
        today_vol = float(daily["volume"].iloc[-1])
        rvol = today_vol / avg_vol if avg_vol > 0 else None

    # ---- Options ----
    call_volume: int = 0
    put_volume: int = 0
    avg_call: float = 1.0
    avg_put: float = 1.0
    if options:
        call_volume = int(options.get("call_volume", 0) or 0)
        put_volume = int(options.get("put_volume", 0) or 0)
        avg_call = float(options.get("avg_call_volume", 1) or 1)
        avg_put = float(options.get("avg_put_volume", 1) or 1)

    call_spike = call_volume / avg_call if avg_call > 0 else None
    cp_ratio = round(call_volume / put_volume, 2) if put_volume > 0 else None

    # ---- Technical structure ----
    breakout = _detect_breakout(daily)
    breakout_level = _get_breakout_level(daily)
    resistance = _get_resistance(daily)

    # ---- Squeeze score ----
    sq = compute_squeeze_score(
        short_pct=short_pct,
        days_to_cover=days_to_cover,
        float_shares=float_shares,
        call_spike=call_spike,
        rvol=rvol,
        cp_ratio=cp_ratio,
        breakout=breakout,
    )

    # ---- ATR for scenario targets ----
    atr: Optional[float] = None
    try:
        from momentum_radar.utils.indicators import compute_atr
        atr = compute_atr(daily)
    except Exception:
        pass

    bull_target1 = round(current_price + (atr * 2), 2) if atr and current_price else None
    bull_target2 = round(current_price + (atr * 4), 2) if atr and current_price else None
    bear_target = round(current_price - (atr * 2), 2) if atr and current_price else None

    return {
        "ticker": ticker,
        "current_price": round(current_price, 2),
        "short_interest_pct": short_pct,
        "days_to_cover": days_to_cover,
        "float_shares": float_shares,
        "float_str": _fmt_float(float_shares),
        "rvol": round(rvol, 2) if rvol is not None else None,
        "call_volume": call_volume,
        "put_volume": put_volume,
        "cp_ratio": cp_ratio,
        "call_spike": round(call_spike, 2) if call_spike is not None else None,
        "breakout": breakout,
        "breakout_level": breakout_level,
        "resistance": resistance,
        "squeeze_score": sq["score"],
        "squeeze_probability_pct": sq["probability_pct"],
        "squeeze_label": sq["label"],
        "squeeze_factors": sq["factors"],
        "bull_target1": bull_target1,
        "bull_target2": bull_target2,
        "bear_target": bear_target,
        "atr": round(atr, 2) if atr else None,
    }


def scan_squeeze_candidates(
    tickers: List[str],
    fetcher: BaseDataFetcher,
    min_score: int = 30,
    top_n: int = 10,
) -> List[Dict]:
    """Scan *tickers* and return those with squeeze score ≥ *min_score*.

    Args:
        tickers:   Tickers to scan.
        fetcher:   Data provider.
        min_score: Minimum squeeze score (0–100).
        top_n:     Maximum results.

    Returns:
        List of squeeze report dicts, sorted by score descending.
    """
    results: List[Dict] = []
    for ticker in tickers:
        try:
            report = build_squeeze_report(ticker, fetcher)
            if report and report["squeeze_score"] >= min_score:
                results.append(report)
        except Exception as exc:
            logger.debug("scan_squeeze_candidates skipped %s: %s", ticker, exc)

    results.sort(key=lambda r: r["squeeze_score"], reverse=True)
    return results[:top_n]


# ---------------------------------------------------------------------------
# Text formatting helpers
# ---------------------------------------------------------------------------

def format_squeeze_report(report: Dict) -> str:
    """Render a squeeze report dict as a formatted text message.

    Args:
        report: Dict returned by :func:`build_squeeze_report`.

    Returns:
        Multi-line string suitable for Telegram or console output.
    """
    ticker = report["ticker"]
    score = report["squeeze_score"]
    label = report["squeeze_label"]

    si_str = f"{report['short_interest_pct']:.1%}" if report.get("short_interest_pct") is not None else "N/A"
    dtc_str = f"{report['days_to_cover']:.1f}" if report.get("days_to_cover") is not None else "N/A"
    rvol_str = f"{report['rvol']:.1f}x" if report.get("rvol") is not None else "N/A"
    cp_str = f"{report['cp_ratio']:.2f}" if report.get("cp_ratio") is not None else "N/A"
    bs_str = "YES" if report.get("breakout") else "NO"
    brk_lvl = f"${report['breakout_level']:.2f}" if report.get("breakout_level") else "N/A"
    res_lvl = f"${report['resistance']:.2f}" if report.get("resistance") else "N/A"

    bt1 = f"${report['bull_target1']:.2f}" if report.get("bull_target1") else "N/A"
    bt2 = f"${report['bull_target2']:.2f}" if report.get("bull_target2") else "N/A"
    bear = f"${report['bear_target']:.2f}" if report.get("bear_target") else "N/A"

    lines = [
        f"SQUEEZE ALERT: {ticker}",
        "",
        f"Short Interest: {si_str}",
        f"Float: {report.get('float_str', 'N/A')}",
        f"Days to Cover: {dtc_str}",
        f"Cost to Borrow: N/A",
        f"Utilization: N/A",
        f"Recent Volume vs Avg: {rvol_str}",
        f"Call/Put Ratio: {cp_str}",
        f"Top Call Strike: N/A",
        f"Gamma Wall: N/A",
        f"Breakout Level: {brk_lvl}",
        f"Resistance Above: {res_lvl}",
        f"Liquidity Void Above: N/A",
        "",
        "Technical Structure",
        f"  Breakout detected: {bs_str}",
        f"  ATR: {'$' + str(report['atr']) if report.get('atr') else 'N/A'}",
        f"  Invalidation: {bear}",
        "",
        "Scenario Projection",
        "  Bull Case:",
        f"    Target 1: {bt1}",
        f"    Target 2: {bt2}",
        "  Bear Case:",
        f"    Breakdown level: {bear}",
        "",
        f"Squeeze Probability: {score}%",
        f"{label}",
    ]

    if report.get("squeeze_factors"):
        lines.append("")
        lines.append("Score Breakdown:")
        for f in report["squeeze_factors"]:
            lines.append(f"  {f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal technical helpers
# ---------------------------------------------------------------------------

def _detect_breakout(daily: pd.DataFrame) -> bool:
    """Simple breakout detection: last close > 20-day high of prior bars."""
    if daily is None or len(daily) < 22:
        return False
    recent_high = float(daily["high"].iloc[-22:-1].max())
    last_close = float(daily["close"].iloc[-1])
    return last_close > recent_high


def _get_breakout_level(daily: pd.DataFrame) -> Optional[float]:
    """Return the 20-day prior high (breakout reference level)."""
    if daily is None or len(daily) < 22:
        return None
    return round(float(daily["high"].iloc[-22:-1].max()), 2)


def _get_resistance(daily: pd.DataFrame) -> Optional[float]:
    """Return the 52-week high as a simple resistance estimate."""
    if daily is None or daily.empty:
        return None
    return round(float(daily["high"].max()), 2)


def _fmt_float(val: Optional[float]) -> str:
    if val is None:
        return "N/A"
    if val >= 1e9:
        return f"{val / 1e9:.1f}B"
    if val >= 1e6:
        return f"{val / 1e6:.1f}M"
    return str(int(val))
