"""
briefing.py – Daily Market Brief Generator.

Assembles a "Market Structure Update" sent automatically each morning:

* Index trend (SPY / QQQ % change)
* VIX regime estimate (high / normal / low)
* Sector rotation summary
* Top short-squeeze candidates
* Top unusual-volume stocks
* Options flow leaders

The main entry point is :func:`generate_market_brief` which returns a
formatted multi-line string ready for Telegram or console output.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import pytz

from momentum_radar.data.data_fetcher import BaseDataFetcher
from momentum_radar.premarket.scanner import (
    scan_unusual_volume,
    scan_most_active,
    scan_options_spikes,
)
from momentum_radar.premarket.squeeze_detector import scan_squeeze_candidates

logger = logging.getLogger(__name__)

EST = pytz.timezone("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _arrow(pct: float) -> str:
    return "▲" if pct >= 0 else "▼"


def _pct_str(pct: Optional[float]) -> str:
    if pct is None:
        return "N/A"
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


def _get_index_context(fetcher: BaseDataFetcher) -> Dict:
    """Fetch SPY and QQQ quotes and return a context dict."""
    out: Dict = {}
    for sym in ("SPY", "QQQ", "VIX"):
        try:
            quote = fetcher.get_quote(sym)
            if quote and quote.get("price") and quote.get("prev_close"):
                price = float(quote["price"])
                prev = float(quote["prev_close"])
                pct = round((price - prev) / prev * 100, 2) if prev else 0.0
                out[sym] = {"price": round(price, 2), "pct": pct}
        except Exception as exc:
            logger.debug("Could not fetch %s: %s", sym, exc)
    return out


def _vix_regime(vix_price: Optional[float]) -> str:
    if vix_price is None:
        return "Unknown"
    if vix_price >= 30:
        return "FEAR (>30) – Elevated risk"
    if vix_price >= 20:
        return "ELEVATED (20–30) – Caution"
    return "CALM (<20) – Normal risk"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_market_brief(
    tickers: List[str],
    fetcher: BaseDataFetcher,
    session_label: str = "Pre-Market",
) -> str:
    """Generate a professional daily market brief.

    Args:
        tickers:       Universe of tickers to scan.
        fetcher:       Data provider.
        session_label: Label shown in the header (e.g. "Pre-Market 4:05 AM").

    Returns:
        Formatted multi-line string.
    """
    now_est = datetime.now(tz=EST)
    timestamp = now_est.strftime("%Y-%m-%d %H:%M ET")

    lines: List[str] = [
        f"MARKET INTELLIGENCE BRIEF – {session_label}",
        f"Generated: {timestamp}",
        "",
        "=" * 40,
        "MARKET STRUCTURE UPDATE",
        "=" * 40,
    ]

    # ---- Index context ----
    indices = _get_index_context(fetcher)
    for sym in ("SPY", "QQQ"):
        info = indices.get(sym)
        if info:
            arr = _arrow(info["pct"])
            lines.append(f"  {sym}: ${info['price']} {arr} {_pct_str(info['pct'])}")
        else:
            lines.append(f"  {sym}: N/A")

    vix_info = indices.get("VIX")
    vix_price = vix_info["price"] if vix_info else None
    lines.append(f"  VIX: {vix_price if vix_price else 'N/A'} – {_vix_regime(vix_price)}")
    lines.append("")

    # ---- Unusual volume ----
    lines.append("TOP UNUSUAL VOLUME")
    lines.append("-" * 30)
    try:
        vol_spikes = scan_unusual_volume(tickers, fetcher, top_n=5)
        if vol_spikes:
            for s in vol_spikes:
                arr = _arrow(s["pct_change"])
                lines.append(
                    f"  {s['ticker']:6s}  RVOL {s['rvol']:.1f}x  "
                    f"{arr}{abs(s['pct_change']):.1f}%  "
                    f"${s['last_close']}"
                )
        else:
            lines.append("  No unusual volume detected.")
    except Exception as exc:
        lines.append(f"  Scan unavailable: {exc}")
    lines.append("")

    # ---- Most active ----
    lines.append("MOST ACTIVE – TOP GAINERS / LOSERS")
    lines.append("-" * 30)
    try:
        active = scan_most_active(tickers, fetcher, top_n=5)
        gainers = active.get("top_gainers", [])
        losers = active.get("top_losers", [])
        if gainers:
            lines.append("  Gainers:")
            for g in gainers[:5]:
                lines.append(f"    {g['ticker']:6s}  +{g['pct_change']:.1f}%  ${g['last_close']}")
        if losers:
            lines.append("  Losers:")
            for l in losers[:5]:
                lines.append(f"    {l['ticker']:6s}  {l['pct_change']:.1f}%  ${l['last_close']}")
    except Exception as exc:
        lines.append(f"  Scan unavailable: {exc}")
    lines.append("")

    # ---- Options flow leaders ----
    lines.append("OPTIONS FLOW LEADERS")
    lines.append("-" * 30)
    try:
        opt_spikes = scan_options_spikes(tickers, fetcher, top_n=5)
        if opt_spikes:
            for o in opt_spikes:
                lines.append(
                    f"  {o['ticker']:6s}  C/P {o['cp_ratio']}  "
                    f"Calls {o['call_volume']:,} ({o['call_spike']}x)  "
                    f"Puts {o['put_volume']:,}  Bias: {o['bias']}"
                )
        else:
            lines.append("  No significant options spikes detected.")
    except Exception as exc:
        lines.append(f"  Scan unavailable: {exc}")
    lines.append("")

    # ---- Short squeeze candidates ----
    lines.append("TOP SQUEEZE CANDIDATES")
    lines.append("-" * 30)
    try:
        squeeze_tickers = tickers[:100]  # limit to avoid long scans
        candidates = scan_squeeze_candidates(squeeze_tickers, fetcher, min_score=30, top_n=5)
        if candidates:
            for c in candidates:
                si_str = f"{c['short_interest_pct']:.1%}" if c.get("short_interest_pct") is not None else "N/A"
                rvol_str = f"  RVOL {c['rvol']}x" if c.get("rvol") else ""
                lines.append(
                    f"  {c['ticker']:6s}  Score {c['squeeze_score']}%  "
                    f"SI {si_str}  Float {c.get('float_str', 'N/A')}{rvol_str}"
                )
        else:
            lines.append("  No squeeze candidates above threshold.")
    except Exception as exc:
        lines.append(f"  Scan unavailable: {exc}")

    lines.append("")
    lines.append("=" * 40)
    lines.append("End of brief. Trade responsibly.")

    return "\n".join(lines)
