"""
structure_supply_demand_engine.py – Institutional Supply & Demand Zone Engine.

This is the top-level orchestration module for supply and demand zone
detection.  It ties together:

1. Zone detection  (:mod:`momentum_radar.signals.supply_demand`)
2. Zone persistence (:mod:`momentum_radar.storage.zone_store`)
3. Third-touch logic and break-of-zone detection
4. Probability contribution to the alert engine

Public API
----------
- :func:`scan_ticker`        – full S&D analysis for a single ticker
- :func:`format_zone_alert`  – render a human-readable zone alert string
- :func:`get_zone_score_bonus` – probability bonus (0–15) to inject into
                                  the confidence engine

Example output (Telegram)::

    🟢 DEMAND ZONE ALERT — AAPL
    ════════════════════════════════
    Zone: Strong Daily Demand  [147.20 – 149.85]
    Status: Fresh (0 tests)
    Strength: 78/100 (Strong)
    Impulse: 2.1× ATR  |  Volume: 1.8× avg

    Price: $148.35  (inside zone)

    📐 Third-Touch Setup detected!
    Prior reactions: 2  |  Win rate: 67%
    Entry: $148.35  Stop: $146.90  Target: $151.80
    R:R: 2.3  Confidence: 74%

Usage::

    from momentum_radar.structure_supply_demand_engine import scan_ticker

    result = scan_ticker("AAPL", fetcher)
    if result["has_active_zone"]:
        print(format_zone_alert(result))
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from momentum_radar.signals.supply_demand import (
    SupplyDemandZone,
    detect_zones,
    get_active_zone,
)
from momentum_radar.storage.alert_stats import get_win_rate
from momentum_radar.storage.zone_store import load_zones, upsert_zone
from momentum_radar.utils.indicators import compute_atr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum zone strength score to be reported in scan results
_MIN_REPORT_SCORE: float = 50.0
# Minimum strength for a zone to contribute to the alert score
_MIN_ALERT_SCORE: float = 60.0
# Number of prior reactions required for a "third-touch" alert
_THIRD_TOUCH_MIN_REACTIONS: int = 2

# Stop-loss ATR multipliers by zone type
_STOP_ATR_MULT: float = 1.5
# Risk:Reward target multiplier
_TARGET_RR: float = 2.5


# ---------------------------------------------------------------------------
# Core scan function
# ---------------------------------------------------------------------------

def scan_ticker(
    ticker: str,
    fetcher=None,
    daily: Optional[pd.DataFrame] = None,
    bars: Optional[pd.DataFrame] = None,
) -> Dict:
    """Run a full supply & demand analysis for *ticker*.

    Supply the bars directly (``daily`` / ``bars``) **or** a *fetcher*
    object that has ``get_daily_bars`` and ``get_intraday_bars`` methods.
    Direct DataFrames take priority over the fetcher.

    Args:
        ticker:  Stock symbol.
        fetcher: Optional data provider instance.
        daily:   Daily OHLCV DataFrame (overrides fetcher).
        bars:    Intraday 1-min OHLCV DataFrame (overrides fetcher).

    Returns:
        Result dict with keys:

        - ``ticker``
        - ``has_active_zone`` (bool)
        - ``active_zone`` (:class:`SupplyDemandZone` or ``None``)
        - ``all_zones`` (list)
        - ``third_touch_setup`` (bool)
        - ``entry``, ``stop``, ``target``, ``rr``
        - ``confidence_pct``
        - ``win_rate_pct``
        - ``zone_score_bonus`` (0–15, for injection into probability engine)
        - ``alert_text`` (formatted string, empty when no active zone)
    """
    result: Dict = {
        "ticker": ticker,
        "has_active_zone": False,
        "active_zone": None,
        "all_zones": [],
        "third_touch_setup": False,
        "entry": None,
        "stop": None,
        "target": None,
        "rr": None,
        "confidence_pct": 0.0,
        "win_rate_pct": 0.0,
        "zone_score_bonus": 0,
        "alert_text": "",
    }

    # ---- Fetch data if not provided ----
    if daily is None and fetcher is not None:
        try:
            daily = fetcher.get_daily_bars(ticker, period="90d")
        except Exception as exc:
            logger.debug("Could not fetch daily bars for %s: %s", ticker, exc)

    if bars is None and fetcher is not None:
        try:
            bars = fetcher.get_intraday_bars(ticker, interval="1m", period="1d")
        except Exception as exc:
            logger.debug("Could not fetch intraday bars for %s: %s", ticker, exc)

    if daily is None or daily.empty:
        return result

    # ---- Detect zones ----
    zones = detect_zones(ticker, daily, bars, min_score=_MIN_REPORT_SCORE)

    # Persist / refresh zones in DB
    for zone in zones:
        try:
            upsert_zone(zone)
        except Exception as exc:
            logger.debug("Zone upsert skipped for %s: %s", ticker, exc)

    result["all_zones"] = zones

    if not zones:
        return result

    # ---- ATR for stop/target calculations ----
    atr = compute_atr(daily)
    if atr is None or atr <= 0:
        return result

    # ---- Current price ----
    if bars is not None and not bars.empty and "close" in bars.columns:
        current_price = float(bars["close"].iloc[-1])
    else:
        current_price = float(daily["close"].iloc[-1])

    active_zone = get_active_zone(ticker, current_price, zones, atr)
    if active_zone is None:
        return result

    result["has_active_zone"] = True
    result["active_zone"] = active_zone

    # ---- Entry / Stop / Target ----
    entry = current_price
    if active_zone.zone_type == "demand":
        stop = active_zone.zone_low - _STOP_ATR_MULT * atr
        risk = entry - stop
        target = entry + risk * _TARGET_RR if risk > 0 else entry + _TARGET_RR * atr
    else:  # supply
        stop = active_zone.zone_high + _STOP_ATR_MULT * atr
        risk = stop - entry
        target = entry - risk * _TARGET_RR if risk > 0 else entry - _TARGET_RR * atr

    rr = abs(target - entry) / abs(risk) if risk != 0 else 0.0

    result["entry"] = round(entry, 2)
    result["stop"] = round(stop, 2)
    result["target"] = round(target, 2)
    result["rr"] = round(rr, 1)

    # ---- Third-touch detection ----
    third_touch = active_zone.touch_count >= _THIRD_TOUCH_MIN_REACTIONS
    result["third_touch_setup"] = third_touch

    # ---- Confidence & win rate ----
    win_rate = get_win_rate(
        "third_touch_support" if third_touch else active_zone.zone_type
    )
    result["win_rate_pct"] = win_rate

    # Confidence: zone strength score scaled to 50–90 range, boosted by third touch
    base_conf = 50.0 + active_zone.strength_score * 0.40
    if third_touch:
        base_conf += 8.0
    result["confidence_pct"] = round(min(base_conf, 92.0), 1)

    # ---- Zone score bonus for probability engine ----
    result["zone_score_bonus"] = get_zone_score_bonus(active_zone)

    # ---- Format alert text ----
    result["alert_text"] = format_zone_alert(result)

    return result


# ---------------------------------------------------------------------------
# Probability bonus
# ---------------------------------------------------------------------------

def get_zone_score_bonus(zone: SupplyDemandZone) -> int:
    """Return a score bonus (0–15) to inject into the probability engine.

    The bonus is proportional to the zone's strength and timeframe.

    Bonus table:
    - Daily/Weekly Institutional (≥80)  → +15
    - Daily/Weekly Strong (65–79)       → +12
    - Daily/Weekly Moderate (50–64)     → +8
    - Intraday Strong (≥65)             → +6
    - Intraday Moderate (50–64)         → +3
    - Weak / unknown                    → 0

    Args:
        zone: :class:`~momentum_radar.signals.supply_demand.SupplyDemandZone`

    Returns:
        Integer score bonus.
    """
    high_tf = zone.timeframe in ("daily", "weekly")

    if high_tf and zone.strength_score >= 80:
        return 15
    if high_tf and zone.strength_score >= 65:
        return 12
    if high_tf and zone.strength_score >= 50:
        return 8
    if not high_tf and zone.strength_score >= 65:
        return 6
    if not high_tf and zone.strength_score >= 50:
        return 3
    return 0


# ---------------------------------------------------------------------------
# Alert formatting
# ---------------------------------------------------------------------------

def format_zone_alert(result: Dict) -> str:
    """Render a human-readable zone alert string.

    Args:
        result: Dict returned by :func:`scan_ticker`.

    Returns:
        Formatted multi-line string, or empty string if no active zone.
    """
    zone: Optional[SupplyDemandZone] = result.get("active_zone")
    if zone is None:
        return ""

    ticker = result["ticker"]
    direction_emoji = "🟢" if zone.zone_type == "demand" else "🔴"
    zone_label = f"{zone.strength_label} {zone.timeframe.title()} {'Demand' if zone.zone_type == 'demand' else 'Supply'}"

    lines = [
        f"{direction_emoji} {zone.zone_type.upper()} ZONE ALERT — {ticker}",
        "═" * 36,
        f"Zone:     {zone_label}  [${zone.zone_low:.2f} – ${zone.zone_high:.2f}]",
        f"Status:   {zone.status.title()} ({zone.touch_count} test{'s' if zone.touch_count != 1 else ''})",
        f"Strength: {zone.strength_score:.0f}/100 ({zone.strength_label})",
        f"Impulse:  {zone.impulse_magnitude:.1f}× ATR  |  Volume: {zone.volume_expansion:.1f}× avg",
        "",
        f"Price: ${result.get('entry', 0):.2f}  (inside zone)",
    ]

    if result.get("third_touch_setup"):
        lines += [
            "",
            "📐 Third-Touch Setup detected!",
            f"Prior reactions: {zone.touch_count}  |  Win rate: {result.get('win_rate_pct', 0):.0f}%",
        ]
    else:
        lines.append(f"Win rate: {result.get('win_rate_pct', 0):.0f}%")

    if result.get("entry") is not None:
        lines += [
            "",
            f"Entry:  ${result['entry']:.2f}",
            f"Stop:   ${result['stop']:.2f}",
            f"Target: ${result['target']:.2f}",
            f"R:R:    {result['rr']:.1f}",
            f"Confidence: {result['confidence_pct']:.0f}%",
        ]

    lines.append(f"\nDetected: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Universe scanner
# ---------------------------------------------------------------------------

def scan_universe(
    tickers: List[str],
    fetcher,
    min_score: float = _MIN_ALERT_SCORE,
    top_n: int = 10,
) -> List[Dict]:
    """Run S&D zone analysis across a list of tickers.

    Args:
        tickers:   List of stock symbols.
        fetcher:   Data provider instance.
        min_score: Only return results with zone strength ≥ this value.
        top_n:     Maximum number of results to return.

    Returns:
        List of result dicts (see :func:`scan_ticker`) sorted by zone
        strength score descending, limited to ``top_n`` entries.
    """
    results = []
    for ticker in tickers:
        try:
            r = scan_ticker(ticker, fetcher=fetcher)
            if r["has_active_zone"] and r["active_zone"].strength_score >= min_score:
                results.append(r)
        except Exception as exc:
            logger.debug("S&D scan failed for %s: %s", ticker, exc)

    results.sort(
        key=lambda r: (r["active_zone"].strength_score if r["active_zone"] else 0),
        reverse=True,
    )
    return results[:top_n]
