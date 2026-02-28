"""
volume.py – Volume spike and relative volume signal detection.

Registered signals
------------------
- ``volume_spike``   – detects unusual intraday volume vs. recent averages
- ``relative_volume`` – compares current daily volume to the 30-day average
"""

import logging
from typing import Dict, Optional

import pandas as pd

from momentum_radar.config import config
from momentum_radar.signals.base import SignalResult
from momentum_radar.signals.scoring import register_signal

logger = logging.getLogger(__name__)


def _safe_mean(series: pd.Series) -> float:
    """Return the mean of *series*, or 0.0 if it is empty/NaN."""
    if series is None or series.empty:
        return 0.0
    val = series.mean()
    return float(val) if pd.notna(val) else 0.0


@register_signal("volume_spike")
def volume_spike(
    ticker: str,
    bars: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
    **kwargs,
) -> SignalResult:
    """Detect unusual intraday volume vs. recent averages.

    Conditions (any one triggers):
    - Current 5-min volume ≥ ``VOLUME_SPIKE_STRONG`` × avg 5-min → +2
    - Current 5-min volume ≥ ``VOLUME_SPIKE_MODERATE`` × avg 5-min → +1
    - Current daily volume ≥ ``DAILY_VOLUME_RATIO`` × 30-day avg volume
    - Latest 1-min volume ≥ ``INTRADAY_VOLUME_RATIO`` × prev 20-min avg

    Args:
        ticker: Stock symbol (unused directly but kept for uniform signature).
        bars: Intraday 1-min OHLCV DataFrame.
        daily: Daily OHLCV DataFrame (30+ days).

    Returns:
        :class:`~momentum_radar.signals.base.SignalResult`
    """
    cfg = config.signals

    if bars is None or bars.empty:
        return SignalResult(triggered=False, score=0, details="No intraday data")

    score = 0
    reasons: list = []

    # ------------------------------------------------------------------
    # 1-min volume: latest bar vs. prior 20-bar average
    # ------------------------------------------------------------------
    if "volume" in bars.columns and len(bars) >= 2:
        last_vol = float(bars["volume"].iloc[-1])
        prior_20_avg = _safe_mean(bars["volume"].iloc[-21:-1])
        if prior_20_avg > 0:
            ratio_1m = last_vol / prior_20_avg
            if ratio_1m >= cfg.intraday_volume_ratio:
                score = max(score, 2)
                reasons.append(
                    f"1m vol {ratio_1m:.1f}x 20-bar avg"
                )

    # ------------------------------------------------------------------
    # 5-min resample: last bar vs. rolling average of prior 5-min bars
    # ------------------------------------------------------------------
    if "volume" in bars.columns and len(bars) >= 10:
        try:
            bars_5m = bars["volume"].resample("5min").sum()
            if len(bars_5m) >= 2:
                last_5m = float(bars_5m.iloc[-1])
                avg_5m = _safe_mean(bars_5m.iloc[:-1])
                if avg_5m > 0:
                    ratio_5m = last_5m / avg_5m
                    if ratio_5m >= cfg.volume_spike_strong:
                        score = max(score, 2)
                        reasons.append(
                            f"5m vol {ratio_5m:.1f}x avg (strong)"
                        )
                    elif ratio_5m >= cfg.volume_spike_moderate:
                        score = max(score, 1)
                        reasons.append(
                            f"5m vol {ratio_5m:.1f}x avg (moderate)"
                        )
        except Exception as exc:
            logger.debug("5m resample failed for %s: %s", ticker, exc)

    # ------------------------------------------------------------------
    # Daily volume vs. 30-day average
    # ------------------------------------------------------------------
    if daily is not None and not daily.empty and "volume" in daily.columns:
        avg_daily = _safe_mean(daily["volume"].iloc[-30:])
        today_vol = float(daily["volume"].iloc[-1]) if not daily.empty else 0.0
        if avg_daily > 0:
            ratio_daily = today_vol / avg_daily
            if ratio_daily >= cfg.daily_volume_ratio:
                score = max(score, 1)
                reasons.append(
                    f"daily vol {ratio_daily:.1f}x 30d avg"
                )

    triggered = score > 0
    return SignalResult(
        triggered=triggered,
        score=score,
        details="; ".join(reasons) if reasons else "No volume spike detected",
    )


@register_signal("relative_volume")
def relative_volume(
    ticker: str,
    bars: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
    **kwargs,
) -> SignalResult:
    """Score based on relative volume (RVOL).

    RVOL is the ratio of the current day's cumulative volume to the 30-day
    average daily volume.

    - RVOL ≥ ``RVOL_STRONG`` (3.0) → +2
    - RVOL ≥ ``RVOL_MODERATE`` (2.0) → +1

    Args:
        ticker: Stock symbol.
        bars: Intraday 1-min OHLCV DataFrame.
        daily: Daily OHLCV DataFrame.

    Returns:
        :class:`~momentum_radar.signals.base.SignalResult`
    """
    cfg = config.signals

    if daily is None or daily.empty or "volume" not in daily.columns:
        return SignalResult(triggered=False, score=0, details="No daily data for RVOL")

    avg_daily = _safe_mean(daily["volume"].iloc[-31:-1])
    if avg_daily <= 0:
        return SignalResult(triggered=False, score=0, details="Avg daily volume is zero")

    # Use intraday cumulative volume if available, else fall back to daily
    if bars is not None and not bars.empty and "volume" in bars.columns:
        current_volume = float(bars["volume"].sum())
    else:
        current_volume = float(daily["volume"].iloc[-1])

    rvol = current_volume / avg_daily

    if rvol >= cfg.rvol_strong:
        return SignalResult(
            triggered=True,
            score=2,
            details=f"RVOL {rvol:.2f} (≥{cfg.rvol_strong}x)",
        )
    if rvol >= cfg.rvol_moderate:
        return SignalResult(
            triggered=True,
            score=1,
            details=f"RVOL {rvol:.2f} (≥{cfg.rvol_moderate}x)",
        )
    return SignalResult(triggered=False, score=0, details=f"RVOL {rvol:.2f} – below threshold")
