"""
services/fake_breakout.py – Fake breakout / liquidity trap detection.

A fake breakout occurs when price pierces a key level (support, resistance,
or a supply/demand zone boundary) but quickly reverses without follow-through.
These traps are often engineered by institutional players to hunt retail stops.

Detection rules
---------------
A breakout is flagged as *fake* when ANY of the following conditions holds:

1. **Low-volume break** – breakout bar volume < ``_VOLUME_MULT`` × recent average
2. **Wick-dominant candle** – breakout candle's body is < ``_BODY_RATIO_MAX``
   of the full range (wick-heavy = indecision)
3. **No follow-through** – the bar immediately after the breakout closes back
   inside the prior level
4. **Divergence** – price makes new high/low but RSI does not confirm
5. **Immediate reclaim** – within ``_RECLAIM_BARS`` bars price has fully
   reclaimed the broken level

Usage::

    from momentum_radar.services.fake_breakout import is_fake_breakout

    fake = is_fake_breakout(bars, level=150.0, direction="above")
    if fake["is_fake"]:
        print(fake["reasons"])
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Volume on breakout bar must exceed this multiple of the recent average
_VOLUME_MULT: float = 1.20
# Body ratio threshold; below this the candle is considered wick-heavy
_BODY_RATIO_MAX: float = 0.50
# Maximum number of bars to check for an immediate reclaim
_RECLAIM_BARS: int = 3
# RSI period for divergence check
_RSI_PERIOD: int = 14


def _compute_rsi(closes: pd.Series, period: int = _RSI_PERIOD) -> Optional[float]:
    """Return the most recent RSI value."""
    if len(closes) < period + 1:
        return None
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period).mean().iloc[-1]
    avg_loss = loss.rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def is_fake_breakout(
    bars: Optional[pd.DataFrame],
    level: float,
    direction: str,
    volume_mult: float = _VOLUME_MULT,
    body_ratio_max: float = _BODY_RATIO_MAX,
    reclaim_bars: int = _RECLAIM_BARS,
) -> Dict:
    """Determine whether the most recent breakout is fake.

    Args:
        bars:           Recent intraday or daily OHLCV bars.  The last row is
                        treated as the breakout bar (or current bar).
        level:          The price level that was broken (support / resistance /
                        zone boundary).
        direction:      ``"above"`` if price broke upward through *level*,
                        ``"below"`` if it broke downward.
        volume_mult:    Minimum volume multiple for a *real* breakout.
        body_ratio_max: Body ratio below which the candle is considered wick-heavy.
        reclaim_bars:   Number of bars to look back for a reclaim.

    Returns:
        Dict with keys:

        - ``is_fake``  – ``True`` when the breakout is considered fake
        - ``reasons``  – list of triggered rule names
        - ``score``    – fake probability score (0–100)
    """
    reasons: List[str] = []

    if bars is None or len(bars) < 4:
        return {"is_fake": False, "reasons": [], "score": 0}

    last = bars.iloc[-1]
    last_close = float(last["close"])
    last_open = float(last["open"])
    last_high = float(last["high"])
    last_low = float(last["low"])
    last_vol = float(last["volume"]) if "volume" in bars.columns else 0.0

    # Average volume over the lookback window (excluding the last bar)
    avg_vol = float(bars["volume"].iloc[:-1].mean()) if "volume" in bars.columns else 0.0

    candle_range = last_high - last_low
    body = abs(last_close - last_open)
    body_ratio = body / candle_range if candle_range > 0 else 0.0

    # 1. Low-volume break
    if avg_vol > 0 and last_vol < volume_mult * avg_vol:
        reasons.append("low_volume_breakout")

    # 2. Wick-dominant candle
    if body_ratio < body_ratio_max:
        reasons.append("wick_dominant_candle")

    # 3. No follow-through – next bar (bars[-2] is the breakout, bars[-1] now)
    #    Check that after a breakout, the most recent close is back inside
    if direction == "above":
        broke_through = last_close > level
        if not broke_through:
            reasons.append("no_follow_through")
    else:  # below
        broke_through = last_close < level
        if not broke_through:
            reasons.append("no_follow_through")

    # 4. RSI divergence: price at extreme but RSI not confirming
    if len(bars) >= _RSI_PERIOD + 2:
        rsi = _compute_rsi(bars["close"].astype(float))
        if rsi is not None:
            if direction == "above" and last_close > level:
                # Bullish breakout should have RSI > 50 ideally
                if rsi < 45:
                    reasons.append("rsi_divergence")
            elif direction == "below" and last_close < level:
                # Bearish breakdown should have RSI < 55 ideally
                if rsi > 55:
                    reasons.append("rsi_divergence")

    # 5. Immediate reclaim – look at the last reclaim_bars bars to see if
    #    price has already crossed back through the level
    lookback = bars.iloc[-min(reclaim_bars + 1, len(bars)):-1]
    if not lookback.empty:
        if direction == "above":
            reclaimed = float(lookback["close"].min()) < level
        else:
            reclaimed = float(lookback["close"].max()) > level
        if reclaimed:
            reasons.append("immediate_reclaim")

    count = len(reasons)
    score = int(min(count / 5.0 * 100.0, 100.0))
    is_fake = count >= 2

    if is_fake:
        logger.debug(
            "Fake breakout detected (level=%.2f %s): %s", level, direction, reasons
        )

    return {"is_fake": is_fake, "reasons": reasons, "score": score}
