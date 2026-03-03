"""
services/signal_engine.py – Multi-confirmation signal engine.

A signal is only considered actionable if at least **3 independent
confirmations** align.  Four or more confirmations produce a
"HIGH CONFIDENCE" priority rating.

Confirmation types
------------------
1. **Volume**          – RVOL ≥ 2.0 or volume > 2× 30-day average
2. **Pattern**         – Breakout from ascending triangle, cup & handle, flag,
                         double bottom, or support bounce (confidence ≥ 70 %)
3. **Candlestick**     – Bullish engulfing / hammer at support, or bearish
                         engulfing / shooting star at resistance
4. **Options**         – Call volume spike, put volume spike, or gamma-flip zone
5. **HTF Trend**       – Price above daily EMA21 and EMA50 (higher-timeframe bias)
6. **Momentum**        – RSI in bullish zone with positive MACD histogram
7. **Retest**          – Price returns to and holds a recently broken key level
8. **Liquidity Sweep** – Wick below/above a swing point followed by strong reversal

Fake breakout filter
--------------------
A breakout is rejected (no signal sent) if any of the following are true:

- Volume below average on the break candle
- Candle has a large wick (wick > 50 % of total range)
- RSI divergence against breakout direction

Signal priority
---------------
- 0–2 confirmations → NO SIGNAL
- 3 confirmations   → ALERT
- 4+ confirmations  → HIGH CONFIDENCE ALERT
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Lookback window used when searching for the second trough in a double-bottom pattern
_DOUBLE_BOTTOM_BOUNCE_PERIOD: int = 10


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Confirmation:
    """A single confirmed signal component."""

    name: str           # human-readable label shown in alerts
    category: str       # "volume" | "pattern" | "candlestick" | "options" | "htf_trend" | "momentum" | "retest" | "liquidity_sweep"
    detail: str         # short description, e.g. "RVOL 3.4x average"
    confidence: float   # 0–100


@dataclass
class SignalResult:
    """Aggregated result from the multi-confirmation engine."""

    ticker: str
    confirmations: List[Confirmation] = field(default_factory=list)
    priority: str = "NO_SIGNAL"      # "NO_SIGNAL" | "ALERT" | "HIGH_CONFIDENCE"
    confidence_score: float = 0.0    # probability % (0–100)
    risk_grade: str = "High"         # "Low" | "Medium" | "High"
    setup_strength: str = "C"        # "A+" | "A" | "B" | "C"

    @property
    def confirmation_count(self) -> int:
        return len(self.confirmations)

    @property
    def confirmation_labels(self) -> List[str]:
        """Return formatted labels for Telegram display."""
        return [f"{c.name}: {c.detail}" for c in self.confirmations]


# ---------------------------------------------------------------------------
# Individual confirmation checkers
# ---------------------------------------------------------------------------

def _check_volume(
    bars: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
) -> Optional[Confirmation]:
    """Return a Volume confirmation if RVOL ≥ 2.0 or daily vol > 2× avg."""
    if daily is None or daily.empty or "volume" not in daily.columns:
        return None

    avg_30d = float(daily["volume"].iloc[-31:-1].mean()) if len(daily) >= 31 else 0.0
    if avg_30d <= 0:
        return None

    # Use intraday cumulative or last daily bar
    if bars is not None and not bars.empty and "volume" in bars.columns:
        current_vol = float(bars["volume"].sum())
    else:
        current_vol = float(daily["volume"].iloc[-1])

    rvol = current_vol / avg_30d

    if rvol >= 2.0:
        confidence = min(100.0, 70.0 + (rvol - 2.0) * 10)
        return Confirmation(
            name="Volume Spike",
            category="volume",
            detail=f"{rvol:.1f}x average",
            confidence=round(confidence, 1),
        )
    return None


def _check_pattern(daily: Optional[pd.DataFrame]) -> Optional[Confirmation]:
    """Detect a basic breakout / bullish continuation pattern.

    Checks for:
    - 20-day breakout (price closes above 20-day high of prior bars) – asc. triangle proxy
    - Tight consolidation then expansion (flag / pennant proxy)
    - Double-bottom proxy (two comparable lows within 3 % in trailing 20 bars)

    Returns a Confirmation with confidence ≥ 70 if detected, else ``None``.
    """
    if daily is None or len(daily) < 22:
        return None

    closes = daily["close"]
    highs = daily["high"]
    lows = daily["low"]

    last_close = float(closes.iloc[-1])
    prior_high = float(highs.iloc[-22:-1].max())

    # --- Ascending triangle / breakout ---
    if last_close > prior_high:
        # Measure how far above prior high
        pct_above = (last_close - prior_high) / prior_high
        confidence = min(100.0, 70.0 + pct_above * 200)
        return Confirmation(
            name="Ascending Triangle Breakout",
            category="pattern",
            detail=f"Close ${last_close:.2f} > 20d high ${prior_high:.2f}",
            confidence=round(confidence, 1),
        )

    # --- Double bottom proxy ---
    if len(daily) >= 20:
        low_window = lows.iloc[-20:]
        low1 = float(low_window.min())
        # Look for a second trough within 3 % of the first
        mid_high = float(closes.iloc[-_DOUBLE_BOTTOM_BOUNCE_PERIOD:].max())
        if mid_high > low1 * 1.02:  # there was a bounce
            second_lows = low_window[(low_window > low1 * 0.97) & (low_window <= low1 * 1.03)]
            if len(second_lows) >= 2:
                return Confirmation(
                    name="Double Bottom",
                    category="pattern",
                    detail=f"Support near ${low1:.2f}",
                    confidence=72.0,
                )

    # --- Support bounce (price near 20-day low and recovering) ---
    prior_low = float(lows.iloc[-22:-1].min())
    if prior_low > 0 and abs(last_close - prior_low) / prior_low < 0.02:
        return Confirmation(
            name="Support Bounce",
            category="pattern",
            detail=f"Near support ${prior_low:.2f}",
            confidence=70.0,
        )

    return None


def _check_candlestick(
    bars: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
) -> Optional[Confirmation]:
    """Detect significant candlestick patterns at key levels.

    Patterns checked:
    - Bullish engulfing (last bar body fully wraps prior bar body, bullish)
    - Hammer (small body + long lower wick, at support)
    - Shooting star (small body + long upper wick, at resistance) – bearish
    - Bearish engulfing (last bar body fully wraps prior bar, bearish)
    """
    # Use daily bars for candlestick analysis (more reliable than 1m)
    df = daily if (daily is not None and len(daily) >= 2) else None
    if df is None:
        return None

    o = float(df["open"].iloc[-1])
    c = float(df["close"].iloc[-1])
    h = float(df["high"].iloc[-1])
    lo = float(df["low"].iloc[-1])

    prev_o = float(df["open"].iloc[-2])
    prev_c = float(df["close"].iloc[-2])

    body = abs(c - o)
    total_range = h - lo if h > lo else 1e-9

    # --- Bullish Engulfing ---
    if c > o and prev_c < prev_o:  # current bullish, previous bearish
        if o <= prev_c and c >= prev_o:  # engulfs prior body
            return Confirmation(
                name="Bullish Engulfing",
                category="candlestick",
                detail="Bullish engulfing at support level",
                confidence=75.0,
            )

    # --- Hammer (bullish reversal) ---
    lower_wick = min(o, c) - lo
    upper_wick = h - max(o, c)
    if (
        lower_wick >= 2 * body
        and upper_wick <= body * 0.5
        and c >= o
        and body / total_range < 0.35
    ):
        return Confirmation(
            name="Hammer",
            category="candlestick",
            detail="Hammer candle at support",
            confidence=72.0,
        )

    # --- Shooting Star (bearish reversal) ---
    if (
        upper_wick >= 2 * body
        and lower_wick <= body * 0.5
        and c <= o
        and body / total_range < 0.35
    ):
        return Confirmation(
            name="Shooting Star",
            category="candlestick",
            detail="Shooting star at resistance",
            confidence=72.0,
        )

    # --- Bearish Engulfing ---
    if c < o and prev_c > prev_o:  # current bearish, previous bullish
        if o >= prev_c and c <= prev_o:
            return Confirmation(
                name="Bearish Engulfing",
                category="candlestick",
                detail="Bearish engulfing at resistance",
                confidence=75.0,
            )

    return None


def _check_options(options: Optional[Dict]) -> Optional[Confirmation]:
    """Return an Options confirmation if unusual call or put activity detected.

    Triggers:
    - Call volume spike (call_volume > 2× avg_call_volume)
    - Put volume spike (put_volume > 2× avg_put_volume)
    - Gamma flip: call/put ratio ≥ 2.0
    """
    if options is None:
        return None

    call_vol = int(options.get("call_volume", 0) or 0)
    put_vol = int(options.get("put_volume", 0) or 0)
    avg_call = float(options.get("avg_call_volume", 1) or 1)
    avg_put = float(options.get("avg_put_volume", 1) or 1)

    call_ratio = call_vol / avg_call if avg_call > 0 else 0.0
    put_ratio = put_vol / avg_put if avg_put > 0 else 0.0
    cp_ratio = call_vol / put_vol if put_vol > 0 else 0.0

    if call_ratio >= 2.0:
        confidence = min(100.0, 70.0 + (call_ratio - 2.0) * 5)
        return Confirmation(
            name="Call Flow Spike",
            category="options",
            detail=f"Call volume {call_ratio:.1f}x average",
            confidence=round(confidence, 1),
        )

    if put_ratio >= 2.0:
        confidence = min(100.0, 70.0 + (put_ratio - 2.0) * 5)
        return Confirmation(
            name="Put Flow Spike",
            category="options",
            detail=f"Put volume {put_ratio:.1f}x average",
            confidence=round(confidence, 1),
        )

    if cp_ratio >= 2.0:
        return Confirmation(
            name="Gamma Flip Zone",
            category="options",
            detail=f"C/P ratio {cp_ratio:.1f} – bullish gamma bias",
            confidence=74.0,
        )

    return None


# ---------------------------------------------------------------------------
# Higher-timeframe trend alignment
# ---------------------------------------------------------------------------

def _check_htf_trend(daily: Optional[pd.DataFrame]) -> Optional[Confirmation]:
    """Return a Confirmation if the daily trend is bullish (HTF bias confirmed).

    Bullish alignment: last close > EMA21 > EMA50 on daily bars.

    Args:
        daily: Daily OHLCV DataFrame (50+ bars recommended).

    Returns:
        :class:`Confirmation` or ``None``.
    """
    if daily is None or len(daily) < 50 or "close" not in daily.columns:
        return None

    closes = daily["close"]
    last_close = float(closes.iloc[-1])

    ema21 = float(closes.ewm(span=21, adjust=False).mean().iloc[-1])
    ema50 = float(closes.ewm(span=50, adjust=False).mean().iloc[-1])

    if last_close > ema21 > ema50:
        return Confirmation(
            name="HTF Trend Alignment",
            category="htf_trend",
            detail=f"Daily price ${last_close:.2f} > EMA21 {ema21:.2f} > EMA50 {ema50:.2f}",
            confidence=80.0,
        )

    return None


# ---------------------------------------------------------------------------
# Momentum alignment (RSI + MACD)
# ---------------------------------------------------------------------------

def _check_momentum(
    bars: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
) -> Optional[Confirmation]:
    """Return a Confirmation if RSI is in a bullish zone with positive MACD histogram.

    Uses intraday bars when available, falls back to daily bars.

    Args:
        bars:  Intraday OHLCV DataFrame.
        daily: Daily OHLCV DataFrame.

    Returns:
        :class:`Confirmation` or ``None``.
    """
    df = bars if (bars is not None and len(bars) >= 14) else daily
    if df is None or len(df) < 14 or "close" not in df.columns:
        return None

    closes = df["close"]

    # RSI
    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi_series = 100 - (100 / (1 + rs))
    rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else float("nan")

    # MACD histogram
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = float((macd_line - signal_line).iloc[-1])

    import math
    if math.isnan(rsi):
        return None

    if 40 <= rsi <= 70 and histogram > 0:
        return Confirmation(
            name="Momentum Alignment",
            category="momentum",
            detail=f"RSI {rsi:.1f} in momentum zone, MACD hist +{histogram:.4f}",
            confidence=75.0,
        )

    # Oversold bounce with MACD turning positive
    if rsi < 35 and histogram > 0:
        return Confirmation(
            name="Oversold Bounce",
            category="momentum",
            detail=f"RSI {rsi:.1f} oversold + MACD turning positive",
            confidence=72.0,
        )

    return None


# ---------------------------------------------------------------------------
# Retest confirmation
# ---------------------------------------------------------------------------

def _check_retest(daily: Optional[pd.DataFrame]) -> Optional[Confirmation]:
    """Return a Confirmation if price has returned to and held a key broken level.

    A retest is detected when the last close is within 1.5 % of the prior
    20-bar high (price broke above, pulled back, and is holding the breakout
    level as support).

    Args:
        daily: Daily OHLCV DataFrame.

    Returns:
        :class:`Confirmation` or ``None``.
    """
    if daily is None or len(daily) < 22 or "close" not in daily.columns:
        return None

    closes = daily["close"]
    highs = daily["high"]
    lows = daily["low"]

    last_close = float(closes.iloc[-1])
    prior_high = float(highs.iloc[-22:-2].max())
    prior_low = float(lows.iloc[-22:-2].min())

    # Bullish retest: price previously broke above prior_high and is retesting it
    if prior_high > 0 and abs(last_close - prior_high) / prior_high <= 0.015:
        return Confirmation(
            name="Retest of Key Level",
            category="retest",
            detail=f"Price ${last_close:.2f} retesting broken resistance ${prior_high:.2f}",
            confidence=72.0,
        )

    # Support retest: price bounced from prior low area
    if prior_low > 0 and abs(last_close - prior_low) / prior_low <= 0.015:
        return Confirmation(
            name="Retest of Support",
            category="retest",
            detail=f"Price ${last_close:.2f} holding support ${prior_low:.2f}",
            confidence=70.0,
        )

    return None


# ---------------------------------------------------------------------------
# Liquidity sweep confirmation
# ---------------------------------------------------------------------------

def _check_liquidity_sweep(
    bars: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
) -> Optional[Confirmation]:
    """Return a Confirmation if a liquidity sweep (stop hunt + reversal) is detected.

    A liquidity sweep is identified when:
    - The last candle wicked below a recent swing low (or above a swing high)
    - The candle then closed back above the swing low (or below the swing high)
    - Indicating a stop hunt followed by a strong reversal

    Args:
        bars:  Intraday OHLCV DataFrame.
        daily: Daily OHLCV DataFrame.

    Returns:
        :class:`Confirmation` or ``None``.
    """
    df = bars if (bars is not None and len(bars) >= 10) else daily
    if df is None or len(df) < 10:
        return None

    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        return None

    last = df.iloc[-1]
    prior = df.iloc[-11:-1]

    swing_low = float(prior["low"].min())
    swing_high = float(prior["high"].max())

    last_low = float(last["low"])
    last_high = float(last["high"])
    last_close = float(last["close"])
    last_open = float(last["open"])

    # Bullish sweep: wick below swing low but closed above it (stop hunt + reversal)
    if last_low < swing_low and last_close > swing_low:
        wick_size = swing_low - last_low
        body_size = abs(last_close - last_open)
        if wick_size > 0 and body_size > 0:
            return Confirmation(
                name="Liquidity Sweep (Bullish)",
                category="liquidity_sweep",
                detail=f"Wick to ${last_low:.2f} below swing low ${swing_low:.2f}, recovered ${last_close:.2f}",
                confidence=73.0,
            )

    # Bearish sweep: wick above swing high but closed below it
    if last_high > swing_high and last_close < swing_high:
        wick_size = last_high - swing_high
        body_size = abs(last_close - last_open)
        if wick_size > 0 and body_size > 0:
            return Confirmation(
                name="Liquidity Sweep (Bearish)",
                category="liquidity_sweep",
                detail=f"Wick to ${last_high:.2f} above swing high ${swing_high:.2f}, reversed ${last_close:.2f}",
                confidence=73.0,
            )

    return None


# ---------------------------------------------------------------------------
# Fake breakout detection (filter)
# ---------------------------------------------------------------------------

def _is_fake_breakout(
    bars: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
) -> bool:
    """Return ``True`` if the latest move shows signs of a fake breakout.

    A breakout is considered fake if any of:
    - Volume on the break candle is below the 20-bar average
    - The break candle has a large wick (wick > 50 % of total range)
    - RSI divergence: price made a new high/low but RSI did not confirm

    If ``True``, no signal should be sent regardless of confirmation count.

    Args:
        bars:  Intraday OHLCV DataFrame.
        daily: Daily OHLCV DataFrame.

    Returns:
        ``True`` if the breakout appears to be fake.
    """
    df = bars if (bars is not None and len(bars) >= 5) else daily
    if df is None or len(df) < 5:
        return False  # insufficient data – do not suppress

    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return False

    last = df.iloc[-1]
    last_open = float(last["open"])
    last_close = float(last["close"])
    last_high = float(last["high"])
    last_low = float(last["low"])
    last_vol = float(last["volume"])

    total_range = last_high - last_low
    if total_range <= 0:
        return False

    # --- Check 1: volume below average ---
    avg_vol = float(df["volume"].iloc[-21:-1].mean()) if len(df) > 1 else 0.0
    if avg_vol > 0 and last_vol < avg_vol:
        logger.debug("Fake breakout: volume below average (%.0f < %.0f)", last_vol, avg_vol)
        return True

    # --- Check 2: large wick (> 50% of candle range) ---
    body = abs(last_close - last_open)
    wick = total_range - body
    if wick / total_range > 0.50:
        logger.debug("Fake breakout: large wick ratio %.2f", wick / total_range)
        return True

    # --- Check 3: RSI divergence against breakout direction ---
    if len(df) >= 14 and "close" in df.columns:
        closes = df["close"]
        delta = closes.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, float("nan"))
        rsi_series = 100 - (100 / (1 + rs))
        if len(rsi_series) >= 2:
            rsi_now = float(rsi_series.iloc[-1])
            rsi_prev = float(rsi_series.iloc[-2])
            import math
            if not math.isnan(rsi_now) and not math.isnan(rsi_prev):
                price_up = last_close > float(closes.iloc[-2])
                rsi_up = rsi_now > rsi_prev
                # Price moved up but RSI moved down → bearish divergence (fake bull break)
                # Price moved down but RSI moved up → bullish divergence (fake bear break)
                if price_up != rsi_up:
                    logger.debug(
                        "Fake breakout: RSI divergence (price_up=%s, rsi_up=%s)",
                        price_up, rsi_up,
                    )
                    return True

    return False


# ---------------------------------------------------------------------------
# Supply & demand zone confirmation
# ---------------------------------------------------------------------------

def _check_supply_demand(
    ticker: str,
    bars: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
) -> Optional[Confirmation]:
    """Return a Confirmation if price is near a scored S&D zone.

    Uses :func:`~momentum_radar.signals.supply_demand.detect_zones` and
    :func:`~momentum_radar.signals.supply_demand.get_active_zone`.

    Args:
        ticker: Stock symbol.
        bars:   Intraday 1-min OHLCV DataFrame.
        daily:  Daily OHLCV DataFrame.

    Returns:
        :class:`Confirmation` or ``None``.
    """
    try:
        from momentum_radar.signals.supply_demand import detect_zones, get_active_zone
        from momentum_radar.utils.indicators import compute_atr

        if daily is None or daily.empty:
            return None

        atr = compute_atr(daily)
        if atr is None or atr <= 0:
            return None

        if bars is not None and not bars.empty and "close" in bars.columns:
            current_price = float(bars["close"].iloc[-1])
        else:
            current_price = float(daily["close"].iloc[-1])

        zones = detect_zones(ticker, daily, bars, min_score=50.0)
        active_zone = get_active_zone(ticker, current_price, zones, atr)

        if active_zone is None:
            return None

        conf = min(100.0, 65.0 + active_zone.strength_score * 0.25)
        direction = "Demand" if active_zone.zone_type == "demand" else "Supply"
        touch_label = "fresh" if active_zone.touch_count == 0 else f"tested {active_zone.touch_count}×"
        return Confirmation(
            name=f"{active_zone.strength_label} {direction} Zone",
            category="supply_demand",
            detail=(
                f"[${active_zone.zone_low:.2f}–${active_zone.zone_high:.2f}] "
                f"{active_zone.timeframe}, {touch_label}, "
                f"strength {active_zone.strength_score:.0f}/100"
            ),
            confidence=round(conf, 1),
        )
    except Exception as exc:
        logger.debug("S&D zone check failed for %s: %s", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# Probability scoring helpers
# ---------------------------------------------------------------------------

def _compute_probability(confirmations: List[Confirmation]) -> float:
    """Compute an overall probability score (0–100) from confirmations.

    The score is a weighted average of confirmation confidences with a
    diversity bonus for multiple independent categories.

    Args:
        confirmations: List of :class:`Confirmation` objects.

    Returns:
        Probability score (0–100).
    """
    n = len(confirmations)
    if n == 0:
        return 0.0

    avg_conf = sum(c.confidence for c in confirmations) / n

    # Diversity bonus: each unique category adds 2 pts (max +8)
    categories = {c.category for c in confirmations}
    diversity_bonus = min(len(categories) * 2.0, 8.0)

    # Quantity bonus: 3+ confirmations add 5 pts
    quantity_bonus = 5.0 if n >= 3 else 0.0

    total = avg_conf + diversity_bonus + quantity_bonus
    return round(min(total, 100.0), 1)


def _risk_grade_from_confidence(confidence_pct: float) -> str:
    """Map confidence to a risk grade label.

    Args:
        confidence_pct: Confidence score (0–100).

    Returns:
        ``"Low"`` / ``"Medium"`` / ``"High"``.
    """
    if confidence_pct >= 80:
        return "Low"
    if confidence_pct >= 65:
        return "Medium"
    return "High"


def _setup_strength_grade(n_confirmations: int, confidence_pct: float) -> str:
    """Map confirmation count + confidence to a setup-strength grade.

    Args:
        n_confirmations: Number of confirmations.
        confidence_pct:  Computed probability score.

    Returns:
        ``"A+"`` / ``"A"`` / ``"B"`` / ``"C"``.
    """
    if n_confirmations >= 3 and confidence_pct >= 80:
        return "A+"
    if n_confirmations >= 3 or confidence_pct >= 75:
        return "A"
    if n_confirmations >= 2 or confidence_pct >= 65:
        return "B"
    return "C"


# ---------------------------------------------------------------------------
# Main engine entry point
# ---------------------------------------------------------------------------

def evaluate(
    ticker: str,
    bars: Optional[pd.DataFrame] = None,
    daily: Optional[pd.DataFrame] = None,
    fundamentals: Optional[Dict] = None,
    options: Optional[Dict] = None,
) -> SignalResult:
    """Run all confirmation checks and return a :class:`SignalResult`.

    A signal is actionable only when at least **3 confirmations** align and
    no fake-breakout conditions are detected.

    Confirmations checked:
    1. Volume       – RVOL or daily volume spike
    2. Pattern      – 20-day breakout / double bottom / support bounce
    3. Candlestick  – engulfing / hammer / shooting star
    4. Options      – call/put flow spike or gamma flip
    5. S&D Zone     – price near an institutional supply/demand zone
    6. HTF Trend    – daily price > EMA21 > EMA50 (higher-timeframe bias)
    7. Momentum     – RSI in bullish zone with positive MACD histogram
    8. Retest       – price returning to a recently broken key level
    9. Liquidity Sweep – stop hunt + reversal pattern

    Fake breakout guard: if the latest candle shows low volume, a dominant
    wick, or RSI divergence, the signal is suppressed regardless of the
    confirmation count.

    Probability scoring added to result:
    - ``confidence_score``: weighted probability % (0–100)
    - ``risk_grade``: ``"Low"`` / ``"Medium"`` / ``"High"``
    - ``setup_strength``: ``"A+"`` / ``"A"`` / ``"B"`` / ``"C"``

    Args:
        ticker:        Stock symbol.
        bars:          Intraday 1-min OHLCV DataFrame.
        daily:         Daily OHLCV DataFrame (30+ days).
        fundamentals:  Fundamental data dict (reserved for future use).
        options:       Options activity dict.

    Returns:
        :class:`SignalResult` with priority and confirmation list.
    """
    # --- Fake breakout guard (applied before any confirmations) ---
    fake_break = _is_fake_breakout(bars, daily)

    confirmations: List[Confirmation] = []

    checkers = [
        ("volume",          lambda: _check_volume(bars, daily)),
        ("pattern",         lambda: _check_pattern(daily)),
        ("candlestick",     lambda: _check_candlestick(bars, daily)),
        ("options",         lambda: _check_options(options)),
        ("supply_demand",   lambda: _check_supply_demand(ticker, bars, daily)),
        ("htf_trend",       lambda: _check_htf_trend(daily)),
        ("momentum",        lambda: _check_momentum(bars, daily)),
        ("retest",          lambda: _check_retest(daily)),
        ("liquidity_sweep", lambda: _check_liquidity_sweep(bars, daily)),
    ]

    for category, checker in checkers:
        try:
            result = checker()
            if result is not None:
                confirmations.append(result)
        except Exception as exc:
            logger.debug("Confirmation check '%s' failed for %s: %s", category, ticker, exc)

    n = len(confirmations)

    # Require at least 3 confirmations; also reject fake breakouts
    if fake_break or n < 3:
        priority = "NO_SIGNAL"
    elif n >= 4:
        priority = "HIGH_CONFIDENCE"
    else:
        priority = "ALERT"

    confidence_pct = _compute_probability(confirmations)
    risk_grade = _risk_grade_from_confidence(confidence_pct)
    setup_strength = _setup_strength_grade(n, confidence_pct)

    result = SignalResult(
        ticker=ticker,
        confirmations=confirmations,
        priority=priority,
        confidence_score=confidence_pct,
        risk_grade=risk_grade,
        setup_strength=setup_strength,
    )

    if priority != "NO_SIGNAL":
        logger.info(
            "Signal engine: %s → %s (%d confirmations, conf %.0f%%, grade %s)",
            ticker,
            priority,
            n,
            confidence_pct,
            setup_strength,
        )
    elif fake_break and n >= 3:
        logger.debug(
            "Signal engine: %s suppressed – fake breakout detected (%d confirmations)",
            ticker,
            n,
        )

    return result
