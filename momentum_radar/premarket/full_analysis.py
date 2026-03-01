"""
full_analysis.py – Full Institutional-Level Stock Analysis Engine.

When the user calls ``/analyze TICKER``, this module assembles:

1. Market data  (price, % change, market cap, float, beta, ATR, 52W hi/lo)
2. Technical analysis  (trend, RSI, MACD, VWAP, MA alignment, S/R, pattern)
3. Options data  (P/C ratio, IV rank, gamma, max pain, walls, dealer bias)
4. Flow & smart money  (unusual flow, dark pool, institutional accumulation)
5. Catalyst scan  (recent news placeholder, earnings proximity)
6. AI summary  (3 scenarios: bullish / neutral / bearish)
7. Trade quality score  (0–100), strategy recommendation, time horizon

All sections degrade gracefully when data is unavailable.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

from momentum_radar.data.data_fetcher import BaseDataFetcher

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Indicator helpers (reuse existing utils where possible)
# ---------------------------------------------------------------------------

def _compute_rsi(closes: pd.Series, period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return round(float(val), 1) if not pd.isna(val) else None


def _compute_macd(closes: pd.Series) -> Optional[Dict]:
    if len(closes) < 26:
        return None
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal
    return {
        "macd": round(float(macd_line.iloc[-1]), 4),
        "signal": round(float(signal.iloc[-1]), 4),
        "histogram": round(float(histogram.iloc[-1]), 4),
        "bullish": float(histogram.iloc[-1]) > 0,
    }


def _compute_sma(closes: pd.Series, period: int) -> Optional[float]:
    if len(closes) < period:
        return None
    return round(float(closes.rolling(period).mean().iloc[-1]), 2)


def _ma_alignment(closes: pd.Series) -> str:
    sma20 = _compute_sma(closes, 20)
    sma50 = _compute_sma(closes, 50)
    sma200 = _compute_sma(closes, 200)
    if sma20 and sma50 and sma200:
        if sma20 > sma50 > sma200:
            return "BULLISH (20>50>200)"
        if sma20 < sma50 < sma200:
            return "BEARISH (20<50<200)"
        return "MIXED"
    return "Insufficient data"


def _trend(closes: pd.Series) -> str:
    if len(closes) < 10:
        return "Unknown"
    short = _compute_sma(closes, 5)
    mid = _compute_sma(closes, 20)
    last = float(closes.iloc[-1])
    if short and mid:
        if last > short > mid:
            return "UPTREND"
        if last < short < mid:
            return "DOWNTREND"
    return "SIDEWAYS"


def _support_resistance(daily: pd.DataFrame) -> Dict:
    if daily is None or len(daily) < 20:
        return {}
    highs = daily["high"].tail(60)
    lows = daily["low"].tail(60)
    last_close = float(daily["close"].iloc[-1])
    # Simple S/R: nearby swing highs and lows
    resistance = round(float(highs.quantile(0.9)), 2)
    support = round(float(lows.quantile(0.1)), 2)
    return {"support": support, "resistance": resistance}


def _get_52w(daily: pd.DataFrame) -> Dict:
    if daily is None or daily.empty:
        return {}
    period = daily.tail(252)
    return {
        "high_52w": round(float(period["high"].max()), 2),
        "low_52w": round(float(period["low"].min()), 2),
    }


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def run_full_analysis(ticker: str, fetcher: BaseDataFetcher) -> Dict:
    """Run a full institutional-level analysis for *ticker*.

    Args:
        ticker:  Stock symbol.
        fetcher: Data provider.

    Returns:
        Dict with all analysis sections.  Individual fields may be ``None``
        when data is not available.
    """
    result: Dict = {"ticker": ticker}

    # ---- Fetch data ----
    try:
        daily = fetcher.get_daily_bars(ticker, period="252d")
    except Exception:
        daily = None

    try:
        intraday = fetcher.get_intraday_bars(ticker, interval="1m", period="1d")
    except Exception:
        intraday = None

    try:
        quote = fetcher.get_quote(ticker)
    except Exception:
        quote = None

    try:
        fundamentals = fetcher.get_fundamentals(ticker)
    except Exception:
        fundamentals = None

    try:
        options = fetcher.get_options_volume(ticker)
    except Exception:
        options = None

    # =========================================================
    # 1. Market Data
    # =========================================================
    price: Optional[float] = None
    prev_close: Optional[float] = None
    pct_change: Optional[float] = None

    if quote:
        price = quote.get("price")
        prev_close = quote.get("prev_close")
        if price and prev_close and prev_close != 0:
            pct_change = round((price - prev_close) / prev_close * 100, 2)

    float_shares: Optional[float] = None
    shares_out: Optional[float] = None
    short_pct: Optional[float] = None
    days_to_cover: Optional[float] = None
    if fundamentals:
        float_shares = fundamentals.get("float_shares")
        shares_out = fundamentals.get("shares_outstanding")
        short_pct = fundamentals.get("short_percent_of_float")
        days_to_cover = fundamentals.get("short_ratio")

    market_cap: Optional[float] = (
        price * shares_out if price and shares_out else None
    )

    atr: Optional[float] = None
    try:
        from momentum_radar.utils.indicators import compute_atr
        if daily is not None:
            atr = compute_atr(daily)
    except Exception:
        pass

    w52 = _get_52w(daily) if daily is not None else {}

    result["market_data"] = {
        "price": round(price, 2) if price else None,
        "pct_change": pct_change,
        "market_cap": round(market_cap, 0) if market_cap else None,
        "float_shares": float_shares,
        "atr": round(atr, 2) if atr else None,
        "high_52w": w52.get("high_52w"),
        "low_52w": w52.get("low_52w"),
    }

    # =========================================================
    # 2. Technical Analysis
    # =========================================================
    tech: Dict = {}
    if daily is not None and not daily.empty:
        closes = daily["close"]
        tech["trend"] = _trend(closes)
        tech["ma_alignment"] = _ma_alignment(closes)
        tech["sma_20"] = _compute_sma(closes, 20)
        tech["sma_50"] = _compute_sma(closes, 50)
        tech["sma_200"] = _compute_sma(closes, 200)
        tech["rsi"] = _compute_rsi(closes)
        tech["macd"] = _compute_macd(closes)

        sr = _support_resistance(daily)
        tech["support"] = sr.get("support")
        tech["resistance"] = sr.get("resistance")

        # VWAP
        vwap: Optional[float] = None
        try:
            from momentum_radar.utils.indicators import compute_vwap
            if intraday is not None:
                vwap = compute_vwap(intraday)
        except Exception:
            pass
        tech["vwap"] = round(vwap, 2) if vwap else None

        # RVOL
        rvol: Optional[float] = None
        try:
            from momentum_radar.utils.indicators import compute_rvol
            if intraday is not None:
                rvol = compute_rvol(intraday, daily)
        except Exception:
            pass
        tech["rvol"] = round(rvol, 2) if rvol else None

        # Simple breakout detection
        from momentum_radar.premarket.squeeze_detector import _detect_breakout
        tech["breakout"] = _detect_breakout(daily)

    result["technical"] = tech

    # =========================================================
    # 3. Options Data
    # =========================================================
    opts_data: Dict = {}
    if options:
        call_vol = int(options.get("call_volume", 0) or 0)
        put_vol = int(options.get("put_volume", 0) or 0)
        cp_ratio = round(call_vol / put_vol, 2) if put_vol > 0 else None
        opts_data["call_volume"] = call_vol
        opts_data["put_volume"] = put_vol
        opts_data["cp_ratio"] = cp_ratio
        if call_vol > put_vol * 1.5:
            opts_data["dealer_bias"] = "BULLISH"
        elif put_vol > call_vol * 1.5:
            opts_data["dealer_bias"] = "BEARISH"
        else:
            opts_data["dealer_bias"] = "NEUTRAL"

    # Try enriched options data via options_analyzer
    try:
        from momentum_radar.options.options_analyzer import get_options_summary
        summary = get_options_summary(ticker)
        if summary:
            opts_data["max_pain"] = summary.get("max_pain_strike")
            opts_data["put_call_ratio"] = summary.get("put_call_ratio")
            opts_data["put_call_interpretation"] = summary.get("put_call_interpretation")
    except Exception:
        pass

    result["options"] = opts_data

    # =========================================================
    # 4. Flow & Smart Money
    # =========================================================
    flow_data: Dict = {}
    try:
        from momentum_radar.options.options_analyzer import get_options_flow
        flow = get_options_flow(ticker)
        if flow:
            flow_data["net_sentiment"] = flow.get("net_sentiment")
            flow_data["dollar_call_flow"] = flow.get("dollar_call_flow")
            flow_data["dollar_put_flow"] = flow.get("dollar_put_flow")
            flow_data["smart_money_signals"] = flow.get("smart_money_signals", [])
            flow_data["call_sweeps_count"] = len(flow.get("call_sweeps", []))
            flow_data["put_sweeps_count"] = len(flow.get("put_sweeps", []))
    except Exception:
        pass

    # Short interest as proxy for institutional positioning
    if short_pct is not None:
        flow_data["short_interest_pct"] = short_pct
    if days_to_cover is not None:
        flow_data["days_to_cover"] = days_to_cover

    result["flow"] = flow_data

    # =========================================================
    # 5. Catalyst Scan
    # =========================================================
    # News and earnings data requires a premium API; we include placeholders
    # and source from yfinance where available.
    catalyst: Dict = {"news": [], "earnings_date": None, "analyst_notes": []}
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        news_items = t.news or []
        catalyst["news"] = [
            n.get("content", {}).get("title", n.get("title", ""))
            for n in news_items[:5]
            if n
        ]
        cal = t.calendar
        if cal is not None and not cal.empty:
            try:
                earnings_val = cal.loc["Earnings Date"].iloc[0]
                catalyst["earnings_date"] = str(earnings_val)[:10]
            except Exception:
                pass
    except Exception:
        pass

    result["catalyst"] = catalyst

    # =========================================================
    # 6 & 7. AI Summary Engine + Trade Quality Score
    # =========================================================
    result["ai_summary"] = _build_ai_summary(result, price, atr, short_pct)

    return result


# ---------------------------------------------------------------------------
# AI Summary Engine
# ---------------------------------------------------------------------------

def _build_ai_summary(
    data: Dict,
    price: Optional[float],
    atr: Optional[float],
    short_pct: Optional[float],
) -> Dict:
    """Synthesise all signals into a structured AI summary.

    Args:
        data:      Full analysis dict (sections already populated).
        price:     Current price.
        atr:       Average True Range.
        short_pct: Short interest as decimal.

    Returns:
        Dict with keys ``signals``, ``dominant_bias``,
        ``bullish_scenario``, ``neutral_scenario``, ``bearish_scenario``,
        ``trade_quality_score``, ``volatility_rating``, ``risk_level``,
        ``strategy_type``, ``time_horizon``, ``position_sizing_note``.
    """
    signals: List[str] = []
    bull_points = 0
    bear_points = 0

    tech = data.get("technical", {})
    opts = data.get("options", {})
    flow = data.get("flow", {})

    # ---- Collect signals ----
    trend = tech.get("trend", "")
    if trend == "UPTREND":
        bull_points += 2
        signals.append("Price trend: UPTREND")
    elif trend == "DOWNTREND":
        bear_points += 2
        signals.append("Price trend: DOWNTREND")

    rsi = tech.get("rsi")
    if rsi is not None:
        if rsi > 70:
            bear_points += 1
            signals.append(f"RSI {rsi} – overbought")
        elif rsi < 30:
            bull_points += 1
            signals.append(f"RSI {rsi} – oversold")
        else:
            signals.append(f"RSI {rsi} – neutral zone")

    macd = tech.get("macd")
    if macd:
        if macd.get("bullish"):
            bull_points += 1
            signals.append("MACD histogram positive – bullish momentum")
        else:
            bear_points += 1
            signals.append("MACD histogram negative – bearish momentum")

    if tech.get("breakout"):
        bull_points += 2
        signals.append("Breakout detected above 20-day high")

    dealer_bias = opts.get("dealer_bias", "")
    if dealer_bias == "BULLISH":
        bull_points += 1
        signals.append("Options dealer positioning: BULLISH")
    elif dealer_bias == "BEARISH":
        bear_points += 1
        signals.append("Options dealer positioning: BEARISH")

    net_sentiment = flow.get("net_sentiment", "")
    if net_sentiment:
        signals.append(f"Smart money flow: {net_sentiment}")
        if "BULL" in net_sentiment.upper():
            bull_points += 1
        elif "BEAR" in net_sentiment.upper():
            bear_points += 1

    if short_pct and short_pct > 0.20:
        bull_points += 1  # high SI = potential squeeze fuel
        signals.append(f"Short interest {short_pct:.1%} – squeeze risk")

    rvol = tech.get("rvol")
    if rvol and rvol > 2.0:
        bull_points += 1
        signals.append(f"RVOL {rvol:.1f}x – elevated volume")

    # ---- Dominant bias ----
    if bull_points > bear_points + 1:
        dominant_bias = "BULLISH"
    elif bear_points > bull_points + 1:
        dominant_bias = "BEARISH"
    else:
        dominant_bias = "NEUTRAL"

    # ---- Scenario targets ----
    atr2 = round(atr * 2, 2) if atr else None
    atr4 = round(atr * 4, 2) if atr else None

    bull_target1 = round(price + atr2, 2) if price and atr2 else None
    bull_target2 = round(price + atr4, 2) if price and atr4 else None
    bear_target = round(price - atr2, 2) if price and atr2 else None
    neutral_high = round(price + (atr or 0), 2) if price else None
    neutral_low = round(price - (atr or 0), 2) if price else None

    bull_prob = min(30 + bull_points * 8, 75)
    bear_prob = min(30 + bear_points * 8, 75)
    neutral_prob = max(100 - bull_prob - bear_prob, 10)

    # ---- Trade quality score (0–100) ----
    quality_score = 0
    quality_score += min(bull_points * 10, 50)
    if rvol and rvol > 1.5:
        quality_score += 10
    if opts.get("max_pain"):
        quality_score += 5
    if tech.get("breakout"):
        quality_score += 15
    if short_pct and short_pct > 0.15:
        quality_score += 10
    quality_score = min(quality_score, 100)

    # ---- Risk & strategy ----
    atr_pct = round(atr / price * 100, 1) if atr and price else None
    if atr_pct and atr_pct > 4:
        volatility_rating = "HIGH"
        risk_level = "HIGH"
        strategy_type = "options spreads / defined-risk"
    elif atr_pct and atr_pct > 2:
        volatility_rating = "MODERATE"
        risk_level = "MODERATE"
        strategy_type = "shares or short-dated calls"
    else:
        volatility_rating = "LOW"
        risk_level = "LOW"
        strategy_type = "shares / covered calls"

    if dominant_bias == "BULLISH" and (rvol or 0) > 2:
        time_horizon = "short-term (1–5 days)"
    elif dominant_bias == "BULLISH":
        time_horizon = "swing (1–3 weeks)"
    else:
        time_horizon = "wait for confirmation"

    return {
        "signals": signals,
        "dominant_bias": dominant_bias,
        "bullish_scenario": {
            "probability_pct": bull_prob,
            "target1": bull_target1,
            "target2": bull_target2,
            "invalidation": bear_target,
        },
        "neutral_scenario": {
            "range_low": neutral_low,
            "range_high": neutral_high,
        },
        "bearish_scenario": {
            "probability_pct": bear_prob,
            "downside_target": bear_target,
        },
        "trade_quality_score": quality_score,
        "volatility_rating": volatility_rating,
        "risk_level": risk_level,
        "strategy_type": strategy_type,
        "time_horizon": time_horizon,
        "position_sizing_note": "Risk no more than 1–2% of portfolio per trade.",
    }


# ---------------------------------------------------------------------------
# Text formatter
# ---------------------------------------------------------------------------

def format_full_analysis(analysis: Dict) -> str:
    """Render a full analysis dict as a multi-line text message.

    Args:
        analysis: Dict returned by :func:`run_full_analysis`.

    Returns:
        Formatted string ready for Telegram or console output.
    """
    ticker = analysis.get("ticker", "?")
    lines: List[str] = [f"FULL ANALYSIS: {ticker}", ""]

    # Market data
    md = analysis.get("market_data", {})
    lines.append("MARKET DATA")
    lines.append(f"  Price: ${md.get('price', 'N/A')}")
    pct = md.get("pct_change")
    lines.append(f"  Change: {('+' if pct and pct >= 0 else '')}{pct}%" if pct is not None else "  Change: N/A")
    mc = md.get("market_cap")
    lines.append(f"  Market Cap: ${mc / 1e9:.2f}B" if mc and mc >= 1e9 else (f"  Market Cap: ${mc / 1e6:.0f}M" if mc else "  Market Cap: N/A"))
    fs = md.get("float_shares")
    lines.append(f"  Float: {fs / 1e6:.1f}M" if fs else "  Float: N/A")
    lines.append(f"  ATR: ${md.get('atr', 'N/A')}")
    lines.append(f"  52W High: ${md.get('high_52w', 'N/A')}")
    lines.append(f"  52W Low: ${md.get('low_52w', 'N/A')}")
    lines.append("")

    # Technical
    tech = analysis.get("technical", {})
    lines.append("TECHNICAL ANALYSIS")
    lines.append(f"  Trend: {tech.get('trend', 'N/A')}")
    lines.append(f"  MA Alignment: {tech.get('ma_alignment', 'N/A')}")
    lines.append(f"  SMA 20/50/200: {tech.get('sma_20', 'N/A')} / {tech.get('sma_50', 'N/A')} / {tech.get('sma_200', 'N/A')}")
    lines.append(f"  RSI: {tech.get('rsi', 'N/A')}")
    macd = tech.get("macd")
    if macd:
        lines.append(f"  MACD: {macd.get('macd')} | Signal: {macd.get('signal')} | Hist: {macd.get('histogram')}")
    lines.append(f"  VWAP: {'$' + str(tech.get('vwap')) if tech.get('vwap') else 'N/A'}")
    lines.append(f"  RVOL: {str(tech.get('rvol')) + 'x' if tech.get('rvol') is not None else 'N/A'}")
    lines.append(f"  Support: ${tech.get('support', 'N/A')}")
    lines.append(f"  Resistance: ${tech.get('resistance', 'N/A')}")
    lines.append(f"  Breakout: {'YES' if tech.get('breakout') else 'NO'}")
    lines.append("")

    # Options
    opts = analysis.get("options", {})
    if opts:
        lines.append("OPTIONS DATA")
        lines.append(f"  Put/Call Ratio: {opts.get('put_call_ratio') or opts.get('cp_ratio', 'N/A')}")
        lines.append(f"  Call Volume: {opts.get('call_volume', 'N/A'):,}" if isinstance(opts.get("call_volume"), int) else f"  Call Volume: N/A")
        lines.append(f"  Put Volume: {opts.get('put_volume', 'N/A'):,}" if isinstance(opts.get("put_volume"), int) else f"  Put Volume: N/A")
        lines.append(f"  Max Pain: ${opts.get('max_pain', 'N/A')}")
        lines.append(f"  Dealer Bias: {opts.get('dealer_bias', 'N/A')}")
        lines.append("")

    # Flow
    flow = analysis.get("flow", {})
    if flow:
        lines.append("FLOW & SMART MONEY")
        lines.append(f"  Net Sentiment: {flow.get('net_sentiment', 'N/A')}")
        dcf = flow.get("dollar_call_flow")
        dpf = flow.get("dollar_put_flow")
        if dcf is not None:
            lines.append(f"  Dollar Flow: ${dcf / 1e6:.1f}M calls vs ${(dpf or 0) / 1e6:.1f}M puts")
        si = flow.get("short_interest_pct")
        if si is not None:
            lines.append(f"  Short Interest: {si:.1%}")
        dtc = flow.get("days_to_cover")
        if dtc is not None:
            lines.append(f"  Days to Cover: {dtc:.1f}")
        smart = flow.get("smart_money_signals", [])
        if smart:
            for s in smart[:3]:
                lines.append(f"  Smart Signal: {s}")
        lines.append("")

    # Catalyst
    cat = analysis.get("catalyst", {})
    news = cat.get("news", [])
    if news:
        lines.append("RECENT NEWS")
        for n in news[:3]:
            lines.append(f"  - {n}")
        lines.append("")
    ed = cat.get("earnings_date")
    if ed:
        lines.append(f"  Earnings Date: {ed}")
        lines.append("")

    # AI Summary
    ai = analysis.get("ai_summary", {})
    if ai:
        lines.append("AI SUMMARY")
        lines.append(f"  Dominant Bias: {ai.get('dominant_bias', 'N/A')}")
        sigs = ai.get("signals", [])
        if sigs:
            for s in sigs[:5]:
                lines.append(f"  Signal: {s}")
        lines.append("")
        lines.append("SCENARIO PROJECTIONS")
        bull = ai.get("bullish_scenario", {})
        lines.append(f"  BULLISH ({bull.get('probability_pct', '?')}%): T1 ${bull.get('target1', 'N/A')}  T2 ${bull.get('target2', 'N/A')}")
        lines.append(f"  Invalidation: ${bull.get('invalidation', 'N/A')}")
        neut = ai.get("neutral_scenario", {})
        lines.append(f"  NEUTRAL: ${neut.get('range_low', 'N/A')} – ${neut.get('range_high', 'N/A')}")
        bear = ai.get("bearish_scenario", {})
        lines.append(f"  BEARISH ({bear.get('probability_pct', '?')}%): Target ${bear.get('downside_target', 'N/A')}")
        lines.append("")
        lines.append("DECISION FRAMEWORK")
        lines.append(f"  Trade Quality Score: {ai.get('trade_quality_score', 'N/A')}/100")
        lines.append(f"  Volatility: {ai.get('volatility_rating', 'N/A')}")
        lines.append(f"  Risk Level: {ai.get('risk_level', 'N/A')}")
        lines.append(f"  Strategy: {ai.get('strategy_type', 'N/A')}")
        lines.append(f"  Time Horizon: {ai.get('time_horizon', 'N/A')}")
        lines.append(f"  {ai.get('position_sizing_note', '')}")

    return "\n".join(lines)
