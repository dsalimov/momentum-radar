"""
fundamentals.py – Company Fundamentals & Earnings Analysis Engine.

Provides three public helpers:

* :func:`get_financial_statements`  – Income statement, cash flow, balance sheet.
* :func:`get_earnings_analysis`     – Earnings history, EPS trend, guidance AI summary.
* :func:`format_fundamentals_report` – Render financials as a multi-line text report.
* :func:`format_earnings_report`    – Render earnings + AI summary as a text report.

All data is sourced from *yfinance*, which is already a required dependency.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_val(val, scale: float = 1.0, prefix: str = "$", suffix: str = "") -> str:
    """Format a numeric value with B/M/K scaling."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "N/A"
    try:
        v = float(val) / scale
        if abs(float(val)) >= 1e9:
            return f"{prefix}{float(val) / 1e9:.2f}B{suffix}"
        if abs(float(val)) >= 1e6:
            return f"{prefix}{float(val) / 1e6:.2f}M{suffix}"
        if abs(float(val)) >= 1e3:
            return f"{prefix}{float(val) / 1e3:.2f}K{suffix}"
        return f"{prefix}{float(val):.2f}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


def _pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None or old == 0:
        return None
    return round((new - old) / abs(old) * 100, 1)


def _row(df: pd.DataFrame, *keys: str) -> Optional[float]:
    """Extract a row value from a financial DataFrame by trying multiple key names."""
    for key in keys:
        if key in df.index:
            series = df.loc[key]
            if not series.empty:
                val = series.iloc[0]
                try:
                    return float(val) if not pd.isna(val) else None
                except (TypeError, ValueError):
                    return None
    return None


# ---------------------------------------------------------------------------
# Financial Statements
# ---------------------------------------------------------------------------

def get_financial_statements(ticker: str) -> Dict:
    """Fetch income statement, cash flow, and balance sheet for *ticker*.

    Uses yfinance annual and quarterly data.

    Args:
        ticker: Stock symbol (e.g. ``"AAPL"``).

    Returns:
        Dict with keys ``income_statement``, ``cash_flow``, ``balance_sheet``,
        each containing a sub-dict of labelled line items (annual and trailing).
        Returns empty dicts per section when data is unavailable.
    """
    result: Dict = {
        "ticker": ticker,
        "income_statement": {},
        "cash_flow": {},
        "balance_sheet": {},
    }

    try:
        import yfinance as yf
        t = yf.Ticker(ticker)

        # ---- Income Statement ----
        inc = t.income_stmt
        q_inc = t.quarterly_income_stmt
        if inc is not None and not inc.empty:
            # Annual columns are sorted newest-first
            cols = list(inc.columns)
            y0 = cols[0] if len(cols) >= 1 else None
            y1 = cols[1] if len(cols) >= 2 else None

            rev_0 = _row(inc, "Total Revenue")
            rev_1 = _row(inc.iloc[:, [1]] if y1 else inc, "Total Revenue") if y1 else None
            ni_0 = _row(inc, "Net Income")
            ni_1 = _row(inc.iloc[:, [1]] if y1 else inc, "Net Income") if y1 else None
            gross_0 = _row(inc, "Gross Profit")
            ebit_0 = _row(inc, "EBIT", "Operating Income")
            eps_basic = _row(inc, "Basic EPS", "Diluted EPS")
            eps_diluted = _row(inc, "Diluted EPS")

            result["income_statement"] = {
                "total_revenue": rev_0,
                "total_revenue_yoy_pct": _pct_change(rev_0, rev_1),
                "gross_profit": gross_0,
                "gross_margin_pct": round(gross_0 / rev_0 * 100, 1) if gross_0 and rev_0 else None,
                "ebit": ebit_0,
                "net_income": ni_0,
                "net_income_yoy_pct": _pct_change(ni_0, ni_1),
                "net_margin_pct": round(ni_0 / rev_0 * 100, 1) if ni_0 and rev_0 else None,
                "eps_basic": eps_basic,
                "eps_diluted": eps_diluted,
                "period_label": str(y0)[:10] if y0 else "N/A",
            }

        # ---- Cash Flow ----
        cf = t.cash_flow
        if cf is not None and not cf.empty:
            ops_cf = _row(cf, "Operating Cash Flow", "Cash From Operations")
            capex = _row(cf, "Capital Expenditure", "Purchase Of PPE")
            free_cf = None
            if ops_cf is not None and capex is not None:
                free_cf = ops_cf + capex  # capex is typically negative
            div_paid = _row(cf, "Payment For Dividends", "Dividends Paid")
            share_repurchase = _row(cf, "Repurchase Of Capital Stock", "Common Stock Repurchased")

            result["cash_flow"] = {
                "operating_cash_flow": ops_cf,
                "capital_expenditure": capex,
                "free_cash_flow": free_cf,
                "dividends_paid": div_paid,
                "share_repurchases": share_repurchase,
                "period_label": str(list(cf.columns)[0])[:10] if not cf.empty else "N/A",
            }

        # ---- Balance Sheet ----
        bs = t.balance_sheet
        if bs is not None and not bs.empty:
            total_assets = _row(bs, "Total Assets")
            total_liabilities = _row(bs, "Total Liabilities Net Minority Interest", "Total Liabilities")
            equity = _row(bs, "Stockholders Equity", "Total Equity Gross Minority Interest")
            cash = _row(bs, "Cash And Cash Equivalents", "Cash")
            total_debt = _row(bs, "Total Debt")
            current_assets = _row(bs, "Current Assets")
            current_liabilities = _row(bs, "Current Liabilities")

            debt_to_equity: Optional[float] = None
            if total_debt is not None and equity and equity != 0:
                debt_to_equity = round(total_debt / abs(equity), 2)

            current_ratio: Optional[float] = None
            if current_assets and current_liabilities and current_liabilities != 0:
                current_ratio = round(current_assets / abs(current_liabilities), 2)

            result["balance_sheet"] = {
                "total_assets": total_assets,
                "total_liabilities": total_liabilities,
                "stockholders_equity": equity,
                "cash_and_equivalents": cash,
                "total_debt": total_debt,
                "current_assets": current_assets,
                "current_liabilities": current_liabilities,
                "debt_to_equity": debt_to_equity,
                "current_ratio": current_ratio,
                "period_label": str(list(bs.columns)[0])[:10] if not bs.empty else "N/A",
            }

    except Exception as exc:
        logger.error("Error fetching financial statements for %s: %s", ticker, exc)

    return result


# ---------------------------------------------------------------------------
# Earnings Analysis
# ---------------------------------------------------------------------------

def get_earnings_analysis(ticker: str) -> Dict:
    """Fetch earnings history, EPS trend, upcoming date, and generate an AI summary.

    Args:
        ticker: Stock symbol.

    Returns:
        Dict with keys ``earnings_history``, ``next_earnings_date``,
        ``eps_trend``, ``revenue_trend``, ``analyst_estimates``,
        ``ai_summary``, ``guidance_summary``.
    """
    result: Dict = {
        "ticker": ticker,
        "earnings_history": [],
        "next_earnings_date": None,
        "eps_trend": None,
        "revenue_trend": None,
        "analyst_estimates": {},
        "ai_summary": {},
        "guidance_summary": "",
    }

    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            pass

        # ---- Next earnings date ----
        try:
            cal = t.calendar
            if cal is not None and not cal.empty:
                earnings_val = cal.loc["Earnings Date"].iloc[0]
                result["next_earnings_date"] = str(earnings_val)[:10]
        except Exception:
            pass

        # ---- Earnings history ----
        history_rows: List[Dict] = []
        try:
            earnings_dates_df = t.earnings_dates
            if earnings_dates_df is not None and not earnings_dates_df.empty:
                # Keep only past rows with reported EPS
                past = earnings_dates_df.dropna(subset=["Reported EPS"]) if "Reported EPS" in earnings_dates_df.columns else earnings_dates_df.head(0)
                for idx, row in past.head(8).iterrows():
                    rep_eps = row.get("Reported EPS")
                    est_eps = row.get("EPS Estimate")
                    surprise = row.get("Surprise(%)")
                    history_rows.append({
                        "date": str(idx)[:10],
                        "reported_eps": float(rep_eps) if rep_eps is not None and not pd.isna(rep_eps) else None,
                        "estimated_eps": float(est_eps) if est_eps is not None and not pd.isna(est_eps) else None,
                        "surprise_pct": float(surprise) if surprise is not None and not pd.isna(surprise) else None,
                    })
        except Exception as exc:
            logger.debug("Could not fetch earnings_dates for %s: %s", ticker, exc)

        result["earnings_history"] = history_rows

        # ---- Analyst estimates ----
        try:
            analyst_estimates: Dict = {}
            eps_fwd = info.get("forwardEps")
            eps_ttm = info.get("trailingEps")
            rev_est = info.get("revenueEstimate")
            target_mean = info.get("targetMeanPrice")
            target_high = info.get("targetHighPrice")
            target_low = info.get("targetLowPrice")
            num_analysts = info.get("numberOfAnalystOpinions")
            recommendation = info.get("recommendationKey", "")

            analyst_estimates = {
                "forward_eps": eps_fwd,
                "trailing_eps": eps_ttm,
                "target_price_mean": target_mean,
                "target_price_high": target_high,
                "target_price_low": target_low,
                "num_analysts": num_analysts,
                "recommendation": recommendation.upper() if recommendation else "N/A",
            }
            result["analyst_estimates"] = analyst_estimates
        except Exception:
            pass

        # ---- EPS trend (beat/miss ratio) ----
        if history_rows:
            beats = sum(1 for r in history_rows if r.get("surprise_pct") is not None and r["surprise_pct"] > 0)
            misses = sum(1 for r in history_rows if r.get("surprise_pct") is not None and r["surprise_pct"] <= 0)
            total = beats + misses
            result["eps_trend"] = {
                "beats": beats,
                "misses": misses,
                "beat_rate_pct": round(beats / total * 100) if total > 0 else None,
            }
            # Average surprise
            surprises = [r["surprise_pct"] for r in history_rows if r.get("surprise_pct") is not None]
            if surprises:
                result["eps_trend"]["avg_surprise_pct"] = round(sum(surprises) / len(surprises), 1)

        # ---- Revenue trend from income statement ----
        try:
            inc = t.income_stmt
            if inc is not None and not inc.empty and len(inc.columns) >= 2:
                revs = []
                for col in list(inc.columns)[:4]:
                    rev = _row(inc[[col]], "Total Revenue")
                    if rev:
                        revs.append({"period": str(col)[:10], "revenue": rev})
                if len(revs) >= 2:
                    growth = _pct_change(revs[0]["revenue"], revs[1]["revenue"])
                    result["revenue_trend"] = {
                        "periods": revs,
                        "latest_yoy_growth_pct": growth,
                    }
        except Exception:
            pass

        # ---- AI Summary ----
        result["ai_summary"] = _build_earnings_ai_summary(result, info)
        result["guidance_summary"] = _build_guidance_summary(result, info)

    except Exception as exc:
        logger.error("Error fetching earnings analysis for %s: %s", ticker, exc)

    return result


# ---------------------------------------------------------------------------
# AI Earnings Summary Engine
# ---------------------------------------------------------------------------

def _build_earnings_ai_summary(data: Dict, info: Dict) -> Dict:
    """Build a rule-based AI summary of earnings quality and outlook.

    Args:
        data: Partial result dict with earnings fields filled in.
        info: yfinance info dict.

    Returns:
        Dict with keys ``quality_score``, ``key_observations``,
        ``bull_case``, ``bear_case``, ``verdict``.
    """
    observations: List[str] = []
    score = 50  # start neutral

    eps_trend = data.get("eps_trend") or {}
    beat_rate = eps_trend.get("beat_rate_pct")
    avg_surprise = eps_trend.get("avg_surprise_pct")
    rev_trend = data.get("revenue_trend") or {}
    rev_growth = rev_trend.get("latest_yoy_growth_pct")
    analyst = data.get("analyst_estimates") or {}
    rec = analyst.get("recommendation", "")
    fwd_eps = analyst.get("forward_eps")
    ttm_eps = analyst.get("trailing_eps")

    # EPS beat rate
    if beat_rate is not None:
        if beat_rate >= 75:
            score += 15
            observations.append(f"Strong earnings beat rate: {beat_rate:.0f}% of last {eps_trend.get('beats', 0) + eps_trend.get('misses', 0)} quarters.")
        elif beat_rate >= 50:
            score += 5
            observations.append(f"Moderate earnings beat rate: {beat_rate:.0f}%.")
        else:
            score -= 10
            observations.append(f"Below-average beat rate: {beat_rate:.0f}% – company has struggled to meet estimates.")

    if avg_surprise is not None:
        if avg_surprise > 5:
            score += 10
            observations.append(f"Average EPS surprise: +{avg_surprise:.1f}% – consistently beats estimates.")
        elif avg_surprise < -2:
            score -= 10
            observations.append(f"Average EPS surprise: {avg_surprise:.1f}% – tends to miss estimates.")

    # Revenue growth
    if rev_growth is not None:
        if rev_growth > 15:
            score += 15
            observations.append(f"Strong revenue growth: +{rev_growth:.1f}% YoY.")
        elif rev_growth > 5:
            score += 8
            observations.append(f"Solid revenue growth: +{rev_growth:.1f}% YoY.")
        elif rev_growth >= 0:
            score += 2
            observations.append(f"Flat revenue growth: +{rev_growth:.1f}% YoY – stable but not expanding.")
        else:
            score -= 10
            observations.append(f"Revenue declining: {rev_growth:.1f}% YoY – headwinds present.")

    # Analyst recommendation
    if rec in ("BUY", "STRONG_BUY"):
        score += 10
        observations.append(f"Analyst consensus: {rec} ({analyst.get('num_analysts', '?')} analysts, mean target ${analyst.get('target_price_mean', 'N/A')}).")
    elif rec in ("SELL", "UNDERPERFORM"):
        score -= 10
        observations.append(f"Analyst consensus: {rec} – bearish institutional view.")
    elif rec == "HOLD":
        observations.append(f"Analyst consensus: HOLD – market fairly valued per analysts.")

    # EPS growth
    if fwd_eps is not None and ttm_eps and ttm_eps != 0:
        eps_growth = _pct_change(fwd_eps, ttm_eps)
        if eps_growth is not None:
            if eps_growth > 15:
                score += 10
                observations.append(f"Forward EPS growth: +{eps_growth:.1f}% – expanding earnings outlook.")
            elif eps_growth < -10:
                score -= 10
                observations.append(f"Forward EPS declining: {eps_growth:.1f}% – earnings contraction expected.")

    # Profitability from info
    pe = info.get("trailingPE")
    if pe and pe > 0:
        if pe < 15:
            score += 5
            observations.append(f"Low P/E ratio: {pe:.1f}x – potentially undervalued vs earnings.")
        elif pe > 50:
            score -= 5
            observations.append(f"High P/E ratio: {pe:.1f}x – premium valuation; high growth expected.")

    score = max(0, min(100, score))

    # Verdict
    if score >= 70:
        verdict = "BULLISH – Strong fundamental backdrop supports higher prices."
    elif score >= 55:
        verdict = "MODERATELY BULLISH – Solid earnings quality with manageable risks."
    elif score >= 45:
        verdict = "NEUTRAL – Mixed signals; monitor next earnings for direction."
    elif score >= 30:
        verdict = "CAUTIOUS – Weak earnings trends; risk management is key."
    else:
        verdict = "BEARISH – Fundamental deterioration; elevated investment risk."

    bull_case = _build_bull_case(data, info, score)
    bear_case = _build_bear_case(data, info, score)

    return {
        "quality_score": score,
        "key_observations": observations,
        "bull_case": bull_case,
        "bear_case": bear_case,
        "verdict": verdict,
    }


def _build_bull_case(data: Dict, info: Dict, score: int) -> str:
    parts: List[str] = []
    rev_trend = data.get("revenue_trend") or {}
    rev_growth = rev_trend.get("latest_yoy_growth_pct")
    eps_trend = data.get("eps_trend") or {}
    beat_rate = eps_trend.get("beat_rate_pct")
    analyst = data.get("analyst_estimates") or {}
    target_high = analyst.get("target_price_high")

    if rev_growth and rev_growth > 0:
        parts.append(f"Revenue growing at {rev_growth:.1f}% YoY")
    if beat_rate and beat_rate >= 50:
        parts.append(f"beats estimates {beat_rate:.0f}% of the time")
    if target_high:
        parts.append(f"analyst high target ${target_high:.2f}")
    if info.get("trailingPE") and info["trailingPE"] < 25:
        parts.append(f"reasonable P/E of {info['trailingPE']:.1f}x")
    return "Company " + "; ".join(parts) + "." if parts else "Continued execution could drive upside."


def _build_bear_case(data: Dict, info: Dict, score: int) -> str:
    parts: List[str] = []
    rev_trend = data.get("revenue_trend") or {}
    rev_growth = rev_trend.get("latest_yoy_growth_pct")
    analyst = data.get("analyst_estimates") or {}
    target_low = analyst.get("target_price_low")

    if rev_growth and rev_growth < 0:
        parts.append(f"revenue declining {rev_growth:.1f}% YoY")
    if target_low:
        parts.append(f"analyst low target ${target_low:.2f}")
    if info.get("trailingPE") and info["trailingPE"] > 40:
        parts.append(f"high valuation P/E {info['trailingPE']:.1f}x leaves little margin for error")
    if info.get("debtToEquity") and info["debtToEquity"] > 100:
        parts.append(f"elevated debt/equity {info['debtToEquity']:.0f}%")
    return "Risk factors: " + "; ".join(parts) + "." if parts else "A miss on next earnings could pressure the stock."


def _build_guidance_summary(data: Dict, info: Dict) -> str:
    """Synthesize available guidance / analyst estimates into an actionable paragraph."""
    analyst = data.get("analyst_estimates") or {}
    rec = analyst.get("recommendation", "")
    target_mean = analyst.get("target_price_mean")
    target_high = analyst.get("target_price_high")
    target_low = analyst.get("target_price_low")
    num = analyst.get("num_analysts")
    fwd_eps = analyst.get("forward_eps")
    next_date = data.get("next_earnings_date")
    eps_trend = data.get("eps_trend") or {}
    beat_rate = eps_trend.get("beat_rate_pct")

    lines: List[str] = []
    if target_mean and target_high and target_low and num:
        lines.append(
            f"{num} analysts cover this stock with a consensus {rec} rating. "
            f"Mean price target: ${target_mean:.2f} (range ${target_low:.2f}–${target_high:.2f})."
        )
    if fwd_eps:
        direction = "positive" if fwd_eps > 0 else "negative"
        lines.append(f"Forward EPS estimate: ${fwd_eps:.2f} ({direction} earnings expected).")
    if beat_rate is not None:
        lines.append(
            f"Historically, this company beats EPS estimates {beat_rate:.0f}% of the time, "
            f"{'suggesting a beat is likely' if beat_rate >= 60 else 'suggesting earnings surprises are mixed'}."
        )
    if next_date:
        lines.append(
            f"Watch the upcoming earnings on {next_date}. "
            "A strong beat with raised guidance could catalyse a multi-day move."
        )
    lines.append(
        "Always size positions according to your risk tolerance and consider setting "
        "stops below key technical support levels before earnings."
    )
    return " ".join(lines) if lines else "Insufficient data for guidance summary."


# ---------------------------------------------------------------------------
# Text Formatters
# ---------------------------------------------------------------------------

def format_fundamentals_report(data: Dict) -> str:
    """Render the financial statements dict as a formatted text report.

    Args:
        data: Dict returned by :func:`get_financial_statements`.

    Returns:
        Multi-line string ready for Telegram or console output.
    """
    ticker = data.get("ticker", "?")
    lines: List[str] = [f"FUNDAMENTALS: {ticker}", ""]

    # ---- Income Statement ----
    inc = data.get("income_statement", {})
    lines.append("INCOME STATEMENT (Annual)")
    lines.append(f"  Period: {inc.get('period_label', 'N/A')}")
    lines.append(f"  Total Revenue:   {_fmt_val(inc.get('total_revenue'))}"
                 + (f"  ({'+' if (inc.get('total_revenue_yoy_pct') or 0) >= 0 else ''}"
                    f"{inc.get('total_revenue_yoy_pct', 'N/A')}% YoY)"
                    if inc.get("total_revenue_yoy_pct") is not None else ""))
    lines.append(f"  Gross Profit:    {_fmt_val(inc.get('gross_profit'))}"
                 + (f"  ({inc.get('gross_margin_pct')}% margin)"
                    if inc.get("gross_margin_pct") is not None else ""))
    lines.append(f"  EBIT:            {_fmt_val(inc.get('ebit'))}")
    lines.append(f"  Net Income:      {_fmt_val(inc.get('net_income'))}"
                 + (f"  ({'+' if (inc.get('net_income_yoy_pct') or 0) >= 0 else ''}"
                    f"{inc.get('net_income_yoy_pct', 'N/A')}% YoY)"
                    if inc.get("net_income_yoy_pct") is not None else ""))
    if inc.get("net_margin_pct") is not None:
        lines.append(f"  Net Margin:      {inc['net_margin_pct']}%")
    if inc.get("eps_diluted") is not None:
        lines.append(f"  Diluted EPS:     ${inc['eps_diluted']:.2f}")
    lines.append("")

    # ---- Cash Flow ----
    cf = data.get("cash_flow", {})
    if cf:
        lines.append("CASH FLOW (Annual)")
        lines.append(f"  Period: {cf.get('period_label', 'N/A')}")
        lines.append(f"  Operating CF:    {_fmt_val(cf.get('operating_cash_flow'))}")
        lines.append(f"  Capital Expenditure: {_fmt_val(cf.get('capital_expenditure'))}")
        lines.append(f"  Free Cash Flow:  {_fmt_val(cf.get('free_cash_flow'))}")
        if cf.get("dividends_paid") is not None:
            lines.append(f"  Dividends Paid:  {_fmt_val(cf.get('dividends_paid'))}")
        if cf.get("share_repurchases") is not None:
            lines.append(f"  Share Repurchases: {_fmt_val(cf.get('share_repurchases'))}")
        lines.append("")

    # ---- Balance Sheet ----
    bs = data.get("balance_sheet", {})
    if bs:
        lines.append("BALANCE SHEET (Annual)")
        lines.append(f"  Period: {bs.get('period_label', 'N/A')}")
        lines.append(f"  Total Assets:    {_fmt_val(bs.get('total_assets'))}")
        lines.append(f"  Total Liabilities: {_fmt_val(bs.get('total_liabilities'))}")
        lines.append(f"  Stockholders Equity: {_fmt_val(bs.get('stockholders_equity'))}")
        lines.append(f"  Cash & Equivalents: {_fmt_val(bs.get('cash_and_equivalents'))}")
        lines.append(f"  Total Debt:      {_fmt_val(bs.get('total_debt'))}")
        if bs.get("debt_to_equity") is not None:
            lines.append(f"  Debt/Equity:     {bs['debt_to_equity']:.2f}x")
        if bs.get("current_ratio") is not None:
            lines.append(f"  Current Ratio:   {bs['current_ratio']:.2f}x")
        lines.append("")

    if not inc and not cf and not bs:
        lines.append("No financial data available for this ticker.")

    return "\n".join(lines)


def format_earnings_report(data: Dict) -> str:
    """Render the earnings analysis dict as a formatted text report.

    Args:
        data: Dict returned by :func:`get_earnings_analysis`.

    Returns:
        Multi-line string ready for Telegram or console output.
    """
    ticker = data.get("ticker", "?")
    lines: List[str] = [f"EARNINGS ANALYSIS: {ticker}", ""]

    # ---- Next earnings date ----
    next_date = data.get("next_earnings_date")
    if next_date:
        lines.append(f"Next Earnings Date: {next_date}")
        lines.append("")

    # ---- EPS history ----
    history = data.get("earnings_history", [])
    if history:
        lines.append("EARNINGS HISTORY (Recent Quarters)")
        for r in history[:6]:
            rep = f"${r['reported_eps']:.2f}" if r.get("reported_eps") is not None else "N/A"
            est = f"${r['estimated_eps']:.2f}" if r.get("estimated_eps") is not None else "N/A"
            sup = r.get("surprise_pct")
            sup_str = ""
            if sup is not None:
                sup_str = f"  {'BEAT' if sup > 0 else 'MISS'} {'+' if sup > 0 else ''}{sup:.1f}%"
            lines.append(f"  {r['date']}  EPS: {rep}  Est: {est}{sup_str}")
        lines.append("")

    # ---- EPS trend ----
    eps_trend = data.get("eps_trend") or {}
    if eps_trend:
        beat_rate = eps_trend.get("beat_rate_pct")
        avg_sup = eps_trend.get("avg_surprise_pct")
        beats = eps_trend.get("beats", 0)
        misses = eps_trend.get("misses", 0)
        lines.append("EPS BEAT/MISS SUMMARY")
        lines.append(f"  Beats: {beats}  Misses: {misses}"
                     + (f"  Beat Rate: {beat_rate:.0f}%" if beat_rate is not None else ""))
        if avg_sup is not None:
            lines.append(f"  Avg Surprise: {'+' if avg_sup >= 0 else ''}{avg_sup:.1f}%")
        lines.append("")

    # ---- Revenue trend ----
    rev_trend = data.get("revenue_trend") or {}
    if rev_trend.get("periods"):
        lines.append("REVENUE TREND (Annual)")
        for r in rev_trend["periods"][:4]:
            lines.append(f"  {r['period']}:  {_fmt_val(r['revenue'])}")
        growth = rev_trend.get("latest_yoy_growth_pct")
        if growth is not None:
            lines.append(f"  YoY Growth: {'+' if growth >= 0 else ''}{growth:.1f}%")
        lines.append("")

    # ---- Analyst estimates ----
    analyst = data.get("analyst_estimates", {})
    if analyst.get("target_price_mean"):
        lines.append("ANALYST ESTIMATES")
        lines.append(f"  Consensus: {analyst.get('recommendation', 'N/A')}")
        low = analyst.get("target_price_low")
        high = analyst.get("target_price_high")
        low_str = f"${low:.2f}" if isinstance(low, (int, float)) else "N/A"
        high_str = f"${high:.2f}" if isinstance(high, (int, float)) else "N/A"
        lines.append(
            f"  Price Target: ${analyst['target_price_mean']:.2f} "
            f"(Low {low_str} / High {high_str})"
        )
        fwd_eps = analyst.get("forward_eps")
        ttm_eps = analyst.get("trailing_eps")
        if fwd_eps is not None:
            lines.append(f"  Forward EPS: ${fwd_eps:.2f}")
        if ttm_eps is not None:
            lines.append(f"  Trailing EPS: ${ttm_eps:.2f}")
        lines.append("")

    # ---- AI Summary ----
    ai = data.get("ai_summary", {})
    if ai:
        lines.append("AI EARNINGS ANALYSIS")
        lines.append(f"  Fundamental Quality Score: {ai.get('quality_score', 'N/A')}/100")
        lines.append(f"  Verdict: {ai.get('verdict', 'N/A')}")
        observations = ai.get("key_observations", [])
        if observations:
            lines.append("  Key Findings:")
            for obs in observations[:5]:
                lines.append(f"    - {obs}")
        lines.append("")
        if ai.get("bull_case"):
            lines.append(f"  Bull Case: {ai['bull_case']}")
        if ai.get("bear_case"):
            lines.append(f"  Bear Case: {ai['bear_case']}")
        lines.append("")

    # ---- Guidance summary ----
    guidance = data.get("guidance_summary", "")
    if guidance:
        lines.append("GUIDANCE & DECISION SUMMARY")
        lines.append(f"  {guidance}")

    return "\n".join(lines)
