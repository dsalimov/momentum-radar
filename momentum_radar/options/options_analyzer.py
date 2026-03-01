"""
options_analyzer.py - Core options market analysis engine.

Uses yfinance to fetch options chains and analyze them for:
- Unusual volume detection
- Put/call ratio
- Max pain calculation
- IV analysis
- Options flow (smart money signals)
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_options_chain(ticker: str, expiry: str = None) -> dict:
    """Fetch the full options chain for a ticker.

    Args:
        ticker: Stock ticker symbol.
        expiry: Expiry date string (YYYY-MM-DD). If None, uses nearest expiry.

    Returns:
        Dict with keys 'calls', 'puts', 'expiry', 'current_price'.
        Returns empty dict if no data is available.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required for options analysis.") from exc

    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return {}

        if expiry is None:
            expiry = expirations[0]
        elif expiry not in expirations:
            return {}

        chain = stock.option_chain(expiry)
        calls = chain.calls[
            ["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]
        ].copy()
        puts = chain.puts[
            ["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]
        ].copy()

        # Fill NaN volumes/OI with 0
        for col in ["volume", "openInterest"]:
            calls[col] = calls[col].fillna(0)
            puts[col] = puts[col].fillna(0)

        info = stock.fast_info
        _price = getattr(info, "last_price", None)
        current_price = float(_price) if _price is not None else 0.0

        return {
            "calls": calls,
            "puts": puts,
            "expiry": expiry,
            "current_price": float(current_price),
        }
    except Exception as exc:
        logger.warning("get_options_chain failed for %s: %s", ticker, exc)
        return {}


def get_unusual_volume(ticker: str) -> dict:
    """Detect unusual options activity for a specific ticker.

    Flags contracts as unusual if:
    - volume > 1000 AND volume/OI > 2.0, OR
    - volume > 5000

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Dict with 'unusual_contracts' (list, sorted by volume desc, top 10)
        and 'ticker'.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required for options analysis.") from exc

    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return {"ticker": ticker, "unusual_contracts": []}

        unusual = []
        for expiry in expirations:
            try:
                chain = stock.option_chain(expiry)
                for opt_type, df in (("call", chain.calls), ("put", chain.puts)):
                    for _, row in df.iterrows():
                        volume = float(row.get("volume") or 0)
                        oi = float(row.get("openInterest") or 0)
                        if volume == 0:
                            continue
                        ratio = volume / oi if oi > 0 else float("inf")
                        if (volume > 1000 and ratio > 2.0) or volume > 5000:
                            unusual.append(
                                {
                                    "ticker": ticker,
                                    "strike": float(row["strike"]),
                                    "expiry": expiry,
                                    "type": opt_type,
                                    "volume": volume,
                                    "openInterest": oi,
                                    "vol_oi_ratio": ratio if ratio != float("inf") else volume,
                                    "impliedVolatility": float(row.get("impliedVolatility") or 0),
                                    "lastPrice": float(row.get("lastPrice") or 0),
                                }
                            )
            except Exception as exc:
                logger.debug("Skipping expiry %s for %s: %s", expiry, ticker, exc)
                continue

        unusual.sort(key=lambda x: x["volume"], reverse=True)
        return {"ticker": ticker, "unusual_contracts": unusual[:10]}
    except Exception as exc:
        logger.warning("get_unusual_volume failed for %s: %s", ticker, exc)
        return {"ticker": ticker, "unusual_contracts": []}


def scan_unusual_volume(tickers: list, top_n: int = 10) -> list:
    """Scan multiple tickers for unusual options activity.

    Args:
        tickers: List of stock ticker symbols.
        top_n: Number of top results to return.

    Returns:
        List of unusual contract dicts sorted by volume descending.
    """
    all_unusual = []
    for ticker in tickers:
        result = get_unusual_volume(ticker)
        all_unusual.extend(result.get("unusual_contracts", []))

    all_unusual.sort(key=lambda x: x["volume"], reverse=True)
    return all_unusual[:top_n]


def get_options_summary(ticker: str) -> dict:
    """Get a comprehensive options summary for a ticker.

    Returns:
        Dict with total call/put volume, put/call ratio, most active strikes,
        max pain, highest IV contracts, and current price.
        Returns empty dict if no data available.
    """
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError as exc:
        raise ImportError("yfinance is required for options analysis.") from exc

    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return {}

        info = stock.fast_info
        _price = getattr(info, "last_price", None)
        current_price = float(_price) if _price is not None else 0.0

        all_calls = []
        all_puts = []
        for expiry in expirations:
            try:
                chain = stock.option_chain(expiry)
                calls = chain.calls.copy()
                puts = chain.puts.copy()
                calls["expiry"] = expiry
                puts["expiry"] = expiry
                for col in ["volume", "openInterest"]:
                    calls[col] = calls[col].fillna(0)
                    puts[col] = puts[col].fillna(0)
                all_calls.append(calls)
                all_puts.append(puts)
            except Exception:
                continue

        if not all_calls and not all_puts:
            return {}

        calls_df = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
        puts_df = pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()

        total_call_vol = int(calls_df["volume"].sum()) if not calls_df.empty else 0
        total_put_vol = int(puts_df["volume"].sum()) if not puts_df.empty else 0
        put_call_ratio = (total_put_vol / total_call_vol) if total_call_vol > 0 else 0.0

        if put_call_ratio < 0.7:
            interpretation = "Moderately Bullish"
        elif put_call_ratio <= 1.0:
            interpretation = "Neutral"
        elif put_call_ratio <= 1.5:
            interpretation = "Bearish"
        else:
            interpretation = "Extreme Fear (Contrarian Bullish)"

        # Most active calls/puts
        top_calls = []
        if not calls_df.empty:
            by_vol = calls_df.nlargest(5, "volume")
            for _, row in by_vol.iterrows():
                top_calls.append({
                    "strike": float(row["strike"]),
                    "expiry": row.get("expiry", ""),
                    "volume": float(row["volume"]),
                    "openInterest": float(row["openInterest"]),
                    "impliedVolatility": float(row.get("impliedVolatility") or 0),
                    "lastPrice": float(row.get("lastPrice") or 0),
                })

        top_puts = []
        if not puts_df.empty:
            by_vol = puts_df.nlargest(5, "volume")
            for _, row in by_vol.iterrows():
                top_puts.append({
                    "strike": float(row["strike"]),
                    "expiry": row.get("expiry", ""),
                    "volume": float(row["volume"]),
                    "openInterest": float(row["openInterest"]),
                    "impliedVolatility": float(row.get("impliedVolatility") or 0),
                    "lastPrice": float(row.get("lastPrice") or 0),
                })

        # Max pain using nearest expiry
        max_pain_result = get_max_pain(ticker, expiry=expirations[0])
        max_pain_strike = max_pain_result.get("max_pain_strike")

        # Highest IV contracts
        high_iv_contracts = []
        for df, opt_type in ((calls_df, "call"), (puts_df, "put")):
            if not df.empty and "impliedVolatility" in df.columns:
                top_iv = df.nlargest(3, "impliedVolatility")
                for _, row in top_iv.iterrows():
                    high_iv_contracts.append({
                        "strike": float(row["strike"]),
                        "expiry": row.get("expiry", ""),
                        "type": opt_type,
                        "impliedVolatility": float(row.get("impliedVolatility") or 0),
                    })
        high_iv_contracts.sort(key=lambda x: x["impliedVolatility"], reverse=True)

        return {
            "ticker": ticker,
            "current_price": current_price,
            "total_call_volume": total_call_vol,
            "total_put_volume": total_put_vol,
            "put_call_ratio": put_call_ratio,
            "put_call_interpretation": interpretation,
            "most_active_calls": top_calls,
            "most_active_puts": top_puts,
            "max_pain_strike": max_pain_strike,
            "highest_iv_contracts": high_iv_contracts,
            "calls_df": calls_df,
            "puts_df": puts_df,
        }
    except Exception as exc:
        logger.warning("get_options_summary failed for %s: %s", ticker, exc)
        return {}


def get_options_flow(ticker: str) -> dict:
    """Analyze options flow — what smart money is doing.

    Returns:
        Dict with net_sentiment, volume data, call/put sweeps,
        dollar-weighted flows, and smart money signals.
    """
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError as exc:
        raise ImportError("yfinance is required for options analysis.") from exc

    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return {}

        info = stock.fast_info
        _price = getattr(info, "last_price", None)
        current_price = float(_price) if _price is not None else 0.0

        all_calls = []
        all_puts = []
        for expiry in expirations:
            try:
                chain = stock.option_chain(expiry)
                calls = chain.calls.copy()
                puts = chain.puts.copy()
                calls["expiry"] = expiry
                puts["expiry"] = expiry
                for col in ["volume", "openInterest"]:
                    calls[col] = calls[col].fillna(0)
                    puts[col] = puts[col].fillna(0)
                all_calls.append(calls)
                all_puts.append(puts)
            except Exception:
                continue

        if not all_calls and not all_puts:
            return {}

        calls_df = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
        puts_df = pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()

        total_call_vol = float(calls_df["volume"].sum()) if not calls_df.empty else 0.0
        total_put_vol = float(puts_df["volume"].sum()) if not puts_df.empty else 0.0

        # Dollar-weighted flows
        if not calls_df.empty:
            dollar_call = float(
                (calls_df["volume"] * calls_df.get("lastPrice", 0).fillna(0) * 100).sum()
            )
        else:
            dollar_call = 0.0

        if not puts_df.empty:
            dollar_put = float(
                (puts_df["volume"] * puts_df.get("lastPrice", 0).fillna(0) * 100).sum()
            )
        else:
            dollar_put = 0.0

        # Call sweeps: volume > OI (large block trades)
        call_sweeps = []
        if not calls_df.empty:
            sweeps = calls_df[
                (calls_df["volume"] > calls_df["openInterest"]) & (calls_df["volume"] > 1000)
            ].copy()
            sweeps["vol_oi_ratio"] = sweeps["volume"] / sweeps["openInterest"].replace(0, 1)
            sweeps = sweeps.nlargest(5, "volume")
            for _, row in sweeps.iterrows():
                call_sweeps.append({
                    "ticker": ticker,
                    "strike": float(row["strike"]),
                    "expiry": row.get("expiry", ""),
                    "type": "call",
                    "volume": float(row["volume"]),
                    "openInterest": float(row["openInterest"]),
                    "vol_oi_ratio": float(row["vol_oi_ratio"]),
                    "impliedVolatility": float(row.get("impliedVolatility") or 0),
                    "lastPrice": float(row.get("lastPrice") or 0),
                })

        put_sweeps = []
        if not puts_df.empty:
            sweeps = puts_df[
                (puts_df["volume"] > puts_df["openInterest"]) & (puts_df["volume"] > 1000)
            ].copy()
            sweeps["vol_oi_ratio"] = sweeps["volume"] / sweeps["openInterest"].replace(0, 1)
            sweeps = sweeps.nlargest(5, "volume")
            for _, row in sweeps.iterrows():
                put_sweeps.append({
                    "ticker": ticker,
                    "strike": float(row["strike"]),
                    "expiry": row.get("expiry", ""),
                    "type": "put",
                    "volume": float(row["volume"]),
                    "openInterest": float(row["openInterest"]),
                    "vol_oi_ratio": float(row["vol_oi_ratio"]),
                    "impliedVolatility": float(row.get("impliedVolatility") or 0),
                    "lastPrice": float(row.get("lastPrice") or 0),
                })

        net_sentiment = "bullish" if total_call_vol > total_put_vol else "bearish"
        if total_call_vol > 0 and total_put_vol > 0:
            ratio = total_call_vol / total_put_vol
            if ratio > 1.5:
                net_sentiment = "strongly bullish"
            elif ratio > 1.1:
                net_sentiment = "bullish"
            elif ratio < 0.67:
                net_sentiment = "strongly bearish"
            elif ratio < 0.9:
                net_sentiment = "bearish"
            else:
                net_sentiment = "neutral"

        smart_signals = []
        if call_sweeps:
            top_call = call_sweeps[0]
            smart_signals.append(
                f"Heavy call buying above ${top_call['strike']:.0f} strike"
            )
        if put_sweeps:
            top_put = put_sweeps[0]
            smart_signals.append(
                f"Put activity at ${top_put['strike']:.0f} strike"
            )
        if dollar_call > dollar_put * 1.5:
            smart_signals.append("Dollar flow strongly favors calls (bullish bias)")
        elif dollar_put > dollar_call * 1.5:
            smart_signals.append("Dollar flow strongly favors puts (bearish bias)")

        return {
            "ticker": ticker,
            "current_price": current_price,
            "total_call_volume": total_call_vol,
            "total_put_volume": total_put_vol,
            "dollar_call_flow": dollar_call,
            "dollar_put_flow": dollar_put,
            "call_sweeps": call_sweeps,
            "put_sweeps": put_sweeps,
            "net_sentiment": net_sentiment,
            "smart_money_signals": smart_signals,
        }
    except Exception as exc:
        logger.warning("get_options_flow failed for %s: %s", ticker, exc)
        return {}


def get_put_call_ratio(ticker: str) -> dict:
    """Calculate put/call ratio with interpretation.

    Returns:
        Dict with volume_pc_ratio, oi_pc_ratio, interpretation, description.
    """
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError as exc:
        raise ImportError("yfinance is required for options analysis.") from exc

    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return {}

        total_call_vol = 0.0
        total_put_vol = 0.0
        total_call_oi = 0.0
        total_put_oi = 0.0

        for expiry in expirations:
            try:
                chain = stock.option_chain(expiry)
                total_call_vol += float(chain.calls["volume"].fillna(0).sum())
                total_put_vol += float(chain.puts["volume"].fillna(0).sum())
                total_call_oi += float(chain.calls["openInterest"].fillna(0).sum())
                total_put_oi += float(chain.puts["openInterest"].fillna(0).sum())
            except Exception:
                continue

        vol_pcr = total_put_vol / total_call_vol if total_call_vol > 0 else 0.0
        oi_pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0.0

        if vol_pcr < 0.7:
            interpretation = "Bullish sentiment"
            description = f"Call volume exceeds put volume — market leaning bullish on {ticker}."
        elif vol_pcr <= 1.0:
            interpretation = "Neutral"
            description = f"Balanced options activity on {ticker}."
        elif vol_pcr <= 1.5:
            interpretation = "Bearish sentiment (or hedging)"
            description = f"Put volume exceeds call volume — market cautious on {ticker}."
        else:
            interpretation = "Extreme fear (contrarian bullish signal)"
            description = f"Very high put volume on {ticker} — potential contrarian buy signal."

        return {
            "ticker": ticker,
            "volume_pc_ratio": vol_pcr,
            "oi_pc_ratio": oi_pcr,
            "total_call_volume": total_call_vol,
            "total_put_volume": total_put_vol,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "interpretation": interpretation,
            "description": description,
        }
    except Exception as exc:
        logger.warning("get_put_call_ratio failed for %s: %s", ticker, exc)
        return {}


def get_max_pain(ticker: str, expiry: str = None) -> dict:
    """Calculate max pain for a specific expiry.

    Max pain = the strike price where option holders collectively lose the most money
    (i.e., the strike with the highest total dollar value of worthless options).

    Args:
        ticker: Stock ticker symbol.
        expiry: Expiry date string (YYYY-MM-DD). If None, uses nearest expiry.

    Returns:
        Dict with max_pain_strike, current_price, distance, percentage_distance, expiry.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required for options analysis.") from exc

    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return {}

        if expiry is None:
            expiry = expirations[0]
        elif expiry not in expirations:
            return {}

        chain = stock.option_chain(expiry)
        calls = chain.calls.copy()
        puts = chain.puts.copy()

        for col in ["openInterest"]:
            calls[col] = calls[col].fillna(0)
            puts[col] = puts[col].fillna(0)

        all_strikes = sorted(
            set(calls["strike"].tolist()) | set(puts["strike"].tolist())
        )
        if not all_strikes:
            return {}

        max_pain_strike = None
        max_pain_value = -1.0

        for strike in all_strikes:
            # Call pain: sum of max(0, strike_price - call_strike) * OI for all calls below this strike
            call_pain = float(
                calls[calls["strike"] < strike].apply(
                    lambda row: max(0.0, strike - row["strike"]) * row["openInterest"], axis=1
                ).sum()
            )
            # Put pain: sum of max(0, put_strike - strike_price) * OI for all puts above this strike
            put_pain = float(
                puts[puts["strike"] > strike].apply(
                    lambda row: max(0.0, row["strike"] - strike) * row["openInterest"], axis=1
                ).sum()
            )
            total_pain = call_pain + put_pain
            if total_pain > max_pain_value:
                max_pain_value = total_pain
                max_pain_strike = strike

        info = stock.fast_info
        _price = getattr(info, "last_price", None)
        current_price = float(_price) if _price is not None else 0.0

        distance = current_price - max_pain_strike if max_pain_strike is not None else 0.0
        pct_distance = (distance / max_pain_strike * 100) if max_pain_strike else 0.0

        return {
            "ticker": ticker,
            "expiry": expiry,
            "max_pain_strike": max_pain_strike,
            "current_price": current_price,
            "distance": distance,
            "percentage_distance": pct_distance,
        }
    except Exception as exc:
        logger.warning("get_max_pain failed for %s: %s", ticker, exc)
        return {}


def get_iv_analysis(ticker: str) -> dict:
    """Implied volatility analysis across expiries and strikes.

    Returns:
        Dict with atm_iv, skew_description, term_structure_description,
        highest_iv_contracts.
    """
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError as exc:
        raise ImportError("yfinance is required for options analysis.") from exc

    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return {}

        info = stock.fast_info
        _price = getattr(info, "last_price", None)
        current_price = float(_price) if _price is not None else 0.0

        # Collect ATM IV per expiry for term structure
        atm_ivs = []
        all_contracts = []

        for expiry in expirations:
            try:
                chain = stock.option_chain(expiry)
                calls = chain.calls.copy()
                puts = chain.puts.copy()
                calls["expiry"] = expiry
                calls["type"] = "call"
                puts["expiry"] = expiry
                puts["type"] = "put"

                if current_price > 0 and not calls.empty:
                    # Find ATM call (closest strike to current price)
                    calls["dist"] = abs(calls["strike"] - current_price)
                    atm_call = calls.nsmallest(1, "dist")
                    if not atm_call.empty:
                        iv_val = float(atm_call.iloc[0].get("impliedVolatility") or 0)
                        if iv_val > 0:
                            atm_ivs.append({"expiry": expiry, "atm_iv": iv_val})

                for col in ["volume", "openInterest"]:
                    calls[col] = calls[col].fillna(0)
                    puts[col] = puts[col].fillna(0)

                all_contracts.append(calls[["strike", "expiry", "type", "impliedVolatility", "volume"]])
                all_contracts.append(puts[["strike", "expiry", "type", "impliedVolatility", "volume"]])
            except Exception:
                continue

        if not all_contracts:
            return {}

        all_df = pd.concat(all_contracts, ignore_index=True)
        all_df["impliedVolatility"] = all_df["impliedVolatility"].fillna(0)

        atm_iv = atm_ivs[0]["atm_iv"] if atm_ivs else 0.0

        # IV skew: compare OTM put IV vs OTM call IV for nearest expiry
        skew_description = "N/A"
        if len(atm_ivs) > 0 and current_price > 0:
            nearest_expiry = expirations[0]
            near_df = all_df[all_df["expiry"] == nearest_expiry]
            otm_calls = near_df[
                (near_df["type"] == "call") & (near_df["strike"] > current_price * 1.05)
            ]
            otm_puts = near_df[
                (near_df["type"] == "put") & (near_df["strike"] < current_price * 0.95)
            ]
            avg_otm_call_iv = float(otm_calls["impliedVolatility"].mean()) if not otm_calls.empty else 0.0
            avg_otm_put_iv = float(otm_puts["impliedVolatility"].mean()) if not otm_puts.empty else 0.0
            diff = avg_otm_put_iv - avg_otm_call_iv
            if avg_otm_put_iv > 0 and avg_otm_call_iv > 0:
                if diff > 0.02:
                    skew_description = (
                        f"Puts trading at {diff*100:.1f}% premium to calls (mild fear/hedging)"
                    )
                elif diff < -0.02:
                    skew_description = (
                        f"Calls trading at {abs(diff)*100:.1f}% premium to puts (speculation)"
                    )
                else:
                    skew_description = "Symmetric (no significant skew)"

        # Term structure
        term_structure_description = "N/A"
        if len(atm_ivs) >= 2:
            near_iv = atm_ivs[0]["atm_iv"]
            far_iv = atm_ivs[-1]["atm_iv"]
            if near_iv > far_iv:
                term_structure_description = (
                    f"Inverted (near-term {near_iv*100:.1f}% > far-term {far_iv*100:.1f}%) "
                    f"— event expected soon"
                )
            else:
                term_structure_description = (
                    f"Normal (near-term {near_iv*100:.1f}% < far-term {far_iv*100:.1f}%)"
                )

        # Highest IV contracts
        high_iv = all_df.nlargest(10, "impliedVolatility")
        highest_iv_contracts = []
        for _, row in high_iv.iterrows():
            highest_iv_contracts.append({
                "strike": float(row["strike"]),
                "expiry": row["expiry"],
                "type": row["type"],
                "impliedVolatility": float(row["impliedVolatility"]),
            })

        return {
            "ticker": ticker,
            "current_price": current_price,
            "atm_iv": atm_iv,
            "atm_iv_by_expiry": atm_ivs,
            "skew_description": skew_description,
            "term_structure_description": term_structure_description,
            "highest_iv_contracts": highest_iv_contracts,
        }
    except Exception as exc:
        logger.warning("get_iv_analysis failed for %s: %s", ticker, exc)
        return {}
