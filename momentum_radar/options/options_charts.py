"""
options_charts.py - Chart generation for options market data.

Generates dark-themed charts matching the pattern chart style:
- Volume bar charts (calls vs puts)
- Open interest distribution charts
- IV skew charts
"""

import logging
import os
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)


def generate_volume_chart(
    ticker: str, options_data: dict, output_path: Optional[str] = None
) -> str:
    """Generate a bar chart showing call vs put volume by strike price.

    Args:
        ticker: Stock ticker symbol.
        options_data: Dict returned by get_options_summary() containing
                      'calls_df', 'puts_df', 'current_price', 'max_pain_strike'.
        output_path: Optional path to save the PNG. If None, a temp file is used.

    Returns:
        Absolute path to the generated PNG file.
    """
    try:
        import matplotlib
        try:
            matplotlib.use("Agg")
        except Exception:
            pass
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for options chart generation."
        ) from exc

    calls_df = options_data.get("calls_df")
    puts_df = options_data.get("puts_df")
    current_price = options_data.get("current_price", 0)
    max_pain_strike = options_data.get("max_pain_strike")

    if calls_df is None or puts_df is None:
        raise ValueError("options_data must contain 'calls_df' and 'puts_df'.")

    # Aggregate volume by strike
    call_vol = calls_df.groupby("strike")["volume"].sum()
    put_vol = puts_df.groupby("strike")["volume"].sum()
    all_strikes = sorted(set(call_vol.index) | set(put_vol.index))

    # Limit to strikes near current price for readability
    if current_price > 0:
        near_strikes = [s for s in all_strikes if 0.7 * current_price <= s <= 1.3 * current_price]
        if len(near_strikes) >= 5:
            all_strikes = near_strikes

    call_vols = [float(call_vol.get(s, 0)) for s in all_strikes]
    put_vols = [float(put_vol.get(s, 0)) for s in all_strikes]

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png", prefix=f"options_vol_{ticker}_")
        os.close(fd)

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    x = np.arange(len(all_strikes))
    width = 0.4

    ax.bar(x - width / 2, call_vols, width=width, color="#00c853", alpha=0.85, label="Calls")
    ax.bar(x + width / 2, put_vols, width=width, color="#ff1744", alpha=0.85, label="Puts")

    # Current price vertical line
    if current_price > 0 and all_strikes:
        closest_idx = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - current_price))
        ax.axvline(x=closest_idx, color="white", linestyle="-", linewidth=1.2, label=f"Price ${current_price:.2f}")

    # Max pain dashed line
    if max_pain_strike is not None and all_strikes:
        mp_idx = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - max_pain_strike))
        ax.axvline(x=mp_idx, color="yellow", linestyle="--", linewidth=1.2, label=f"Max Pain ${max_pain_strike:.2f}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"${s:.0f}" for s in all_strikes], rotation=45, ha="right", color="white", fontsize=7)
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.set_title(f"{ticker} — Options Volume by Strike", color="white", fontsize=13)
    ax.set_xlabel("Strike Price", color="white")
    ax.set_ylabel("Volume (contracts)", color="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0e1117")
    plt.close(fig)

    logger.info("Options volume chart saved: %s", output_path)
    return output_path


def generate_oi_chart(
    ticker: str, options_data: dict, output_path: Optional[str] = None
) -> str:
    """Generate a bar chart showing call vs put open interest by strike.

    Args:
        ticker: Stock ticker symbol.
        options_data: Dict returned by get_options_summary() containing
                      'calls_df', 'puts_df', 'current_price', 'max_pain_strike'.
        output_path: Optional path to save the PNG. If None, a temp file is used.

    Returns:
        Absolute path to the generated PNG file.
    """
    try:
        import matplotlib
        try:
            matplotlib.use("Agg")
        except Exception:
            pass
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for options chart generation."
        ) from exc

    calls_df = options_data.get("calls_df")
    puts_df = options_data.get("puts_df")
    current_price = options_data.get("current_price", 0)
    max_pain_strike = options_data.get("max_pain_strike")

    if calls_df is None or puts_df is None:
        raise ValueError("options_data must contain 'calls_df' and 'puts_df'.")

    call_oi = calls_df.groupby("strike")["openInterest"].sum()
    put_oi = puts_df.groupby("strike")["openInterest"].sum()
    all_strikes = sorted(set(call_oi.index) | set(put_oi.index))

    if current_price > 0:
        near_strikes = [s for s in all_strikes if 0.7 * current_price <= s <= 1.3 * current_price]
        if len(near_strikes) >= 5:
            all_strikes = near_strikes

    call_ois = [float(call_oi.get(s, 0)) for s in all_strikes]
    put_ois = [float(put_oi.get(s, 0)) for s in all_strikes]

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png", prefix=f"options_oi_{ticker}_")
        os.close(fd)

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    x = np.arange(len(all_strikes))
    width = 0.4

    ax.bar(x - width / 2, call_ois, width=width, color="#00c853", alpha=0.85, label="Call OI")
    ax.bar(x + width / 2, put_ois, width=width, color="#ff1744", alpha=0.85, label="Put OI")

    if current_price > 0 and all_strikes:
        closest_idx = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - current_price))
        ax.axvline(x=closest_idx, color="white", linestyle="-", linewidth=1.2, label=f"Price ${current_price:.2f}")

    if max_pain_strike is not None and all_strikes:
        mp_idx = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - max_pain_strike))
        ax.axvline(x=mp_idx, color="yellow", linestyle="--", linewidth=1.2, label=f"Max Pain ${max_pain_strike:.2f}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"${s:.0f}" for s in all_strikes], rotation=45, ha="right", color="white", fontsize=7)
    ax.tick_params(colors="white")
    ax.set_title(f"{ticker} — Open Interest by Strike", color="white", fontsize=13)
    ax.set_xlabel("Strike Price", color="white")
    ax.set_ylabel("Open Interest (contracts)", color="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0e1117")
    plt.close(fig)

    logger.info("Options OI chart saved: %s", output_path)
    return output_path


def generate_iv_skew_chart(
    ticker: str, iv_data: dict, output_path: Optional[str] = None
) -> str:
    """Generate an IV skew chart showing implied volatility across strike prices.

    Args:
        ticker: Stock ticker symbol.
        iv_data: Dict returned by get_iv_analysis() containing
                 'highest_iv_contracts' and 'current_price'.
        output_path: Optional path to save the PNG. If None, a temp file is used.

    Returns:
        Absolute path to the generated PNG file.
    """
    try:
        import matplotlib
        try:
            matplotlib.use("Agg")
        except Exception:
            pass
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for options chart generation."
        ) from exc

    current_price = iv_data.get("current_price", 0)
    contracts = iv_data.get("highest_iv_contracts", [])
    atm_iv = iv_data.get("atm_iv", 0)

    if not contracts:
        raise ValueError("iv_data must contain 'highest_iv_contracts'.")

    # Sort by strike
    contracts_sorted = sorted(contracts, key=lambda c: c["strike"])
    strikes = [c["strike"] for c in contracts_sorted]
    ivs = [c["impliedVolatility"] * 100 for c in contracts_sorted]

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png", prefix=f"options_iv_{ticker}_")
        os.close(fd)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    ax.plot(strikes, ivs, color="#00bcd4", linewidth=2, marker="o", markersize=5, label="IV")

    # ATM IV reference line
    if atm_iv > 0:
        ax.axhline(y=atm_iv * 100, color="yellow", linestyle="--", linewidth=1, label=f"ATM IV {atm_iv*100:.1f}%")

    # Current price vertical line
    if current_price > 0:
        ax.axvline(x=current_price, color="white", linestyle="-", linewidth=1.2, label=f"Price ${current_price:.2f}")

    ax.tick_params(colors="white")
    ax.set_title(f"{ticker} — IV Skew", color="white", fontsize=13)
    ax.set_xlabel("Strike Price", color="white")
    ax.set_ylabel("Implied Volatility (%)", color="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0e1117")
    plt.close(fig)

    logger.info("IV skew chart saved: %s", output_path)
    return output_path
