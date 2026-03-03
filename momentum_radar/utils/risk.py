"""
risk.py – Risk management utilities.

Provides position sizing, stop-loss suggestion, and risk-to-reward
calculation helpers based on the configured paper trading / risk parameters.
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def compute_position_size(
    account_size: float,
    risk_per_trade_pct: float,
    entry_price: float,
    stop_loss_price: float,
) -> Tuple[int, float]:
    """Calculate optimal position size using fixed-risk method.

    Position size = (account_size × risk_per_trade_pct) / (entry − stop_loss)

    Args:
        account_size:       Total account value in dollars.
        risk_per_trade_pct: Fraction of account to risk per trade (e.g. 0.01 = 1 %).
        entry_price:        Planned entry price.
        stop_loss_price:    Planned stop-loss price.

    Returns:
        Tuple of (``shares``, ``dollar_risk``).
    """
    if entry_price <= 0 or stop_loss_price <= 0:
        return 0, 0.0
    risk_per_share = abs(entry_price - stop_loss_price)
    if risk_per_share <= 0:
        return 0, 0.0
    dollar_risk = account_size * risk_per_trade_pct
    shares = int(dollar_risk / risk_per_share)
    return shares, round(dollar_risk, 2)


def suggest_stop_loss(
    entry_price: float,
    atr: Optional[float],
    atr_multiplier: float = 1.5,
    support_level: Optional[float] = None,
) -> float:
    """Suggest a stop-loss level using ATR or support.

    Uses ATR-based stop by default (entry − ATR × multiplier).  If a
    structural support level is provided and is *closer* to the entry, it
    takes priority.

    Args:
        entry_price:     Planned entry price.
        atr:             14-day Average True Range.
        atr_multiplier:  Multiple of ATR below entry.
        support_level:   Optional structural support price.

    Returns:
        Suggested stop-loss price.
    """
    if atr and atr > 0:
        atr_stop = entry_price - atr * atr_multiplier
    else:
        atr_stop = entry_price * 0.97  # 3 % default

    if support_level and 0 < support_level < entry_price:
        # Use the tighter of the two stops
        stop = max(atr_stop, support_level * 0.99)
    else:
        stop = atr_stop

    return round(max(stop, 0.01), 2)


def compute_risk_reward(
    entry: float,
    stop_loss: float,
    target: float,
) -> Optional[float]:
    """Compute the risk-to-reward ratio for a trade.

    Args:
        entry:      Entry price.
        stop_loss:  Stop-loss price.
        target:     Profit target price.

    Returns:
        R:R ratio as a float (e.g. 2.5 means 2.5× reward per unit risk),
        or ``None`` if the inputs are invalid.
    """
    risk = abs(entry - stop_loss)
    reward = abs(target - entry)
    if risk <= 0:
        return None
    return round(reward / risk, 2)


def format_risk_summary(
    ticker: str,
    entry: float,
    stop_loss: float,
    target1: float,
    target2: Optional[float],
    shares: int,
    dollar_risk: float,
    confidence_pct: float,
) -> str:
    """Format a risk management summary string for alerts.

    Args:
        ticker:          Stock symbol.
        entry:           Entry price.
        stop_loss:       Stop-loss price.
        target1:         First profit target.
        target2:         Second profit target (optional).
        shares:          Suggested share count.
        dollar_risk:     Dollar amount at risk.
        confidence_pct:  Signal confidence percentage.

    Returns:
        Formatted multi-line string.
    """
    rr1 = compute_risk_reward(entry, stop_loss, target1)
    rr2 = compute_risk_reward(entry, stop_loss, target2) if target2 else None

    lines = [
        f"Risk Summary: {ticker}",
        f"  Entry:      ${entry:.2f}",
        f"  Stop Loss:  ${stop_loss:.2f}",
        f"  Target 1:   ${target1:.2f}  (R:R {rr1:.1f}x)" if rr1 is not None else f"  Target 1:   ${target1:.2f}",
        f"  Target 2:   ${target2:.2f}  (R:R {rr2:.1f}x)" if (target2 is not None and rr2 is not None) else "",
        f"  Shares:     {shares}",
        f"  $ at Risk:  ${dollar_risk:.2f}",
        f"  Confidence: {confidence_pct:.0f}%",
    ]
    return "\n".join(l for l in lines if l)
