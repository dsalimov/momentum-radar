"""
core/risk_engine.py – Strategy-aware risk management engine.

Wraps :mod:`momentum_radar.utils.risk` and provides stop and target
suggestions tailored to each strategy's risk model:

* **Scalp**          – tight stops (0.75× ATR), R:R ≥ 2.0
* **Intraday**       – medium stops (1.25× ATR), R:R ≥ 2.5
* **Swing**          – wider stops (1.75× ATR), R:R ≥ 3.0
* **Chart Pattern**  – medium stops (1.25× ATR), R:R ≥ 2.0
* **Unusual Volume** – medium stops (1.00× ATR), R:R ≥ 2.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from momentum_radar.utils.risk import compute_risk_reward, suggest_stop_loss

logger = logging.getLogger(__name__)

# ATR multiplier and minimum R:R per strategy
_STRATEGY_RISK_PARAMS: Dict[str, Dict] = {
    "scalp":          {"atr_mult": 0.75, "min_rr": 2.0},
    "intraday":       {"atr_mult": 1.25, "min_rr": 2.5},
    "swing":          {"atr_mult": 1.75, "min_rr": 3.0},
    "chart_pattern":  {"atr_mult": 1.25, "min_rr": 2.0},
    "unusual_volume": {"atr_mult": 1.00, "min_rr": 2.0},
}


@dataclass
class TradeParameters:
    """Calculated trade parameters for one setup.

    Attributes:
        entry:   Suggested entry price.
        stop:    Stop-loss price.
        target:  Primary take-profit price.
        target2: Secondary take-profit price (swing strategies only; 0 = not set).
        rr:      Computed risk-to-reward ratio.
        valid:   True if R:R meets the strategy minimum.
    """

    entry: float
    stop: float
    target: float
    target2: float
    rr: float
    valid: bool


def compute_trade_params(
    strategy: str,
    entry: float,
    atr: Optional[float],
    direction: str = "long",
    support: Optional[float] = None,
    resistance: Optional[float] = None,
) -> TradeParameters:
    """Compute entry / stop / target for a given strategy and entry price.

    Args:
        strategy:   Strategy key (``"scalp"`` / ``"intraday"`` / ``"swing"`` /
                    ``"chart_pattern"`` / ``"unusual_volume"``).
        entry:      Planned entry price.
        atr:        14-day ATR used for stop placement (``None`` → 1 % default).
        direction:  ``"long"`` or ``"short"``.
        support:    Optional structural support price (refines long stop).
        resistance: Optional structural resistance price (used as target).

    Returns:
        :class:`TradeParameters`.
    """
    params = _STRATEGY_RISK_PARAMS.get(strategy, _STRATEGY_RISK_PARAMS["intraday"])
    atr_mult = params["atr_mult"]
    min_rr   = params["min_rr"]

    if direction == "long":
        stop = suggest_stop_loss(
            entry_price=entry,
            atr=atr,
            atr_multiplier=atr_mult,
            support_level=support,
        )
        risk = max(entry - stop, entry * 0.001)
        if resistance and resistance > entry + risk * min_rr:
            target = float(resistance)
        else:
            target = entry + risk * min_rr
        # Second target: one extra risk unit beyond target1 for multi-target strategies
        target2 = entry + risk * (min_rr + 1.0) if strategy in ("swing", "chart_pattern") else 0.0
    else:
        # Short trade – stop above entry
        atr_value = (atr * atr_mult) if atr else entry * 0.01
        stop = entry + atr_value
        risk = max(stop - entry, entry * 0.001)
        if resistance and resistance < entry - risk * min_rr:
            target = float(resistance)
        else:
            target = entry - risk * min_rr
        target2 = entry - risk * (min_rr + 1.0) if strategy in ("swing", "chart_pattern") else 0.0

    rr = compute_risk_reward(entry, stop, target) or 0.0
    valid = rr >= min_rr

    return TradeParameters(
        entry=round(entry, 2),
        stop=round(stop, 2),
        target=round(target, 2),
        target2=round(target2, 2),
        rr=round(rr, 2),
        valid=valid,
    )


def get_min_rr(strategy: str) -> float:
    """Return the minimum R:R requirement for the given strategy.

    Args:
        strategy: Strategy key.

    Returns:
        Minimum R:R float (default 2.0 for unknown strategies).
    """
    return _STRATEGY_RISK_PARAMS.get(strategy, {}).get("min_rr", 2.0)
