"""
strategies/base.py – Shared dataclasses for all strategy engines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

#: Maps each strategy name to its user-facing strategy classification label.
#: Only DAY TRADE and SWING TRADE are supported.  Any former scalp-style
#: strategy is promoted to DAY TRADE.
STRATEGY_TYPE_MAP: dict = {
    "scalp":          "DAY TRADE",
    "intraday":       "DAY TRADE",
    "swing":          "SWING TRADE",
    "chart_pattern":  "SWING TRADE",
    "unusual_volume": "DAY TRADE",
}


@dataclass
class StrategySignal:
    """A tradeable signal produced by a strategy engine.

    Attributes:
        ticker:               Stock symbol.
        strategy:             Strategy name (e.g. ``"intraday"``).
        strategy_type:        User-facing classification label (``"DAY TRADE"`` or ``"SWING TRADE"``).
                              Auto-derived from *strategy* if left empty.
        direction:            ``"BUY"`` or ``"SELL"``.
        timeframe:            Active timeframe (e.g. ``"2m"``, ``"5m"``, ``"1H"``).
        score:                Strategy score 0–100.
        grade:                Letter grade (``"A+"`` / ``"A"`` / ``"B"`` / ``"C"``).
        confirmations:        List of confirmation label strings.
        entry:                Suggested entry price.
        stop:                 Stop-loss price.
        target:               Primary take-profit price.
        target2:              Secondary take-profit price (swing/chart_pattern only; 0 = not set).
        options_flow_label:   Optional human-readable options-flow annotation shown in the alert.
        rr:                   Computed risk-to-reward ratio.
        regime:               Market regime display string.
        htf_bias:             Higher-timeframe bias label.
        session:              Session name (e.g. ``"open"``).
        fake_breakout_passed: True if the fake-breakout filter was passed.
        valid:                True if all quality gates were satisfied.
    """

    ticker: str
    strategy: str
    strategy_type: str = ""
    direction: str = "BUY"
    timeframe: str = "5m"
    score: int = 0
    grade: str = "C"
    confirmations: List[str] = field(default_factory=list)
    entry: float = 0.0
    stop: float = 0.0
    target: float = 0.0
    target2: float = 0.0
    options_flow_label: str = ""
    rr: float = 0.0
    regime: str = "ranging"
    htf_bias: str = "Neutral"
    session: str = ""
    fake_breakout_passed: bool = False
    valid: bool = False

    def __post_init__(self) -> None:
        if not self.strategy_type:
            self.strategy_type = STRATEGY_TYPE_MAP.get(self.strategy, "DAY TRADE")

    @property
    def confirmation_count(self) -> int:
        """Number of confirmations that fired."""
        return len(self.confirmations)
