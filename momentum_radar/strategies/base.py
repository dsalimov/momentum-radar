"""
strategies/base.py – Shared dataclasses for all strategy engines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class StrategySignal:
    """A tradeable signal produced by a strategy engine.

    Attributes:
        ticker:               Stock symbol.
        strategy:             Strategy name (e.g. ``"scalp"``).
        direction:            ``"BUY"`` or ``"SELL"``.
        timeframe:            Active timeframe (e.g. ``"2m"``, ``"5m"``, ``"1H"``).
        score:                Strategy score 0–100.
        grade:                Letter grade (``"A+"`` / ``"A"`` / ``"B"`` / ``"C"``).
        confirmations:        List of confirmation label strings.
        entry:                Suggested entry price.
        stop:                 Stop-loss price.
        target:               Take-profit price.
        rr:                   Computed risk-to-reward ratio.
        regime:               Market regime display string.
        htf_bias:             Higher-timeframe bias label.
        session:              Session name (e.g. ``"open"``).
        fake_breakout_passed: True if the fake-breakout filter was passed.
        valid:                True if all quality gates were satisfied.
    """

    ticker: str
    strategy: str
    direction: str = "BUY"
    timeframe: str = "5m"
    score: int = 0
    grade: str = "C"
    confirmations: List[str] = field(default_factory=list)
    entry: float = 0.0
    stop: float = 0.0
    target: float = 0.0
    rr: float = 0.0
    regime: str = "ranging"
    htf_bias: str = "Neutral"
    session: str = ""
    fake_breakout_passed: bool = False
    valid: bool = False

    @property
    def confirmation_count(self) -> int:
        """Number of confirmations that fired."""
        return len(self.confirmations)
