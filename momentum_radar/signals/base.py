"""
base.py – Shared dataclass models for signal modules.
"""

from dataclasses import dataclass


@dataclass
class SignalResult:
    """Standardised result returned by every signal module.

    Attributes:
        triggered: Whether the signal condition was met.
        score: Points contribution to the total score when triggered.
        details: Human-readable description of what triggered the signal.
    """

    triggered: bool
    score: int
    details: str
