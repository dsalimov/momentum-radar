"""
options_flow.py – Options activity signal detection (optional module).

Registered signals
------------------
- ``options_flow`` – detects unusual call/put volume vs. average
"""

import logging
from typing import Dict, Optional

import pandas as pd

from momentum_radar.config import config
from momentum_radar.signals.base import SignalResult
from momentum_radar.signals.scoring import register_signal

logger = logging.getLogger(__name__)


@register_signal("options_flow")
def options_flow(
    ticker: str,
    options: Optional[Dict],
    **kwargs,
) -> SignalResult:
    """Detect unusual options activity.

    Trigger conditions:
    - Call or put volume ≥ ``OPTIONS_VOLUME_RATIO`` × average → +2

    Args:
        ticker: Stock symbol.
        options: Dict from
            :meth:`~momentum_radar.data.data_fetcher.BaseDataFetcher.get_options_volume`.

    Returns:
        :class:`~momentum_radar.signals.base.SignalResult`
    """
    cfg = config.signals

    if options is None:
        return SignalResult(triggered=False, score=0, details="No options data")

    call_vol = options.get("call_volume", 0) or 0
    put_vol = options.get("put_volume", 0) or 0
    avg_call = options.get("avg_call_volume", 1) or 1
    avg_put = options.get("avg_put_volume", 1) or 1

    call_ratio = call_vol / avg_call if avg_call > 0 else 0
    put_ratio = put_vol / avg_put if avg_put > 0 else 0

    if call_ratio >= cfg.options_volume_ratio or put_ratio >= cfg.options_volume_ratio:
        side = "calls" if call_ratio >= put_ratio else "puts"
        ratio = max(call_ratio, put_ratio)
        return SignalResult(
            triggered=True,
            score=2,
            details=f"Unusual {side} volume ({ratio:.1f}x avg)",
        )

    return SignalResult(
        triggered=False,
        score=0,
        details=f"Options flow normal (call {call_ratio:.1f}x, put {put_ratio:.1f}x)",
    )
