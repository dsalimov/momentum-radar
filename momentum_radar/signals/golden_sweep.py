"""
golden_sweep.py – Institutional options Golden Sweep detector.

A **Golden Sweep** is an unusually large, aggressive options order (sweep or
block trade) characterised by:

* Volume exceeds open interest on the contract (strong directional conviction).
* Minimum contract count threshold (``GOLDEN_SWEEP_MIN_CONTRACTS``).
* Short-dated expiry suggesting tactical, conviction-driven flow.

Sweep classification
--------------------
- **Weekly sweep** (0–7 days to expiry) → Day Trade signal.
- **2-3 Week sweep** (8–21 days to expiry) → Swing Trade signal.
- **Monthly sweep** (22+ days to expiry) → Swing/positional context.

Underlying confirmation
-----------------------
Before a Golden Sweep is flagged as a high-probability setup the bot checks:

1. RVOL ≥ 1.5 on the underlying stock.
2. Volume spike ≥ 3× recent average on the underlying.
3. Sweep direction aligned with intraday momentum (bullish sweep → intraday
   uptrend, bearish sweep → intraday downtrend).
4. Minimum risk:reward ≥ 1.5:1 on the resulting trade.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional

import pandas as pd

from momentum_radar.config import config
from momentum_radar.signals.base import SignalResult
from momentum_radar.signals.scoring import register_signal
from momentum_radar.utils.indicators import compute_atr, compute_rvol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (overridable via config)
# ---------------------------------------------------------------------------

MIN_RISK_REWARD: float = 1.5
STOP_ATR_MULT: float = 1.0
TARGET_RR_MULT: float = 2.0


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class GoldenSweepSetup:
    ticker: str
    direction: str  # "Bullish" | "Bearish"
    sweep_type: str  # "Weekly" | "2-3 Week" | "Monthly"
    trade_type: str  # "Day Trade" | "Swing Trade"
    contract_type: str  # "Call" | "Put"
    strike: float
    expiry: str
    contracts: int
    underlying_price: float
    entry: float
    stop: float
    target: float
    rvol: float
    volume_spike: float
    supply_demand_zone: Optional[str] = None
    confidence: str = "High"
    timestamp: datetime = field(default_factory=datetime.now)
    details: str = ""

    @property
    def risk_reward(self) -> float:
        risk = abs(self.entry - self.stop)
        reward = abs(self.target - self.entry)
        if risk <= 0:
            return 0.0
        return round(reward / risk, 1)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _days_to_expiry(expiry_str: str) -> Optional[int]:
    try:
        expiry_date = date.fromisoformat(expiry_str)
        return (expiry_date - date.today()).days
    except (ValueError, TypeError):
        return None


def _classify_sweep_type(days: Optional[int]) -> tuple[str, str]:
    cfg = config.signals
    if days is None:
        return "Unknown", "Swing Trade"
    if days <= cfg.golden_sweep_weekly_days:
        return "Weekly", "Day Trade"
    if days <= cfg.golden_sweep_swing_days:
        return "2-3 Week", "Swing Trade"
    return "Monthly", "Swing Trade"


def _recent_avg_volume(bars: pd.DataFrame, lookback: int = 20) -> float:
    if "volume" not in bars.columns or len(bars) < 2:
        return 0.0
    return float(bars["volume"].iloc[-lookback - 1 : -1].mean())


def _volume_spike_mult(bars: pd.DataFrame, lookback: int = 20) -> float:
    avg = _recent_avg_volume(bars, lookback)
    if avg <= 0:
        return 0.0
    return round(float(bars["volume"].iloc[-1]) / avg, 2)


def _intraday_trend(bars: pd.DataFrame) -> str:
    if bars is None or len(bars) < 10 or "close" not in bars.columns:
        return "flat"
    closes = bars["close"]
    ema_fast = float(closes.ewm(span=5, adjust=False).mean().iloc[-1])
    ema_slow = float(closes.ewm(span=20, adjust=False).mean().iloc[-1])
    if ema_fast > ema_slow * 1.001:
        return "up"
    if ema_fast < ema_slow * 0.999:
        return "down"
    return "flat"


def _nearest_sd_zone(
    bars: pd.DataFrame,
    current_price: float,
    direction: str,
    lookback: int = 30,
    tolerance_pct: float = 0.015,
) -> Optional[str]:
    if bars is None or bars.empty:
        return None

    df = bars.tail(lookback)
    price_series = df["low"] if direction == "Bullish" else df["high"]
    threshold = current_price * tolerance_pct

    nearby = price_series[abs(price_series - current_price) <= threshold]
    if len(nearby) < 2:
        return None

    zone_low = round(float(nearby.min()), 2)
    zone_high = round(float(nearby.max()), 2)
    label = "Demand Zone" if direction == "Bullish" else "Supply Zone"
    return f"{label} {zone_low:.2f}–{zone_high:.2f}"


def _select_best_sweep(sweeps: List[Dict]) -> Optional[Dict]:
    min_contracts = config.signals.golden_sweep_min_contracts
    qualified = [s for s in sweeps if int(s.get("volume", 0)) >= min_contracts]
    if not qualified:
        return None
    return max(qualified, key=lambda s: s.get("volume", 0))


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def detect_golden_sweep(
    ticker: str,
    options_flow_data: Optional[Dict],
    bars: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
) -> Optional[GoldenSweepSetup]:
    cfg = config.signals

    if not options_flow_data:
        return None

    call_sweeps: List[Dict] = options_flow_data.get("call_sweeps", []) or []
    put_sweeps: List[Dict] = options_flow_data.get("put_sweeps", []) or []
    if not call_sweeps and not put_sweeps:
        return None

    best_call = _select_best_sweep(call_sweeps)
    best_put = _select_best_sweep(put_sweeps)
    if best_call is None and best_put is None:
        return None

    if best_call is None:
        chosen_sweep = best_put
        direction = "Bearish"
        contract_type = "Put"
    elif best_put is None:
        chosen_sweep = best_call
        direction = "Bullish"
        contract_type = "Call"
    else:
        if int(best_call.get("volume", 0)) >= int(best_put.get("volume", 0)):
            chosen_sweep = best_call
            direction = "Bullish"
            contract_type = "Call"
        else:
            chosen_sweep = best_put
            direction = "Bearish"
            contract_type = "Put"

    current_price = float(options_flow_data.get("current_price", 0.0))
    if bars is not None and not bars.empty and "close" in bars.columns:
        current_price = float(bars["close"].iloc[-1])
    if current_price <= 0:
        return None

    rvol = compute_rvol(bars, daily) if (bars is not None and daily is not None) else 0.0
    vol_spike = _volume_spike_mult(bars) if bars is not None else 0.0

    if rvol < cfg.golden_sweep_rvol_min:
        return None
    if vol_spike < cfg.golden_sweep_volume_spike_min:
        return None

    trend = _intraday_trend(bars)
    if direction == "Bullish" and trend == "down":
        return None
    if direction == "Bearish" and trend == "up":
        return None

    atr = compute_atr(daily) if daily is not None and not daily.empty else None
    if atr is None or atr <= 0:
        atr = current_price * 0.005

    entry = current_price
    if direction == "Bullish":
        stop = round(entry - STOP_ATR_MULT * atr, 2)
        target = round(entry + TARGET_RR_MULT * abs(entry - stop), 2)
    else:
        stop = round(entry + STOP_ATR_MULT * atr, 2)
        target = round(entry - TARGET_RR_MULT * abs(stop - entry), 2)

    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk <= 0:
        return None
    if reward / risk < MIN_RISK_REWARD:
        return None

    expiry_str = chosen_sweep.get("expiry", "")
    days_left = _days_to_expiry(expiry_str)
    sweep_type, trade_type = _classify_sweep_type(days_left)

    contracts = int(chosen_sweep.get("volume", 0))
    strike = float(chosen_sweep.get("strike", 0.0))

    sd_zone = _nearest_sd_zone(bars, current_price, direction)
    confidence = "High" if rvol >= 2.0 and vol_spike >= 4.0 else "Medium"

    details = (
        f"{sweep_type} {contract_type} Sweep; {contracts:,} contracts; "
        f"strike ${strike:.0f}; expiry {expiry_str}; RVOL {rvol:.1f}; "
        f"vol spike {vol_spike:.1f}x"
    )

    return GoldenSweepSetup(
        ticker=ticker,
        direction=direction,
        sweep_type=sweep_type,
        trade_type=trade_type,
        contract_type=contract_type,
        strike=strike,
        expiry=expiry_str,
        contracts=contracts,
        underlying_price=current_price,
        entry=entry,
        stop=stop,
        target=target,
        rvol=round(rvol, 1),
        volume_spike=round(vol_spike, 1),
        supply_demand_zone=sd_zone,
        confidence=confidence,
        timestamp=datetime.now(),
        details=details,
    )


# ---------------------------------------------------------------------------
# Scoring-registry wrapper
# ---------------------------------------------------------------------------

@register_signal("golden_sweep")
def golden_sweep_signal(
    ticker: str,
    options: Optional[Dict] = None,
    bars: Optional[pd.DataFrame] = None,
    daily: Optional[pd.DataFrame] = None,
    **kwargs,
) -> SignalResult:
    setup = detect_golden_sweep(ticker, options, bars, daily)

    if setup is None:
        return SignalResult(
            triggered=False,
            score=0,
            details="No golden sweep detected",
        )

    score = 3 if setup.confidence == "High" else 2
    return SignalResult(
        triggered=True,
        score=score,
        details=(
            f"{setup.sweep_type} {setup.contract_type} Sweep → {setup.direction} {setup.trade_type} "
            f"({setup.contracts:,} contracts, strike ${setup.strike:.0f})"
        ),
    )