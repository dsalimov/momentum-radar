"""
strategies/ – Isolated, modular trading strategy engines.

Each strategy module exposes a single ``evaluate()`` function that takes raw
market data and returns a :class:`~momentum_radar.strategies.base.StrategySignal`
describing the setup quality, trade parameters, and whether all quality gates
were passed.

Strategies
----------
- :mod:`scalp_strategy`          – 2m/5m momentum scalp
- :mod:`intraday_strategy`       – 5m/10m trend + supply/demand
- :mod:`swing_strategy`          – 1H/4H/Daily HTF zone plays
- :mod:`chart_pattern_strategy`  – Pattern-based breakout engine
- :mod:`unusual_volume_strategy` – Volume spike + level break engine
"""
