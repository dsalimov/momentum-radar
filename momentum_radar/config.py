"""
config.py – Centralised configuration for Momentum Signal Radar.

All thresholds, API keys, and tunable parameters are loaded here.
Sensitive values are read from environment variables (via .env file).
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _float(key: str, default: float) -> float:
    return float(os.getenv(key, default))


def _int(key: str, default: int) -> int:
    return int(os.getenv(key, default))


def _str(key: str, default: str = "") -> str:
    return os.getenv(key, default)


@dataclass
class TelegramConfig:
    """Telegram alert delivery configuration."""

    bot_token: str = field(default_factory=lambda: _str("TELEGRAM_BOT_TOKEN"))
    chat_id: str = field(default_factory=lambda: _str("TELEGRAM_CHAT_ID"))
    max_retries: int = 3
    retry_delay: float = 2.0


@dataclass
class DataConfig:
    """Data provider configuration."""

    provider: str = field(default_factory=lambda: _str("DATA_PROVIDER", "finnhub"))
    finnhub_api_key: Optional[str] = field(
        default_factory=lambda: _str("FINNHUB_API_KEY") or None
    )
    polygon_api_key: Optional[str] = field(
        default_factory=lambda: _str("POLYGON_API_KEY") or None
    )
    alpaca_api_key: Optional[str] = field(
        default_factory=lambda: _str("ALPACA_API_KEY") or None
    )
    alpaca_secret_key: Optional[str] = field(
        default_factory=lambda: _str("ALPACA_SECRET_KEY") or None
    )


@dataclass
class SignalThresholds:
    """All signal detection thresholds."""

    # Volume
    volume_spike_strong: float = field(
        default_factory=lambda: _float("VOLUME_SPIKE_STRONG", 2.0)
    )
    volume_spike_moderate: float = field(
        default_factory=lambda: _float("VOLUME_SPIKE_MODERATE", 1.5)
    )
    daily_volume_ratio: float = field(
        default_factory=lambda: _float("DAILY_VOLUME_RATIO", 1.5)
    )
    intraday_volume_ratio: float = field(
        default_factory=lambda: _float("INTRADAY_VOLUME_RATIO", 3.0)
    )
    rvol_moderate: float = field(
        default_factory=lambda: _float("RVOL_MODERATE", 2.0)
    )
    rvol_strong: float = field(
        default_factory=lambda: _float("RVOL_STRONG", 3.0)
    )

    # Volatility
    atr_ratio_moderate: float = field(
        default_factory=lambda: _float("ATR_RATIO_MODERATE", 1.5)
    )
    atr_ratio_strong: float = field(
        default_factory=lambda: _float("ATR_RATIO_STRONG", 2.0)
    )

    # Short Interest
    short_interest_min: float = field(
        default_factory=lambda: _float("SHORT_INTEREST_MIN", 0.15)
    )
    days_to_cover_min: float = field(
        default_factory=lambda: _float("DAYS_TO_COVER_MIN", 3.0)
    )
    float_max: float = field(
        default_factory=lambda: _float("FLOAT_MAX", 200_000_000)
    )

    # Options
    options_volume_ratio: float = field(
        default_factory=lambda: _float("OPTIONS_VOLUME_RATIO", 3.0)
    )

    # Golden Sweep detection
    golden_sweep_min_contracts: int = field(
        default_factory=lambda: _int("GOLDEN_SWEEP_MIN_CONTRACTS", 500)
    )
    golden_sweep_volume_spike_min: float = field(
        default_factory=lambda: _float("GOLDEN_SWEEP_VOLUME_SPIKE_MIN", 3.0)
    )
    golden_sweep_rvol_min: float = field(
        default_factory=lambda: _float("GOLDEN_SWEEP_RVOL_MIN", 1.5)
    )
    golden_sweep_weekly_days: int = field(
        default_factory=lambda: _int("GOLDEN_SWEEP_WEEKLY_DAYS", 7)
    )
    golden_sweep_swing_days: int = field(
        default_factory=lambda: _int("GOLDEN_SWEEP_SWING_DAYS", 21)
    )

    # Supply & Demand Zones
    sd_zone_min_score: float = field(
        default_factory=lambda: _float("SD_ZONE_MIN_SCORE", 50.0)
    )
    sd_impulse_atr_mult: float = field(
        default_factory=lambda: _float("SD_IMPULSE_ATR_MULT", 1.20)
    )
    sd_impulse_vol_mult: float = field(
        default_factory=lambda: _float("SD_IMPULSE_VOL_MULT", 1.30)
    )
    sd_base_atr_mult: float = field(
        default_factory=lambda: _float("SD_BASE_ATR_MULT", 0.80)
    )

    # Support/resistance touch tolerance (fraction of price)
    sr_touch_tolerance: float = field(
        default_factory=lambda: _float("SR_TOUCH_TOLERANCE", 0.02)
    )

    # Volatility squeeze thresholds
    squeeze_bb_threshold: float = field(
        default_factory=lambda: _float("SQUEEZE_BB_THRESHOLD", 0.04)
    )
    squeeze_expansion_ratio: float = field(
        default_factory=lambda: _float("SQUEEZE_EXPANSION_RATIO", 1.10)
    )

    # Chop filter: if day range / ATR is below this ratio the market is ranging
    chop_range_multiplier: float = field(
        default_factory=lambda: _float("CHOP_RANGE_MULTIPLIER", 0.5)
    )


@dataclass
class ScoreThresholds:
    """Score-to-alert-level mapping."""

    watchlist: int = field(default_factory=lambda: _int("SCORE_WATCHLIST", 4))
    high_priority: int = field(default_factory=lambda: _int("SCORE_HIGH_PRIORITY", 6))
    strong_momentum: int = field(
        default_factory=lambda: _int("SCORE_STRONG_MOMENTUM", 8)
    )
    alert_minimum: int = field(default_factory=lambda: _int("SCORE_ALERT_MINIMUM", 5))
    # Minimum confidence % for the advanced alert engine (0–100)
    min_confidence_pct: float = field(
        default_factory=lambda: _float("MIN_CONFIDENCE_PCT", 70.0)
    )
    # High-probability signal gate: weighted score must be >= this value (0-100+ scale)
    signal_score_minimum: int = field(
        default_factory=lambda: _int("SIGNAL_SCORE_MINIMUM", 75)
    )
    # Minimum number of independent modules that must fire before sending an alert
    min_signal_confirmations: int = field(
        default_factory=lambda: _int("MIN_SIGNAL_CONFIRMATIONS", 3)
    )


@dataclass
class ScanConfig:
    """Scanning behaviour configuration."""

    interval_seconds: int = field(
        default_factory=lambda: _int("SCAN_INTERVAL", 60)
    )
    alert_cooldown_seconds: int = field(
        # 900 s = 15 minutes – no duplicate signals per ticker/setup within this window
        default_factory=lambda: _int("ALERT_COOLDOWN", 900)
    )


@dataclass
class UniverseConfig:
    """Stock universe filter configuration."""

    min_price: float = field(default_factory=lambda: _float("MIN_PRICE", 5.0))
    min_avg_volume: int = field(
        default_factory=lambda: _int("MIN_AVG_VOLUME", 1_000_000)
    )
    universe_size: int = field(
        default_factory=lambda: _int("UNIVERSE_SIZE", 1000)
    )


@dataclass
class MarketHoursConfig:
    """Market hours and session configuration."""

    market_open: str = field(
        default_factory=lambda: _str("MARKET_OPEN", "09:35")
    )
    market_close: str = field(
        default_factory=lambda: _str("MARKET_CLOSE", "15:45")
    )
    lunch_start: str = field(
        default_factory=lambda: _str("LUNCH_START", "12:00")
    )
    lunch_end: str = field(
        default_factory=lambda: _str("LUNCH_END", "13:30")
    )
    flat_market_threshold: float = 0.003  # 0.3 %


@dataclass
class PaperTradingConfig:
    """Paper trading and backtesting configuration."""

    enabled: bool = field(default_factory=lambda: _str("PAPER_TRADING", "false").lower() == "true")
    initial_capital: float = field(default_factory=lambda: _float("PAPER_CAPITAL", 100_000.0))
    max_position_size_pct: float = field(
        default_factory=lambda: _float("MAX_POSITION_SIZE_PCT", 0.10)
    )
    max_daily_loss_pct: float = field(
        default_factory=lambda: _float("MAX_DAILY_LOSS_PCT", 0.02)
    )
    risk_per_trade_pct: float = field(
        default_factory=lambda: _float("RISK_PER_TRADE_PCT", 0.01)
    )
    min_rr_ratio: float = field(
        default_factory=lambda: _float("MIN_RR_RATIO", 2.0)
    )
    confidence_threshold: float = field(
        default_factory=lambda: _float("CONFIDENCE_THRESHOLD", 60.0)
    )


@dataclass
class AppConfig:
    """Top-level application configuration."""

    log_level: str = field(default_factory=lambda: _str("LOG_LEVEL", "INFO"))
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    data: DataConfig = field(default_factory=DataConfig)
    signals: SignalThresholds = field(default_factory=SignalThresholds)
    scores: ScoreThresholds = field(default_factory=ScoreThresholds)
    scan: ScanConfig = field(default_factory=ScanConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    market_hours: MarketHoursConfig = field(default_factory=MarketHoursConfig)
    paper_trading: PaperTradingConfig = field(default_factory=PaperTradingConfig)


# Singleton config instance used throughout the application
config = AppConfig()
