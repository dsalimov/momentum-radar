# Momentum Signal Radar

A **real-time stock signal scanner** that scans liquid US equities for unusual
activity indicating potential high-percentage moves.  The system detects
momentum signals, assigns weighted scores, sends alerts via Telegram and console,
and logs everything to SQLite and CSV.

> **⚠️ This system does NOT place trades. It is an alert-only tool.**

---

## Features

- 🔍 **Multi-signal detection** – volume spikes, relative volume (RVOL),
  volatility expansion, price structure breaks, short interest, and options flow
- 🏗️ **Signal registry pattern** – add new signals without touching core logic
- 📊 **Weighted scoring** – each module contributes points; total score maps to
  an alert level (`WATCHLIST`, `HIGH_PRIORITY`, `STRONG_MOMENTUM`)
- 📣 **Dual alert delivery** – Telegram Bot API + console output
- 💾 **Persistent storage** – SQLite (via SQLAlchemy) + daily CSV logs
- ⏱️ **Market-hours aware** – only scans 9:35–15:45 EST; reduced frequency
  during lunch lull (12:00–13:30)
- 📉 **Market context** – flat SPY/QQQ penalises scores by 1 to reduce noise
- 🔁 **Alert cooldown** – configurable per-ticker cooldown (default 10 min)
- 🔌 **Pluggable data providers** – abstract base class allows swapping
  `yfinance` for Polygon, Alpaca, or any other source
- ⚙️ **Fully config-driven** – all thresholds via `.env` / environment variables

---

## Project Structure

```
momentum-radar/
│
├── main.py                        ← top-level entry point
├── requirements.txt
├── .env.example                   ← copy to .env and fill in your keys
├── .gitignore
├── README.md
│
├── momentum_radar/
│   ├── __init__.py
│   ├── config.py                  ← all configuration via dataclasses + dotenv
│   ├── main.py                    ← async scan loop
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_fetcher.py        ← abstract BaseDataFetcher + YFinanceFetcher
│   │   └── universe_builder.py   ← filters the scanning universe
│   │
│   ├── signals/
│   │   ├── __init__.py
│   │   ├── base.py                ← SignalResult dataclass
│   │   ├── scoring.py             ← registry, AlertLevel enum, compute_score()
│   │   ├── volume.py              ← volume_spike + relative_volume signals
│   │   ├── volatility.py          ← volatility_expansion signal
│   │   ├── structure.py           ← structure_break signal
│   │   ├── short_interest.py      ← short_interest signal
│   │   └── options_flow.py        ← options_flow signal
│   │
│   ├── alerts/
│   │   ├── __init__.py
│   │   ├── formatter.py           ← format_alert()
│   │   └── telegram_alert.py      ← send_telegram_alert()
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── database.py            ← SQLAlchemy ORM + save_alert()
│   │   └── logger.py              ← daily CSV logger
│   │
│   └── utils/
│       ├── __init__.py
│       ├── indicators.py          ← ATR, VWAP, RVOL calculations
│       └── market_hours.py        ← session checks, lunch lull, market context
│
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_scoring.py
    ├── test_volume.py
    ├── test_market_hours.py
    └── test_formatter.py
```

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/dsalimov/momentum-radar.git
cd momentum-radar
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your values (see **Configuration Reference** below).

At minimum you need valid `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` values to
receive Telegram notifications.  All other settings have sensible defaults.

---

## Usage

### Run the scanner

```bash
python main.py
```

Or as a module:

```bash
python -m momentum_radar.main
```

The scanner will:
1. Build the stock universe (filtered by price and average volume)
2. Wait for market hours (9:35 AM – 3:45 PM EST)
3. Scan every ticker once per minute
4. Print and Telegram-deliver any alert with score ≥ 5
5. Save all alerts to `momentum_radar.db` and `logs/alerts_YYYYMMDD.csv`

### Run tests

```bash
pytest tests/ -v
```

---

## Alert Format

```
🚨 HIGH PRIORITY SIGNAL

Ticker: XYZ
Price: 42.15
% Change: +6.4%
RVOL: 2.8
Score: 7

Triggers:
  - Volume Spike: 5m vol 3.2x avg (strong)
  - Structure Break: Break of prev-day high (strong)
  - Short Interest: Short 20.0%, DTC 4.5, Float 50M

Range vs ATR: 1.9x
Float: 50M
Short Interest: 20.0%
Time: 10:42 AM EST
```

---

## Alert Levels

| Score | Level | Telegram? |
|-------|-------|-----------|
| 0–3   | IGNORE | ✗ |
| 4–5   | WATCHLIST | depends on SCORE_ALERT_MINIMUM |
| 6–7   | HIGH_PRIORITY | ✓ |
| 8+    | STRONG_MOMENTUM | ✓ |

---

## Architecture Overview

```
                         ┌─────────────────────┐
                         │   main.py (async)   │
                         └──────────┬──────────┘
                                    │
               ┌────────────────────┼────────────────────┐
               ▼                    ▼                    ▼
   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
   │  UniverseBuilder │  │  MarketHoursCheck│  │  MarketContext   │
   └──────────────────┘  └──────────────────┘  │  (SPY / QQQ)    │
               │                               └──────────────────┘
               ▼
   ┌──────────────────────────────┐
   │        Ticker Loop           │
   └──────────────┬───────────────┘
                  │  For each ticker:
                  ▼
   ┌──────────────────────────────┐
   │      BaseDataFetcher         │
   │  (YFinance / Polygon / ...)  │
   └──────────────┬───────────────┘
                  │  bars, daily, fundamentals, options
                  ▼
   ┌──────────────────────────────┐
   │       Signal Registry        │
   │  volume_spike                │
   │  relative_volume             │
   │  volatility_expansion        │
   │  structure_break             │
   │  short_interest              │
   │  options_flow                │
   └──────────────┬───────────────┘
                  │  SignalResult (triggered, score, details)
                  ▼
   ┌──────────────────────────────┐
   │       compute_score()        │
   │   total = Σ scores - penalty │
   └──────────────┬───────────────┘
                  │  score ≥ SCORE_ALERT_MINIMUM?
                  ▼
   ┌──────────────────────────────┐
   │      Alert Delivery          │
   │  format_alert()              │
   │  send_telegram_alert()       │
   │  save_alert() (SQLite)       │
   │  log_alert_csv() (CSV)       │
   └──────────────────────────────┘
```

---

## How to Add a New Signal Module

The signal registry pattern means you can add a new signal with **zero changes**
to core scanning logic.

1. Create `momentum_radar/signals/my_signal.py`:

```python
from momentum_radar.signals.base import SignalResult
from momentum_radar.signals.scoring import register_signal

@register_signal("my_signal")
def my_signal(ticker, bars, daily, fundamentals, options, **kwargs) -> SignalResult:
    """Describe what this signal detects."""
    # ... your detection logic ...
    triggered = True   # or False
    return SignalResult(triggered=triggered, score=2, details="My signal fired!")
```

2. Import it in `momentum_radar/main.py`:

```python
import momentum_radar.signals.my_signal  # noqa: F401
```

That's it.  The signal will now be automatically evaluated on every scan cycle.

---

## Configuration Reference

All values can be overridden in your `.env` file.

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | – | Telegram Bot API token |
| `TELEGRAM_CHAT_ID` | – | Target chat / channel ID |
| `DATA_PROVIDER` | `yfinance` | Data source (`yfinance` in V1) |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `VOLUME_SPIKE_STRONG` | `2.0` | 5-min vol multiplier for strong spike (+2) |
| `VOLUME_SPIKE_MODERATE` | `1.5` | 5-min vol multiplier for moderate spike (+1) |
| `DAILY_VOLUME_RATIO` | `1.5` | Daily vol vs 30d avg ratio trigger |
| `INTRADAY_VOLUME_RATIO` | `3.0` | 1-min vol vs 20-bar avg ratio trigger |
| `RVOL_MODERATE` | `2.0` | RVOL threshold for +1 score |
| `RVOL_STRONG` | `3.0` | RVOL threshold for +2 score |
| `ATR_RATIO_MODERATE` | `1.5` | Day range / ATR for +1 score |
| `ATR_RATIO_STRONG` | `2.0` | Day range / ATR for +2 score |
| `SHORT_INTEREST_MIN` | `0.15` | Minimum short interest % (as decimal) |
| `DAYS_TO_COVER_MIN` | `3` | Minimum days-to-cover |
| `FLOAT_MAX` | `200000000` | Maximum float shares (200M) |
| `OPTIONS_VOLUME_RATIO` | `3.0` | Options vol vs avg for +2 score |
| `SCORE_WATCHLIST` | `4` | Minimum score for WATCHLIST level |
| `SCORE_HIGH_PRIORITY` | `6` | Minimum score for HIGH_PRIORITY level |
| `SCORE_STRONG_MOMENTUM` | `8` | Minimum score for STRONG_MOMENTUM level |
| `SCORE_ALERT_MINIMUM` | `5` | Minimum score to trigger any alert delivery |
| `SCAN_INTERVAL` | `60` | Seconds between scan cycles |
| `ALERT_COOLDOWN` | `600` | Per-ticker alert cooldown in seconds (10 min) |
| `MIN_PRICE` | `5.0` | Minimum stock price for universe inclusion |
| `MIN_AVG_VOLUME` | `1000000` | Minimum 30-day avg daily volume |
| `UNIVERSE_SIZE` | `1000` | Maximum number of tickers to scan |
| `MARKET_OPEN` | `09:35` | Start of scanning window (EST, HH:MM) |
| `MARKET_CLOSE` | `15:45` | End of scanning window (EST, HH:MM) |
| `LUNCH_START` | `12:00` | Start of lunch-lull window (EST) |
| `LUNCH_END` | `13:30` | End of lunch-lull window (EST) |

---

## Plugging in a New Data Provider

1. Subclass `BaseDataFetcher` in `momentum_radar/data/data_fetcher.py`:

```python
class PolygonDataFetcher(BaseDataFetcher):
    def get_intraday_bars(self, ticker, interval="1m", period="1d"):
        ...
    # implement all abstract methods
```

2. Register it in the `get_data_fetcher()` factory:

```python
providers = {
    "yfinance": YFinanceDataFetcher,
    "polygon": PolygonDataFetcher,
}
```

3. Set `DATA_PROVIDER=polygon` in your `.env`.

---

## Future Expansion

- **News sentiment analysis** – integrate a news API and add a
  `news_sentiment` signal module
- **Dark pool / block trade detection** – subscribe to a dark-pool data feed
- **ML scoring** – replace or augment the rule-based scorer with a trained
  gradient-boosting model
- **Web dashboard** – FastAPI + React dashboard to visualise live alerts
- **Backtesting** – replay historical data through the signal pipeline to
  measure historical signal accuracy
- **Multi-exchange support** – extend the universe to cover TSX, LSE, etc.
- **Push notifications** – add Discord, Slack, or SMS delivery via additional
  `alerts/` modules

---

## License

MIT
