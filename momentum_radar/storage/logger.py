"""
logger.py – CSV alert logging.

Each day a new CSV file is created under the ``logs/`` directory with the
current date in its filename.  Every call to :func:`log_alert_csv` appends
one row to that day's file.
"""

import csv
import logging
import os
from datetime import datetime, date
from typing import List, Optional

logger = logging.getLogger(__name__)

_LOG_DIR = "logs"


def _ensure_log_dir() -> None:
    """Create the log directory if it does not already exist."""
    os.makedirs(_LOG_DIR, exist_ok=True)


def _daily_csv_path(log_date: Optional[date] = None) -> str:
    """Return the CSV file path for the given date (default: today).

    Args:
        log_date: Date to use; defaults to ``date.today()``.

    Returns:
        Absolute path string for the CSV log file.
    """
    d = log_date or date.today()
    return os.path.join(_LOG_DIR, f"alerts_{d.strftime('%Y%m%d')}.csv")


_CSV_FIELDNAMES = [
    "timestamp",
    "ticker",
    "price",
    "pct_change",
    "score",
    "alert_level",
    "modules_triggered",
    "rvol",
    "atr_ratio",
    "short_interest",
    "float_shares",
    "market_condition",
]


def log_alert_csv(
    ticker: str,
    price: Optional[float],
    score: int,
    alert_level: str,
    modules_triggered: List[str],
    rvol: Optional[float] = None,
    atr_ratio: Optional[float] = None,
    short_interest: Optional[float] = None,
    float_shares: Optional[float] = None,
    market_condition: Optional[str] = None,
    pct_change: Optional[float] = None,
    timestamp: Optional[datetime] = None,
) -> None:
    """Append one alert row to today's CSV log file.

    The file is created with a header row on first write.

    Args:
        ticker: Stock symbol.
        price: Current price.
        score: Total signal score.
        alert_level: Alert level string.
        modules_triggered: List of signal names that fired.
        rvol: Relative volume.
        atr_ratio: Day range / ATR ratio.
        short_interest: Short interest as decimal.
        float_shares: Float share count.
        market_condition: Market context string.
        pct_change: Percentage price change today.
        timestamp: Override timestamp (defaults to ``datetime.utcnow()``).
    """
    _ensure_log_dir()
    path = _daily_csv_path()
    ts = timestamp or datetime.utcnow()
    write_header = not os.path.exists(path)

    row = {
        "timestamp": ts.isoformat(),
        "ticker": ticker,
        "price": price,
        "pct_change": pct_change,
        "score": score,
        "alert_level": alert_level,
        "modules_triggered": "|".join(modules_triggered),
        "rvol": rvol,
        "atr_ratio": atr_ratio,
        "short_interest": short_interest,
        "float_shares": float_shares,
        "market_condition": market_condition,
    }

    try:
        with open(path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        logger.debug("Alert logged to CSV: %s", path)
    except Exception as exc:
        logger.error("Failed to write CSV log: %s", exc)
