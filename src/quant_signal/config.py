from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import os


# -----------------------------
# Data / Market Config
# -----------------------------
@dataclass(frozen=True)
class SearchConfig:
    yahoo_search_url: str = os.getenv(
        "YAHOO_SEARCH_URL",
        "https://query1.finance.yahoo.com/v1/finance/search",
    )
    region: str = os.getenv("YAHOO_SEARCH_REGION", "IN")
    lang: str = os.getenv("YAHOO_SEARCH_LANG", "en-US")
    default_limit: int = int(os.getenv("YAHOO_SEARCH_LIMIT", "10"))



@dataclass(frozen=True)
class DataConfig:
    """
    Configuration for historical market data.

    This is the single source of truth for:
      - start/end dates
      - interval
      - retries / backoff
      - auto_adjust (yfinance behavior)
    """
    start: str = "2015-01-01"
    end: Optional[str] = None          # None = "today"
    interval: str = "1d"               # '1d', '1h', '5m', etc.
    auto_adjust: bool = False
    max_retries: int = 3
    retry_backoff_sec: float = 2.0


# -----------------------------
# Strategy Configs
# -----------------------------


@dataclass(frozen=True)
class SMAStrategyConfig:
    """
    Configuration for a simple SMA crossover strategy.
    """
    short_window: int = 50
    long_window: int = 200

    def validate(self) -> None:
        if self.short_window <= 0 or self.long_window <= 0:
            raise ValueError("SMA windows must be positive integers.")
        if self.short_window >= self.long_window:
            raise ValueError(
                f"short_window ({self.short_window}) must be < long_window ({self.long_window})"
            )


@dataclass(frozen=True)
class MLConfig:
    """
    Configuration for ML dataset and label construction.

    - horizon: number of forward bars (days) for label
    - threshold: minimum forward return considered 'positive'
    - feature_cols: which engineered features the model will see
    """
    horizon: int = 5
    threshold: float = 0.02
    feature_cols: List[str] = field(
        default_factory=lambda: [
            "ret_1",
            "ret_5",
            "ret_10",
            "vol_10",
            "vol_20",
            "SMA_20",
            "SMA_50",
        ]
    )

    def validate(self) -> None:
        if self.horizon <= 0:
            raise ValueError("ML horizon must be > 0.")
        if self.threshold <= 0:
            raise ValueError("ML threshold must be > 0.")
        if not self.feature_cols:
            raise ValueError("feature_cols must not be empty.")


# -----------------------------
# Application-wide Defaults
# -----------------------------


@dataclass(frozen=True)
class AppConfig:
    """
    Top-level configuration aggregating defaults.

    In a larger system this could be loaded from YAML/JSON,
    environment variables, etc. For now it's in-code.
    """
    data: DataConfig = DataConfig()
    sma: SMAStrategyConfig = SMAStrategyConfig()
    ml: MLConfig = MLConfig()
    search: SearchConfig = SearchConfig()


# global default config object
DEFAULT_CONFIG = AppConfig()
