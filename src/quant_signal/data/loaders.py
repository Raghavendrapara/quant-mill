from __future__ import annotations

from datetime import datetime
from typing import Optional, Dict, Tuple

import logging
import time

import pandas as pd
import yfinance as yf

from quant_signal.config import DataConfig, DEFAULT_CONFIG


logger = logging.getLogger(__name__)

# simple in-memory cache so repeated calls in one session don't hit Yahoo again
_DataCacheKey = Tuple[str, str, Optional[str], str, bool]
_DATA_CACHE: Dict[_DataCacheKey, pd.DataFrame] = {}


def _make_cache_key(symbol: str, cfg: DataConfig) -> _DataCacheKey:
    return symbol, cfg.start, cfg.end, cfg.interval, cfg.auto_adjust


def download_ohlcv(
    symbol: str,
    cfg: Optional[DataConfig] = None,
) -> pd.DataFrame:
    """
    Download OHLCV data for a given symbol using yfinance.

    This function:
      - Uses DataConfig (start/end/interval/auto_adjust/retries)
      - Adds basic retry with exponential backoff
      - Caches results in memory for the current process

    Args:
        symbol: e.g. "TCS.NS"
        cfg: DataConfig instance; if None, DEFAULT_CONFIG.data is used.

    Returns:
        DataFrame with columns: [Open, High, Low, Close, Adj Close, Volume]
        indexed by DatetimeIndex (UTC-naive, as returned by yfinance).

    Raises:
        RuntimeError: if data could not be fetched after retries
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.data

    cfg_key = _make_cache_key(symbol, cfg)
    if cfg_key in _DATA_CACHE:
        logger.debug("Using cached data for %s (%s -> %s, %s)", symbol, cfg.start, cfg.end, cfg.interval)
        return _DATA_CACHE[cfg_key].copy()

    end = cfg.end
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    last_exc: Optional[Exception] = None

    for attempt in range(1, cfg.max_retries + 1):
        try:
            logger.info(
                "Downloading %s data for %s (start=%s, end=%s, interval=%s, auto_adjust=%s) [attempt %d/%d]",
                "OHLCV",
                symbol,
                cfg.start,
                end,
                cfg.interval,
                cfg.auto_adjust,
                attempt,
                cfg.max_retries,
            )

            df = yf.download(
                symbol,
                start=cfg.start,
                end=end,
                interval=cfg.interval,
                auto_adjust=cfg.auto_adjust,
                progress=False,
            )

            if df.empty:
                msg = f"No data returned for symbol={symbol}, start={cfg.start}, end={end}, interval={cfg.interval}"
                logger.warning(msg)
                last_exc = RuntimeError(msg)
            else:
                _DATA_CACHE[cfg_key] = df.copy()
                return df

        except Exception as exc:  # noqa: BLE001 - top-level network catch
            logger.exception("Error while downloading data for %s: %s", symbol, exc)
            last_exc = exc

        # if we reached here, we need to retry (if any attempts left)
        if attempt < cfg.max_retries:
            sleep_time = cfg.retry_backoff_sec * attempt
            logger.info("Retrying in %.1f seconds...", sleep_time)
            time.sleep(sleep_time)

    # exhausted retries
    if last_exc is not None:
        raise RuntimeError(f"Failed to download data for {symbol} after {cfg.max_retries} attempts") from last_exc
    raise RuntimeError(f"Failed to download data for {symbol} (unknown error)")
