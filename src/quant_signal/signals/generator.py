from __future__ import annotations

from typing import Iterable, List, Dict, Any

import pandas as pd

from quant_signal.config import DEFAULT_CONFIG, DataConfig, SMAStrategyConfig
from quant_signal.data.loaders import download_ohlcv
from quant_signal.signals.sma import apply_sma_strategy, last_signal_label


def generate_sma_signals_for_universe(
    universe: Iterable[str],
    data_cfg: DataConfig | None = None,
    sma_cfg: SMAStrategyConfig | None = None,
) -> pd.DataFrame:
    """
    Generate rule-based SMA crossover signals for a list of symbols.

    For each symbol:
      - Downloads OHLCV using DataConfig
      - Applies SMA crossover strategy
      - Extracts latest buy/sell/none signal
      - Returns a consolidated DataFrame

    Returns:
        DataFrame with columns:
            symbol, signal, price, date
    """
    if data_cfg is None:
        data_cfg = DEFAULT_CONFIG.data
    if sma_cfg is None:
        sma_cfg = DEFAULT_CONFIG.sma

    sma_cfg.validate()

    results: List[Dict[str, Any]] = []

    for symbol in universe:
        try:
            df = download_ohlcv(symbol, cfg=data_cfg)
            if df.empty:
                continue

            df = apply_sma_strategy(df, sma_cfg=sma_cfg)
            label = last_signal_label(df)  # BUY / SELL / None

            results.append(
                {
                    "symbol": symbol,
                    "signal": label or "NONE",
                    "price": float(df["Close"].iat[-1]),
                    "date": df.index[-1],
                }
            )
        except Exception as exc:
            results.append(
                {
                    "symbol": symbol,
                    "signal": "ERROR",
                    "price": None,
                    "date": None,
                }
            )

    return pd.DataFrame(results)
