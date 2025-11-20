from __future__ import annotations

from dataclasses import replace
from typing import Optional, Tuple

import pandas as pd

from quant_signal.config import (
    DataConfig,
    SMAStrategyConfig,
    MLConfig,
    DEFAULT_CONFIG,
)
from quant_signal.data.loaders import download_ohlcv
from quant_signal.features.technical import build_feature_frame


def build_features_and_labels_for_df(
    df: pd.DataFrame,
    sma_cfg: Optional[SMAStrategyConfig] = None,
    ml_cfg: Optional[MLConfig] = None,
    price_col: str = "Close",
) -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """
    Core dataset builder: given a price DataFrame with at least 'Close',
    construct:

      - X: feature matrix (columns = ml_cfg.feature_cols)
      - y: binary label Series (0/1)
      - idx: Date index aligned with X and y

    No symbol or IO knowledge here; purely transforms the data.

    Label definition (by MLConfig):
      label = 1 if forward_return(horizon) > threshold else 0
    where:
      forward_return(h) = Close[t+h] / Close[t] - 1
    """
    if sma_cfg is None:
        sma_cfg = DEFAULT_CONFIG.sma
    if ml_cfg is None:
        ml_cfg = DEFAULT_CONFIG.ml

    sma_cfg.validate()
    ml_cfg.validate()

    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame.")

    # 1) Build feature frame X based on configs
    df_features = build_feature_frame(df.copy(), sma_cfg=sma_cfg, ml_cfg=ml_cfg, price_col=price_col)

    # 2) Build forward return and labels on the original df
    fwd_price = df[price_col].shift(-ml_cfg.horizon)
    df["fwd_ret"] = (fwd_price / df[price_col]) - 1.0
    df["label"] = (df["fwd_ret"] > ml_cfg.threshold).astype(int)

    # 3) Align features and labels, drop rows with NaNs in any required columns
    expected_cols = list(df_features.columns) + ["label"]
    joined = df_features.join(df["label"], how="inner")

    joined = joined.dropna(subset=expected_cols)
    if joined.empty:
        raise RuntimeError(
            "After dropping NaNs, no rows left. "
            f"Check data length, horizon={ml_cfg.horizon}, threshold={ml_cfg.threshold}."
        )

    X = joined[df_features.columns].copy()
    y = joined["label"].copy()
    idx = joined.index

    return X, y, idx


def build_ml_dataset_for_symbol(
    symbol: str,
    data_cfg: Optional[DataConfig] = None,
    sma_cfg: Optional[SMAStrategyConfig] = None,
    ml_cfg: Optional[MLConfig] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """
    Convenience wrapper: download OHLCV for a symbol using DataConfig,
    then build features + labels using SMAStrategyConfig and MLConfig.

    This keeps IO separated from the pure transformation logic, but
    makes it easy to use from higher-level code (CLI, training scripts).
    """
    if data_cfg is None:
        data_cfg = DEFAULT_CONFIG.data
    if sma_cfg is None:
        sma_cfg = DEFAULT_CONFIG.sma
    if ml_cfg is None:
        ml_cfg = DEFAULT_CONFIG.ml

    sma_cfg.validate()
    ml_cfg.validate()

    df = download_ohlcv(symbol, cfg=data_cfg)
    if df.empty:
        raise RuntimeError(f"No data downloaded for symbol={symbol} with {data_cfg}.")

    X, y, idx = build_features_and_labels_for_df(
        df=df,
        sma_cfg=sma_cfg,
        ml_cfg=ml_cfg,
        price_col="Close",
    )

    return X, y, idx
