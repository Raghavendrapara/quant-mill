from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from quant_signal.config import SMAStrategyConfig, MLConfig


def add_sma(df: pd.DataFrame, window: int, price_col: str = "Close") -> pd.DataFrame:
    """
    Add a Simple Moving Average (SMA_<window>) column to the DataFrame.

    This function mutates the input DataFrame and also returns it
    for chaining convenience.
    """
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame.")

    if window <= 0:
        raise ValueError("SMA window must be > 0.")

    col_name = f"SMA_{window}"
    df[col_name] = df[price_col].rolling(window).mean()
    return df


def add_sma_pair(
    df: pd.DataFrame,
    sma_cfg: SMAStrategyConfig,
    price_col: str = "Close",
) -> pd.DataFrame:
    """
    Add both short and long SMAs as defined in SMAStrategyConfig.
    """
    sma_cfg.validate()

    df = add_sma(df, sma_cfg.short_window, price_col=price_col)
    df = add_sma(df, sma_cfg.long_window, price_col=price_col)
    return df


def add_return_features(
    df: pd.DataFrame,
    horizons: Sequence[int] = (1, 5, 10),
    price_col: str = "Close",
    prefix: str = "ret_",
) -> pd.DataFrame:
    """
    Add rolling log/percentage return features for given horizons.

    For horizon h, we compute:
        ret_h = Close.pct_change(h)
    """
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame.")

    for h in horizons:
        if h <= 0:
            raise ValueError("Return horizon must be > 0.")
        col = f"{prefix}{h}"
        df[col] = df[price_col].pct_change(h)

    return df


def add_volatility_features(
    df: pd.DataFrame,
    windows: Sequence[int] = (10, 20),
    base_return_col: str = "ret_1",
    prefix: str = "vol_",
) -> pd.DataFrame:
    """
    Add rolling volatility (standard deviation) features based on a
    base return column (e.g. ret_1).
    """
    if base_return_col not in df.columns:
        raise ValueError(f"Column '{base_return_col}' not found in DataFrame.")

    for w in windows:
        if w <= 0:
            raise ValueError("Volatility window must be > 0.")
        col = f"{prefix}{w}"
        df[col] = df[base_return_col].rolling(w).std()

    return df


def build_feature_frame(
    df: pd.DataFrame,
    sma_cfg: SMAStrategyConfig,
    ml_cfg: MLConfig,
    price_col: str = "Close",
) -> pd.DataFrame:
    """
    High-level helper: given a raw OHLCV DataFrame plus configs,
    augment it with all features required by MLConfig.feature_cols
    (SMA, returns, volatility) and return the feature frame.

    This function assumes:
      - df has at least 'Close' (and ideally OHLCV)
      - SMAs/windows in ml_cfg.feature_cols are compatible with sma_cfg
    """
    sma_cfg.validate()
    ml_cfg.validate()

    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame.")

    # 1) basic returns
    # detect which return horizons we need based on feature names like "ret_5"
    needed_rets: set[int] = set()
    for name in ml_cfg.feature_cols:
        if name.startswith("ret_"):
            try:
                h = int(name.split("_", 1)[1])
                needed_rets.add(h)
            except (IndexError, ValueError):
                continue

    if needed_rets:
        df = add_return_features(df, horizons=sorted(needed_rets), price_col=price_col)

    # 2) volatility features
    needed_vols: set[int] = set()
    for name in ml_cfg.feature_cols:
        if name.startswith("vol_"):
            try:
                w = int(name.split("_", 1)[1])
                needed_vols.add(w)
            except (IndexError, ValueError):
                continue

    if needed_vols:
        # we assume base_return_col is "ret_1"; if that is not in features,
        # it will still be created above when needed_rets includes 1
        if "ret_1" not in df.columns:
            df = add_return_features(df, horizons=(1,), price_col=price_col)
        df = add_volatility_features(df, windows=sorted(needed_vols), base_return_col="ret_1")

    # 3) SMAs (short + long)
    
    df = add_sma_pair(df, sma_cfg, price_col=price_col)

    # 4) At this point, all required feature columns should exist
    missing = [c for c in ml_cfg.feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Missing expected feature columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # return only feature columns; caller can join with labels/index as needed
    return df[ml_cfg.feature_cols].copy()
