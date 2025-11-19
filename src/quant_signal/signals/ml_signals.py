from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any, Tuple

import joblib
import pandas as pd

from quant_signal.config import DataConfig, SMAStrategyConfig, MLConfig
from quant_signal.data.loaders import download_ohlcv
from quant_signal.features.technical import build_feature_frame


def load_model_bundle(model_path: str | Path) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a model bundle saved by train_random_forest.

    The bundle has the structure:
        {
            "model": sklearn estimator,
            "metadata": { ... }
        }
    """
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "model" not in bundle or "metadata" not in bundle:
        raise ValueError(f"Model file {model_path} does not contain a valid bundle.")
    return bundle["model"], bundle["metadata"]


def _config_from_metadata(meta: Dict[str, Any]) -> Tuple[DataConfig, SMAStrategyConfig, MLConfig]:
    """
    Reconstruct DataConfig, SMAStrategyConfig, and MLConfig from the metadata dict
    stored inside the model bundle.
    """
    data_cfg = DataConfig(**meta["data_config"])
    sma_cfg = SMAStrategyConfig(**meta["sma_config"])
    ml_cfg = MLConfig(**meta["ml_config"])

    # validate early
    sma_cfg.validate()
    ml_cfg.validate()

    return data_cfg, sma_cfg, ml_cfg


def build_latest_feature_row_for_symbol(
    symbol: str,
    data_cfg: DataConfig,
    sma_cfg: SMAStrategyConfig,
    ml_cfg: MLConfig,
    feature_names: List[str],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Timestamp]]:
    """
    Download data for `symbol`, build feature frame using the provided configs,
    and return the LAST valid feature row aligned with `feature_names`.

    Returns:
        (X_last, ts_last)  where:
            X_last is a 1-row DataFrame (or None if not available)
            ts_last is the corresponding timestamp index
    """
    df = download_ohlcv(symbol, cfg=data_cfg)
    if df.empty:
        return None, None

    # Build all features expected by the model
    feat_frame = build_feature_frame(df.copy(), sma_cfg=sma_cfg, ml_cfg=ml_cfg, price_col="Close")

    # Ensure we have all feature names from training
    missing = [c for c in feature_names if c not in feat_frame.columns]
    if missing:
        # If training used features that no longer exist, we can't safely predict
        return None, None

    feat_frame = feat_frame[feature_names].dropna()
    if feat_frame.empty:
        return None, None

    X_last = feat_frame.iloc[[-1]]  # keep as DataFrame with 1 row
    ts_last = feat_frame.index[-1]

    return X_last, ts_last


def generate_ml_signals_for_universe(
    model_path: str | Path,
    universe: Iterable[str],
    prob_cutoff: float = 0.6,
) -> pd.DataFrame:
    """
    Generate ML-based BUY signals for a list of symbols, using a trained model bundle.

    Logic:
      - Load model + metadata
      - Rebuild DataConfig / SMAStrategyConfig / MLConfig from metadata
      - For each symbol:
          - Build latest feature row
          - Compute P(label=1 | features)
          - If probability >= prob_cutoff -> emit BUY signal

    Returns:
      DataFrame with columns: [symbol, signal, probability, date]
      (or empty DataFrame if no symbols qualify)
    """
    model, metadata = load_model_bundle(model_path)
    data_cfg, sma_cfg, ml_cfg = _config_from_metadata(metadata)

    feature_names: List[str] = metadata.get("feature_names", ml_cfg.feature_cols)
    rows: List[Dict[str, Any]] = []

    for symbol in universe:
        X_last, ts_last = build_latest_feature_row_for_symbol(
            symbol=symbol,
            data_cfg=data_cfg,
            sma_cfg=sma_cfg,
            ml_cfg=ml_cfg,
            feature_names=feature_names,
        )

        if X_last is None or ts_last is None:
            continue

        if not hasattr(model, "predict_proba"):
            raise TypeError("Loaded model does not support predict_proba().")

        prob = float(model.predict_proba(X_last)[0, 1])
        if prob >= prob_cutoff:
            rows.append(
                {
                    "symbol": symbol,
                    "signal": "BUY",
                    "probability": prob,
                    "date": ts_last,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["symbol", "signal", "probability", "date"])

    df_signals = pd.DataFrame(rows).sort_values("probability", ascending=False).reset_index(drop=True)
    return df_signals
