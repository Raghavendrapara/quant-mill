from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit

from quant_signal.config import (
    DataConfig,
    SMAStrategyConfig,
    MLConfig,
    DEFAULT_CONFIG,
)
from quant_signal.models.dataset import build_ml_dataset_for_symbol

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "artifacts"
DEFAULT_MODEL_DIR.mkdir(exist_ok=True)


def _serialize_config(cfg) -> Dict[str, Any]:
    """
    Safely turn a dataclass config into a JSON-serializable dict.
    """
    return asdict(cfg)


def train_random_forest(
    symbol: str,
    data_cfg: Optional[DataConfig] = None,
    sma_cfg: Optional[SMAStrategyConfig] = None,
    ml_cfg: Optional[MLConfig] = None,
    n_splits: int = 4,
    model_dir: Optional[Path] = None,
    random_state: int = 42,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Train a RandomForest classifier to predict forward return labels
    for a given symbol, using time-series cross-validation.

    Steps:
      1. Build ML dataset (features + labels) for the symbol.
      2. Use TimeSeriesSplit to evaluate CV performance.
      3. Print classification report on CV out-of-fold predictions.
      4. Fit final model on the full dataset.
      5. Save model + metadata as a joblib artifact.

    Returns:
      (model_path, metadata_dict)
    """
    if data_cfg is None:
        data_cfg = DEFAULT_CONFIG.data
    if sma_cfg is None:
        sma_cfg = DEFAULT_CONFIG.sma
    if ml_cfg is None:
        ml_cfg = DEFAULT_CONFIG.ml

    sma_cfg.validate()
    ml_cfg.validate()

    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Building ML dataset for symbol=%s, data_cfg=%s, sma_cfg=%s, ml_cfg=%s",
        symbol,
        data_cfg,
        sma_cfg,
        ml_cfg,
    )

    X, y, idx = build_ml_dataset_for_symbol(
        symbol=symbol,
        data_cfg=data_cfg,
        sma_cfg=sma_cfg,
        ml_cfg=ml_cfg,
    )

    logger.info("Dataset built: %d samples, %d features", X.shape[0], X.shape[1])

    # -----------------------------
    # Time-series cross-validation
    # -----------------------------

    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_preds = np.zeros_like(y, dtype=int)

    fold = 0
    for train_idx, test_idx in tscv.split(X):
        fold += 1
        logger.info("TS CV fold %d/%d: train=%d, test=%d", fold, n_splits, len(train_idx), len(test_idx))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        oof_preds[test_idx] = preds

    logger.info("Time-series CV complete. Generating classification report.")
    report_str = classification_report(y, oof_preds, digits=3)
    print("Time-series CV classification report:")
    print(report_str)

    # -----------------------------
    # Final model on full data
    # -----------------------------
    final_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    final_model.fit(X, y)

    # -----------------------------
    # Build metadata
    # -----------------------------
    metadata: Dict[str, Any] = {
        "symbol": symbol,
        "data_config": _serialize_config(data_cfg),
        "sma_config": _serialize_config(sma_cfg),
        "ml_config": _serialize_config(ml_cfg),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_names": list(X.columns),
        "index_start": idx.min().isoformat() if len(idx) > 0 else None,
        "index_end": idx.max().isoformat() if len(idx) > 0 else None,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "model_type": "RandomForestClassifier",
        "n_splits": n_splits,
        "random_state": random_state,
        "cv_classification_report": report_str,
    }

    # -----------------------------
    # Save model + metadata bundle
    # -----------------------------
    safe_symbol = symbol.replace(".", "_").replace("/", "_")
    horizon = ml_cfg.horizon
    threshold = ml_cfg.threshold

    filename = f"rf_{safe_symbol}_h{horizon}_t{threshold:.3f}.joblib"
    model_path = model_dir / filename

    bundle = {
        "model": final_model,
        "metadata": metadata,
    }

    joblib.dump(bundle, model_path)
    logger.info("Saved model bundle to %s", model_path)

    # Also save a small JSON sidecar for quick inspection (optional)
    json_path = model_dir / f"{filename}.meta.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata JSON to %s", json_path)

    return model_path, metadata
