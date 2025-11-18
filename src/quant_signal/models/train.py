import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report

from quant_signal.models.dataset import build_ml_dataset


MODEL_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_DIR.mkdir(exist_ok=True)


def train_random_forest(
    symbol: str,
    start: str = "2015-01-01",
    horizon: int = 5,
    threshold: float = 0.02,
    n_splits: int = 4,
):
    X, y, idx = build_ml_dataset(symbol, start=start, horizon=horizon, threshold=threshold)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    all_preds = np.zeros_like(y, dtype=int)

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        all_preds[test_idx] = preds

    print("Time-series CV classification report:")
    print(classification_report(y, all_preds, digits=3))

    # final fit on full data
    final_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X, y)

    model_path = MODEL_DIR / f"rf_{symbol.replace('.', '_')}_h{horizon}_t{threshold:.3f}.joblib"
    joblib.dump(final_model, model_path)

    print(f"\nSaved model to {model_path}")
    return model_path
