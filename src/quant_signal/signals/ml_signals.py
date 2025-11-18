from pathlib import Path

import joblib
import pandas as pd

from quant_signal.models.dataset import build_ml_dataset


def load_model(path: str):
    return joblib.load(path)


def generate_ml_signal_for_symbol(model_path: str, symbol: str, start: str,
                                  horizon: int, threshold: float,
                                  prob_cutoff: float = 0.6) -> pd.DataFrame:
    """
    Generate a single-row signal DataFrame for one symbol using a trained model.
    """
    model = load_model(model_path)

    X, y, idx = build_ml_dataset(symbol, start=start, horizon=horizon, threshold=threshold)
    if X.empty:
        return pd.DataFrame()

    last_features = X.iloc[[-1]]
    last_index = idx[-1]

    prob = model.predict_proba(last_features)[0, 1]

    if prob >= prob_cutoff:
        signal_type = "BUY"
    else:
        # no high-probability opportunity
        return pd.DataFrame()

    row = {
        "symbol": symbol,
        "signal": signal_type,
        "probability": prob,
        "date": last_index,
    }
    return pd.DataFrame([row])

def generate_ml_signals_for_universe(model_path: str, universe, start: str,
                                     horizon: int, threshold: float,
                                     prob_cutoff: float = 0.6) -> pd.DataFrame:
    rows = []
    for symbol in universe:
        df_sig = generate_ml_signal_for_symbol(
            model_path, symbol, start=start,
            horizon=horizon, threshold=threshold,
            prob_cutoff=prob_cutoff,
        )
        if not df_sig.empty:
            rows.append(df_sig)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows).reset_index(drop=True)
