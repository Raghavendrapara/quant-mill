import numpy as np
import pandas as pd
from quant_signal.features.technical import add_sma_pair


def apply_sma_strategy(df: pd.DataFrame, short=50, long=200):
    df = add_sma_pair(df, short, long)
    df["Position"] = np.where(df[f"SMA_{short}"] > df[f"SMA_{long}"], 1, -1)
    df["Signal"] = df["Position"].diff()
    return df


def last_signal(df: pd.DataFrame):
    signal_val = df["Signal"].iloc[-1]

    # Prevent int(np.nan) crash
    if pd.isna(signal_val):
        return None

    # diff() produces floats like 2.0, so normalize to int
    signal_val = int(signal_val)

    if signal_val == 2:
        return "BUY"
    elif signal_val == -2:
        return "SELL"
    else:
        return None

def get_last_crossovers(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Return the last n SMA crossovers (BUY/SELL) with dates and prices.
    """
    events = df[df["Signal"].isin([2, -2])].copy()
    if events.empty:
        return events  # empty DF

    events["signal_type"] = np.where(events["Signal"] == 2, "BUY", "SELL")
    events["price"] = events["Close"]

    # Keep only useful columns
    events = events[["signal_type", "price"]]
    events.index.name = "date"

    # Return last n events
    return events.tail(n)