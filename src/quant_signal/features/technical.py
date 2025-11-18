import pandas as pd


def add_sma(df: pd.DataFrame, period: int, price_col="Close"):
    df[f"SMA_{period}"] = df[price_col].rolling(period).mean()
    return df


def add_sma_pair(df: pd.DataFrame, short=50, long=200):
    df = add_sma(df, short)
    df = add_sma(df, long)
    return df
