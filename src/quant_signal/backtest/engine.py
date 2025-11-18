import pandas as pd


def compute_returns(df: pd.DataFrame):
    df = df.copy()
    df["Daily"] = df["Close"].pct_change()

    df["Pos"] = (df["Position"] == 1).astype(int).shift(1)
    df["Strat"] = df["Daily"] * df["Pos"]

    df["BH_Cum"] = (1 + df["Daily"]).cumprod()
    df["Strat_Cum"] = (1 + df["Strat"]).cumprod()
    return df
