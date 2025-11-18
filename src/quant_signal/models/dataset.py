import pandas as pd

from quant_signal.data.loaders import download_ohlcv
from quant_signal.features.technical import add_sma_pair


def build_ml_dataset(
    symbol: str,
    start: str = "2015-01-01",
    horizon: int = 5,
    threshold: float = 0.02,
) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """
    Build a supervised ML dataset for a single symbol.

    Features:
      - ret_1, ret_5, ret_10   : past returns
      - vol_10, vol_20         : rolling volatility of daily returns
      - SMA_20, SMA_50         : simple moving averages

    Label:
      - 1 if forward horizon return > threshold, else 0
    """
    # 1) Download price data
    df = download_ohlcv(symbol, start=start)
    if df.empty:
        raise RuntimeError(f"No data downloaded for {symbol} from {start}")
    if "Close" not in df.columns:
        raise RuntimeError(
            f"'Close' column not found in data for {symbol}. "
            f"Columns present: {list(df.columns)}"
        )

    # 2) Add SMAs (creates SMA_20 and SMA_50)
    df = add_sma_pair(df, short=20, long=50)

    # 3) Price-based features
    df["ret_1"] = df["Close"].pct_change(1)
    df["ret_5"] = df["Close"].pct_change(5)
    df["ret_10"] = df["Close"].pct_change(10)

    df["vol_10"] = df["ret_1"].rolling(10).std()
    df["vol_20"] = df["ret_1"].rolling(20).std()

    # 4) Forward return + label
    fwd_price = df["Close"].shift(-horizon)
    df["fwd_ret"] = (fwd_price / df["Close"]) - 1.0
    df["label"] = (df["fwd_ret"] > threshold).astype(int)

    feature_cols = [
        "ret_1",
        "ret_5",
        "ret_10",
        "vol_10",
        "vol_20",
        "SMA_20",
        "SMA_50",
    ]
    expected_cols = feature_cols + ["label"]

    # Debug print so we can see what exists
    print("DEBUG – columns in df before selecting features:")
    print(list(df.columns))

    # 5) Work only with features + label, then drop NaNs
    #    (⚠ NO 'subset=' here at all)
    features_df = df[expected_cols].copy()
    features_df = features_df.dropna()

    if features_df.empty:
        raise RuntimeError(
            f"After dropping NaNs, no rows left for {symbol}. "
            f"Check horizon={horizon}, start={start}, and data quality."
        )

    X = features_df[feature_cols]
    y = features_df["label"]
    idx = features_df.index

    return X, y, idx
