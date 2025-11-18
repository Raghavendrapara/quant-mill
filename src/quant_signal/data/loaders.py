import yfinance as yf
import pandas as pd
from datetime import datetime


def download_ohlcv(symbol: str, start="2015-01-01", end=None, interval="1d") -> pd.DataFrame:
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data for {symbol}")
    return df
