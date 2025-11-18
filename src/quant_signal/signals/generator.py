from quant_signal.data.loaders import download_ohlcv
from quant_signal.signals.sma import apply_sma_strategy, last_signal
import pandas as pd


def generate_signals(universe):
    rows = []
    for symbol in universe:
        df = download_ohlcv(symbol, start="2018-01-01")
        df = apply_sma_strategy(df)

        sig = last_signal(df)
        if sig:
            rows.append({
                "symbol": symbol,
                "signal": sig,
                "price": df["Close"].iloc[-1],
                "date": df.index[-1]
            })

    return pd.DataFrame(rows)
