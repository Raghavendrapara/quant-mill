import numpy as np
import pandas as pd
from quant_signal.features.technical import add_sma_pair
import matplotlib.pyplot as plt


def plot_sma_crossovers(df, symbol, short=50, long=200):
    """
    Display a price chart with SMA crossovers, buy/sell markers.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df["Close"], label="Close Price", linewidth=1)
    plt.plot(df[f"SMA_{short}"], label=f"SMA {short}", linewidth=1.2)
    plt.plot(df[f"SMA_{long}"], label=f"SMA {long}", linewidth=1.2)

    # Mark BUY signals
    buys = df[df["Signal"] == 2]
    sells = df[df["Signal"] == -2]

    plt.scatter(buys.index, buys["Close"], marker="^", color="green", s=80, label="BUY")
    plt.scatter(sells.index, sells["Close"], marker="v", color="red", s=80, label="SELL")

    plt.title(f"SMA Crossovers for {symbol}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


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

def build_long_trades_from_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build completed long trades from SMA crossover signals.

    Assumptions:
      - BUY on Signal == 2  (Golden Cross)
      - SELL on Signal == -2 (Death Cross)
      - Long-only: we either hold 1 unit or are flat.
      - Any last open BUY without a later SELL is ignored.

    Returns a DataFrame with:
      entry_date, exit_date, entry_price, exit_price, pct_return
    """
    events = df[df["Signal"].isin([2, -2])].copy()
    if events.empty:
        return pd.DataFrame(columns=[
            "entry_date", "exit_date", "entry_price", "exit_price", "pct_return"
        ])

    trades = []
    position = 0  # 0 = flat, 1 = long
    entry_date = None
    entry_price = None

    for date, sig, price in events[["Signal", "Close"]].itertuples(index=True, name=None):
        sig = int(sig)
        price = float(price)

        # BUY signal
        if sig == 2 and position == 0:
            position = 1
            entry_date = date
            entry_price = price

        # SELL signal
        elif sig == -2 and position == 1:
            exit_date = date
            exit_price = price
            pct_return = (exit_price / entry_price) - 1.0
            trades.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pct_return": pct_return,
            })
            position = 0
            entry_date = None
            entry_price = None

    return pd.DataFrame(trades)

def compute_compounded_return(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    growth = (1 + trades["pct_return"]).prod()
    return growth - 1
