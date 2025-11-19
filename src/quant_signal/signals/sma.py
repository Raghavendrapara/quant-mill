from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import numpy as np
import pandas as pd

from quant_signal.config import SMAStrategyConfig, DEFAULT_CONFIG
from quant_signal.features.technical import add_sma_pair

# Plotly is optional; import only when needed
def _lazy_plotly_import():
    try:
        import plotly.graph_objects as go
        return go
    except ImportError as e:
        raise RuntimeError(
            "Plotly is required for plotting. Install it with:\n\n"
            "    pip install plotly\n"
        ) from e

# -------------------------------------------------------------------
# Core SMA strategy logic
# -------------------------------------------------------------------


def apply_sma_strategy(
    df: pd.DataFrame,
    sma_cfg: Optional[SMAStrategyConfig] = None,
    price_col: str = "Close",
) -> pd.DataFrame:
    """
    Apply an SMA crossover strategy to a price DataFrame.

    Adds the following columns:
      - SMA_<short>, SMA_<long>
      - Position: 1 (long) or -1 (short/flat)
      - Signal: 2 (BUY), -2 (SELL), 0 (no change)

    This function mutates the DataFrame and returns it for convenience.
    """
    if sma_cfg is None:
        sma_cfg = DEFAULT_CONFIG.sma

    sma_cfg.validate()

    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame.")

    df = add_sma_pair(df, sma_cfg, price_col=price_col)

    short_col = f"SMA_{sma_cfg.short_window}"
    long_col = f"SMA_{sma_cfg.long_window}"

    df["Position"] = np.where(df[short_col] > df[long_col], 1, -1)
    df["Signal"] = df["Position"].diff().fillna(0)

    return df


def last_signal_label(df: pd.DataFrame) -> Optional[str]:
    """
    Interpret the last Signal value as a semantic label.

        2  -> 'BUY'
       -2  -> 'SELL'
        0  -> None (no crossover on the last bar)
    """
    if "Signal" not in df.columns:
        raise ValueError("DataFrame must have a 'Signal' column. Did you call apply_sma_strategy()?")

    signal_val = df["Signal"].iloc[-1]
    if pd.isna(signal_val):
        return None

    signal_val = int(signal_val)

    if signal_val == 2:
        return "BUY"
    if signal_val == -2:
        return "SELL"
    return None


def get_last_crossovers(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Return a small DataFrame containing the last `n` SMA crossovers.

    Requires 'Signal' and 'Close' columns to be present (from apply_sma_strategy).
    """
    if "Signal" not in df.columns or "Close" not in df.columns:
        raise ValueError("DataFrame must have 'Signal' and 'Close' columns.")

    events = df[df["Signal"].isin([2, -2])].copy()
    if events.empty:
        empty = pd.DataFrame(columns=["signal_type", "price"])
        empty.index.name = "date"
        return empty

    events["signal_type"] = np.where(events["Signal"] == 2, "BUY", "SELL")
    events["price"] = events["Close"]
    events.index.name = "date"

    return events[["signal_type", "price"]].tail(n)


# -------------------------------------------------------------------
# Trade & PnL helpers
# -------------------------------------------------------------------


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
    if "Signal" not in df.columns or "Close" not in df.columns:
        raise ValueError("DataFrame must have 'Signal' and 'Close' columns.")

    events = df[df["Signal"].isin([2, -2])].copy()
    if events.empty:
        return pd.DataFrame(
            columns=["entry_date", "exit_date", "entry_price", "exit_price", "pct_return"]
        )

    trades = []
    position = 0  # 0 = flat, 1 = long
    entry_date = None
    entry_price = None

    # Use itertuples to avoid Series -> int/float FutureWarnings
    for date, sig, price in events[["Signal", "Close"]].itertuples(index=True, name=None):
        sig = int(sig)
        price = float(price)

        # BUY
        if sig == 2 and position == 0:
            position = 1
            entry_date = date
            entry_price = price

        # SELL
        elif sig == -2 and position == 1:
            exit_date = date
            exit_price = price
            pct_return = (exit_price / entry_price) - 1.0
            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pct_return": pct_return,
                }
            )
            position = 0
            entry_date = None
            entry_price = None

    return pd.DataFrame(trades)


def compute_compounded_return(trades: pd.DataFrame) -> float:
    """
    Compute total compounded return from a series of trades.

    Each trade has pct_return = (exit / entry) - 1.
    Compounded return is:
        Π (1 + pct_return_i) - 1
    """
    if trades.empty:
        return 0.0

    growth = (1.0 + trades["pct_return"]).prod()
    return growth - 1.0


# -------------------------------------------------------------------
# Visualization (interactive Plotly)
# -------------------------------------------------------------------


def plot_sma_crossovers_interactive(
    df: pd.DataFrame,
    symbol: str,
    sma_cfg: Optional[SMAStrategyConfig] = None,
    price_col: str = "Close",
    show: bool = True,
    save_path: Optional[str] = None,
):

    """
    Interactive Plotly chart:

      - Candlestick price
      - SMA short & long
      - BUY / SELL markers

    This function returns the Plotly Figure object. By default it also
    calls fig.show(). If save_path is provided, an HTML file is written.

    In a non-interactive / server context, set show=False and use save_path.
    """
    go = _lazy_plotly_import()

    if sma_cfg is None:
        sma_cfg = DEFAULT_CONFIG.sma

    sma_cfg.validate()

    short_col = f"SMA_{sma_cfg.short_window}"
    long_col = f"SMA_{sma_cfg.long_window}"

    required_cols = {"Open", "High", "Low", price_col, short_col, long_col, "Signal"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame missing required columns for plotting: {missing}. "
            f"Available: {list(df.columns)}"
        )

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df[price_col],
                name="Price",
            )
        ]
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[short_col],
            name=short_col,
            mode="lines",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[long_col],
            name=long_col,
            mode="lines",
        )
    )

    buys = df[df["Signal"] == 2]
    sells = df[df["Signal"] == -2]

    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=buys.index,
                y=buys[price_col],
                mode="markers",
                name="BUY",
                marker=dict(symbol="triangle-up", size=10),
            )
        )

    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=sells.index,
                y=sells[price_col],
                mode="markers",
                name="SELL",
                marker=dict(symbol="triangle-down", size=10),
            )
        )

    fig.update_layout(
        title=f"SMA {sma_cfg.short_window}/{sma_cfg.long_window} Crossovers – {symbol}",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
    )

    if save_path is not None:
        fig.write_html(save_path)

    if show:
        fig.show()

    return fig
