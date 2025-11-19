from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

from quant_signal.config import DataConfig, SMAStrategyConfig, DEFAULT_CONFIG
from quant_signal.data.loaders import download_ohlcv
from quant_signal.signals.sma import apply_sma_strategy, build_long_trades_from_signals, compute_compounded_return


@dataclass
class BacktestResult:
    symbol: str
    initial_capital: float
    final_capital: float
    total_return: float
    n_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    equity_curve: pd.Series


def _compute_equity_curve_from_trades(
    trades: pd.DataFrame,
    initial_capital: float = 100_000.0,
) -> pd.Series:
    """
    Build a simple equity curve that steps at each trade's exit_date, assuming:

      - Full capital allocated to each trade sequentially
      - No overlapping trades
      - No transaction costs or slippage
    """
    capital = initial_capital
    equity = []
    dates = []

    for _, row in trades.iterrows():
        capital *= (1.0 + row["pct_return"])
        equity.append(capital)
        dates.append(row["exit_date"])

    if not dates:
        return pd.Series([], dtype=float, name="equity")

    equity_series = pd.Series(equity, index=pd.to_datetime(dates), name="equity")
    return equity_series


def _max_drawdown(equity: pd.Series) -> float:
    """
    Compute maximum drawdown from an equity curve:

      max_drawdown = max(1 - equity / running_max_equity)
    """
    if equity.empty:
        return 0.0

    running_max = equity.cummax()
    drawdown = 1.0 - equity / running_max
    return float(drawdown.max())


def backtest_sma_long_only(
    symbol: str,
    data_cfg: Optional[DataConfig] = None,
    sma_cfg: Optional[SMAStrategyConfig] = None,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    """
    Backtest a long-only SMA crossover strategy:

      - BUY on Signal == 2
      - SELL on Signal == -2
      - 1x capital sequentially allocated per trade

    Returns BacktestResult which includes summary stats and equity curve.
    """
    if data_cfg is None:
        data_cfg = DEFAULT_CONFIG.data
    if sma_cfg is None:
        sma_cfg = DEFAULT_CONFIG.sma

    sma_cfg.validate()

    df = download_ohlcv(symbol, cfg=data_cfg)
    df = apply_sma_strategy(df, sma_cfg=sma_cfg)

    trades = build_long_trades_from_signals(df)
    if trades.empty:
        equity_curve = pd.Series([], dtype=float, name="equity")
        return BacktestResult(
            symbol=symbol,
            initial_capital=initial_capital,
            final_capital=initial_capital,
            total_return=0.0,
            n_trades=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            max_drawdown=0.0,
            equity_curve=equity_curve,
        )

    equity_curve = _compute_equity_curve_from_trades(trades, initial_capital=initial_capital)
    final_capital = float(equity_curve.iloc[-1])
    total_return = (final_capital / initial_capital) - 1.0

    wins = trades[trades["pct_return"] > 0]
    losses = trades[trades["pct_return"] <= 0]

    n_trades = len(trades)
    win_rate = float(len(wins) / n_trades) if n_trades > 0 else 0.0
    avg_win = float(wins["pct_return"].mean()) if not wins.empty else 0.0
    avg_loss = float(losses["pct_return"].mean()) if not losses.empty else 0.0

    mdd = _max_drawdown(equity_curve)

    return BacktestResult(
        symbol=symbol,
        initial_capital=initial_capital,
        final_capital=final_capital,
        total_return=total_return,
        n_trades=n_trades,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        max_drawdown=mdd,
        equity_curve=equity_curve,
    )
