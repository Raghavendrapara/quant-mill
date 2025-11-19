from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import List

from quant_signal.data.search import YahooSearchConfig, search_yahoo_symbols
from quant_signal.signals.generator import generate_sma_signals_for_universe

import click
import pandas as pd

from quant_signal.config import (
    DEFAULT_CONFIG,
    SMAStrategyConfig,
    MLConfig,
    DataConfig,
)
from quant_signal.data.loaders import download_ohlcv
from quant_signal.models.train import train_random_forest
from quant_signal.signals.sma import (
    apply_sma_strategy,
    last_signal_label,
)
from quant_signal.signals.ml_signals import generate_ml_signals_for_universe

# interactive shell (existing file)
try:
    from quant_signal.interactive import run_shell
except ImportError:
    run_shell = None


# Basic logging setup for CLI usage
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)


@click.group()
def cli():
    """Quant-Mill command line interface."""
    pass


# --------------------------------------------------------------------
# 1) SMA rule-based signals (non-ML)
# --------------------------------------------------------------------


@cli.command("signals")
@click.option(
    "--symbols",
    "-s",
    multiple=True,
    required=True,
    help="List of tickers, e.g. -s TCS.NS -s RELIANCE.NS",
)
@click.option(
    "--short-window",
    type=int,
    default=None,
    help="Override short SMA window (default from config).",
)
@click.option(
    "--long-window",
    type=int,
    default=None,
    help="Override long SMA window (default from config).",
)
def signals_cmd(symbols: List[str], short_window: int | None, long_window: int | None):
    """
    Generate simple SMA crossover signals for the latest bar.
    """
    sma_cfg = DEFAULT_CONFIG.sma
    if short_window is not None or long_window is not None:
        sma_cfg = SMAStrategyConfig(
            short_window=short_window or sma_cfg.short_window,
            long_window=long_window or sma_cfg.long_window,
        )
        sma_cfg.validate()

    data_cfg = DEFAULT_CONFIG.data

    rows = []
    for symbol in symbols:
        df = download_ohlcv(symbol, cfg=data_cfg)
        df = apply_sma_strategy(df, sma_cfg=sma_cfg)
        label = last_signal_label(df)
        price = price = float(df["Close"].iat[-1])
        date = df.index[-1]

        rows.append(
            {
                "symbol": symbol,
                "signal": label or "NONE",
                "price": price,
                "date": date,
            }
        )

    df_out = pd.DataFrame(rows)
    click.echo(df_out)


# --------------------------------------------------------------------
# 2) Train ML model
# --------------------------------------------------------------------


@cli.command("train-ml")
@click.option("--symbol", "-s", required=True, help="Symbol to train on, e.g. TCS.NS")
@click.option(
    "--start",
    type=str,
    default=None,
    help="Override start date for data (YYYY-MM-DD). Default from config.",
)
@click.option(
    "--horizon",
    type=int,
    default=None,
    help="Forward-return horizon in bars (days). Default from config.",
)
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="Forward-return threshold (e.g. 0.02 for 2%%). Default from config.",
)
@click.option(
    "--model-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Directory to save model artifacts. Default = src/quant_signal/models/artifacts",
)
def train_ml_cmd(
    symbol: str,
    start: str | None,
    horizon: int | None,
    threshold: float | None,
    model_dir: Path | None,
):
    """
    Train a RandomForest ML model for one symbol with config-aware settings.
    """
    data_cfg = DEFAULT_CONFIG.data
    if start is not None:
        data_cfg = DataConfig(
            start=start,
            end=data_cfg.end,
            interval=data_cfg.interval,
            auto_adjust=data_cfg.auto_adjust,
            max_retries=data_cfg.max_retries,
            retry_backoff_sec=data_cfg.retry_backoff_sec,
        )

    ml_cfg = DEFAULT_CONFIG.ml
    if horizon is not None or threshold is not None:
        ml_cfg = MLConfig(
            horizon=horizon or ml_cfg.horizon,
            threshold=threshold or ml_cfg.threshold,
            feature_cols=ml_cfg.feature_cols,
        )

    _, metadata = train_random_forest(
        symbol=symbol,
        data_cfg=data_cfg,
        sma_cfg=DEFAULT_CONFIG.sma,
        ml_cfg=ml_cfg,
        model_dir=model_dir,
    )

    click.echo("Training complete. Model metadata:")
    click.echo(metadata)


# --------------------------------------------------------------------
# 3) ML-based signals using trained model bundle
# --------------------------------------------------------------------


@cli.command("signals-ml")
@click.option(
    "--symbols",
    "-s",
    multiple=True,
    required=True,
    help="Universe of tickers (can be different from training symbol).",
)
@click.option(
    "--model-path",
    "-m",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to trained model .joblib bundle.",
)
@click.option(
    "--prob-cutoff",
    type=float,
    default=0.6,
    show_default=True,
    help="Minimum P(label=1) to emit a BUY signal.",
)
def signals_ml_cmd(
    symbols: List[str],
    model_path: Path,
    prob_cutoff: float,
):
    """
    Generate ML-based BUY signals for the given universe using a trained model bundle.

    The model's own MLConfig (horizon, threshold, features) is loaded from metadata,
    so you don't have to re-specify it here.
    """
    df = generate_ml_signals_for_universe(
        model_path=model_path,
        universe=list(symbols),
        prob_cutoff=prob_cutoff,
    )
    if df.empty:
        click.echo("No ML signals today.")
    else:
        click.echo(df)
        out_path = Path("today_signals_ml.csv")
        df.to_csv(out_path, index=False)
        click.echo(f"Saved to {out_path}")

@cli.command("scan-sma")
@click.option(
    "--symbols",
    "-s",
    multiple=True,
    required=True,
    help="List of tickers, e.g. -s TCS.NS -s RELIANCE.NS",
)
@click.option(
    "--short-window",
    type=int,
    default=None,
    help="Override short SMA window (default from config).",
)
@click.option(
    "--long-window",
    type=int,
    default=None,
    help="Override long SMA window (default from config).",
)
def scan_sma_cmd(symbols: List[str], short_window: int | None, long_window: int | None):
    """
    Scan a set of tickers for today's SMA crossover signals.

    Example:
        quant-signal scan-sma -s TCS.NS -s RELIANCE.NS
    """
    if not symbols:
        click.echo("No symbols provided. Use -s SYMBOL")
        return

    sma_cfg = DEFAULT_CONFIG.sma
    if short_window is not None or long_window is not None:
        sma_cfg = SMAStrategyConfig(
            short_window=short_window or sma_cfg.short_window,
            long_window=long_window or sma_cfg.long_window,
        )
        sma_cfg.validate()

    data_cfg = DEFAULT_CONFIG.data

    click.echo(
        f"Scanning {len(symbols)} symbols with SMA "
        f"{sma_cfg.short_window}/{sma_cfg.long_window}..."
    )

    df = generate_sma_signals_for_universe(
        universe=list(symbols),
        data_cfg=data_cfg,
        sma_cfg=sma_cfg,
    )

    if df.empty:
        click.echo("No signals.")
        return

    click.echo(df)
    df.to_csv("sma_scan.csv", index=False)
    click.echo("Saved to sma_scan.csv")


@cli.command("search-symbol")
@click.option(
    "--query",
    "-q",
    type=str,
    required=True,
    help="Partial company name or ticker (e.g. 'tcs', 'reliance', 'hdfc').",
)
@click.option(
    "--exchange",
    "-e",
    type=str,
    default=None,
    help="Optional exchange filter (e.g. 'NSE', 'NSI', 'BSE').",
)
@click.option(
    "--region",
    "-r",
    type=str,
    default="IN",
    show_default=True,
    help="Yahoo region code (e.g. IN, US).",
)
@click.option(
    "--limit",
    "-n",
    type=int,
    default=10,
    show_default=True,
    help="Maximum number of symbols to return.",
)
def search_symbol_cmd(query: str, exchange: str | None, region: str, limit: int):
    """
    Search Yahoo Finance for matching symbols, optionally filtered to NSE/BSE.

    Examples:
      quant-signal search-symbol -q tata -e NSE
      quant-signal search-symbol -q reliance -e BSE
    """
    cfg = YahooSearchConfig(region=region.upper(), quotes_count=limit)
    df = search_yahoo_symbols(query, cfg=cfg, exchange_filter=exchange)

    if df.empty:
        click.echo(f"No matches found for '{query}' (exchange={exchange or 'ANY'}, region={region}).")
        return

    # Show the most relevant columns
    out = df[["symbol", "exchange", "quoteType", "shortname"]]
    click.echo(out)

    click.echo(
        "\nUse the 'symbol' column with other commands, e.g.:\n"
        "  quant-signal scan-sma -s TCS.NS\n"
        "  quant-signal sma-history -s TCS.NS\n"
        "  quant-signal signals-ml -s TCS.NS -m model.joblib"
    )

@click.command("config-search")
def config_search():
    """Print effective Yahoo search config."""
    from quant_signal.config import DEFAULT_CONFIG
    click.echo(DEFAULT_CONFIG.search)

@cli.command("sma-history")
@click.option(
    "--symbol",
    "-s",
    required=True,
    help="Ticker symbol, e.g. TCS.NS",
)
@click.option(
    "--n",
    type=int,
    default=10,
    show_default=True,
    help="Number of most recent crossovers to show.",
)
@click.option(
    "--with-backtest/--no-backtest",
    default=True,
    show_default=True,
    help="Whether to run full SMA backtest summary.",
)
def sma_history_cmd(symbol: str, n: int, with_backtest: bool):
    """
    Show recent SMA crossovers + trades + PnL + backtest summary.
    """
    from quant_signal.config import DEFAULT_CONFIG
    from quant_signal.data.loaders import download_ohlcv
    from quant_signal.signals.sma import (
        apply_sma_strategy,
        get_last_crossovers,
        build_long_trades_from_signals,
        compute_compounded_return,
    )
    from quant_signal.backtest.sma_backtest import backtest_sma_long_only

    data_cfg = DEFAULT_CONFIG.data
    sma_cfg = DEFAULT_CONFIG.sma

    df = download_ohlcv(symbol, cfg=data_cfg)
    df = apply_sma_strategy(df, sma_cfg=sma_cfg)

    print(f"\n--- Last {n} SMA crossovers for {symbol} (SMA {sma_cfg.short_window}/{sma_cfg.long_window}) ---")
    hist = get_last_crossovers(df, n)
    if hist.empty:
        print("No crossovers found.")
    else:
        print(hist)

    trades = build_long_trades_from_signals(df)
    if trades.empty:
        print("\nNo completed trades.")
    else:
        last_trades = trades.tail(n)
        sum_ret = last_trades["pct_return"].sum()
        comp_ret = compute_compounded_return(last_trades)

        print(f"\nLast {min(n, len(last_trades))} trades:")
        print(last_trades)

        print(f"\nPnL (sum of returns):       {sum_ret * 100:.2f}%")
        print(f"PnL (compounded actual):     {comp_ret * 100:.2f}%")

    if with_backtest:
        print("\n--- Full SMA Backtest Summary ---")
        bt = backtest_sma_long_only(symbol, data_cfg=data_cfg, sma_cfg=sma_cfg)
        print(f"Initial capital:  {bt.initial_capital:,.2f}")
        print(f"Final capital:    {bt.final_capital:,.2f}")
        print(f"Total return:     {bt.total_return * 100:.2f}%")
        print(f"Trades:           {bt.n_trades}")
        print(f"Win rate:         {bt.win_rate * 100:.2f}%")
        print(f"Avg win:          {bt.avg_win * 100:.2f}%")
        print(f"Avg loss:         {bt.avg_loss * 100:.2f}%")


# --------------------------------------------------------------------
# 4) Interactive shell
# --------------------------------------------------------------------


@cli.command("interactive")
def interactive_cmd():
    """
    Start an interactive menu-driven session (if available).

    This reuses the existing interactive.py module.
    """
    if run_shell is None:
        click.echo("Interactive shell is not available (interactive.py not found).")
        return
    run_shell()


if __name__ == "__main__":
    cli()
