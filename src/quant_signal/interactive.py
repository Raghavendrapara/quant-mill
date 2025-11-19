from __future__ import annotations

from typing import List

import pandas as pd

from quant_signal.config import DEFAULT_CONFIG, SMAStrategyConfig, DataConfig
from quant_signal.data.loaders import download_ohlcv
from quant_signal.data.search import YahooSearchConfig, search_yahoo_symbols
from quant_signal.signals.sma import (
    apply_sma_strategy,
    get_last_crossovers,
    build_long_trades_from_signals,
    compute_compounded_return,
    plot_sma_crossovers_interactive,
)
from quant_signal.signals.ml_signals import generate_ml_signals_for_universe


# -------------------------------------------------------------------
# Small utility helpers
# -------------------------------------------------------------------


def _clear_screen():
    # Simple console "clear" without OS calls
    print("\n" * 80)


def _press_enter():
    input("\nPress ENTER to continue...")


# -------------------------------------------------------------------
# SMA menu actions
# -------------------------------------------------------------------


def _run_sma_today_signals(symbols: List[str], data_cfg: DataConfig, sma_cfg: SMAStrategyConfig):
    print("\n--- Today's SMA Signals ---")
    for symbol in symbols:
        df = download_ohlcv(symbol, cfg=data_cfg)
        df = apply_sma_strategy(df, sma_cfg=sma_cfg)
        sig_val = df["Signal"].iloc[-1]
        price = float(df["Close"].iat[-1])
        date = df.index[-1]

        if sig_val == 2:
            label = "BUY"
        elif sig_val == -2:
            label = "SELL"
        else:
            label = "NONE"

        print(f"{symbol}: {label} (price={price:.2f}, date={date.date()})")


def _run_sma_history_and_pnl(symbols: List[str], data_cfg: DataConfig, sma_cfg: SMAStrategyConfig):
    try:
        n = int(input("How many past crossovers to show (e.g., 10)? ").strip() or "5")
    except ValueError:
        print("Invalid number. Using 5.")
        n = 5

    print("\n--- Historical Crossovers & PnL ---")

    for symbol in symbols:
        print(f"\nSymbol: {symbol}")
        df = download_ohlcv(symbol, cfg=data_cfg)
        df = apply_sma_strategy(df, sma_cfg=sma_cfg)

        hist = get_last_crossovers(df, n)
        if hist.empty:
            print("  No crossovers found.")
        else:
            print("  Last crossovers:")
            print(hist)

        ans = input("\n  Compute PnL for this SMA strategy? [y/N]: ").strip().lower()
        if ans != "y":
            continue

        trades = build_long_trades_from_signals(df)
        if trades.empty:
            print("  No completed trades (no BUY->SELL pairs).")
            continue

        last_trades = trades.tail(n)

        sum_return = last_trades["pct_return"].sum()
        comp_return = compute_compounded_return(last_trades)

        print("\n  Last trades:")
        print(last_trades)

        print(f"\n  Total PnL (sum of returns):       {sum_return * 100:.2f}%")
        print(f"  Total PnL (compounded, real PnL): {comp_return * 100:.2f}%")

    _press_enter()


def _run_sma_plot(symbols: List[str], data_cfg: DataConfig, sma_cfg: SMAStrategyConfig):
    print("\n--- SMA Crossover Chart (Interactive) ---")
    for symbol in symbols:
        print(f"Plotting for {symbol}...")
        df = download_ohlcv(symbol, cfg=data_cfg)
        df = apply_sma_strategy(df, sma_cfg=sma_cfg)
        # This opens an interactive Plotly chart in the browser
        plot_sma_crossovers_interactive(df, symbol, sma_cfg=sma_cfg)

    _press_enter()


def _sma_menu(symbols: List[str]):
    """
    Sub-menu for SMA-related operations (today, history + PnL, plots).
    """
    if not symbols:
        print("No symbols selected. Use option 1 in the main menu to search/add.")
        _press_enter()
        return

    data_cfg = DEFAULT_CONFIG.data
    sma_cfg = DEFAULT_CONFIG.sma

    while True:
        print("\n=== SMA Crossover Signals ===")
        print("Selected symbols:", ", ".join(symbols))
        print("\nChoose:")
        print("  1. Check today's SMA signal")
        print("  2. Show last N crossover events (history + PnL)")
        print("  3. Plot interactive SMA crossover chart")
        print("  4. Back to main menu\n")

        choice = input("Select option [1-4]: ").strip()

        if choice == "1":
            _run_sma_today_signals(symbols, data_cfg, sma_cfg)
            _press_enter()
        elif choice == "2":
            _run_sma_history_and_pnl(symbols, data_cfg, sma_cfg)
        elif choice == "3":
            _run_sma_plot(symbols, data_cfg, sma_cfg)
        elif choice == "4":
            return
        else:
            print("Invalid choice, try again.")
            _press_enter()


# -------------------------------------------------------------------
# ML menu action (using model bundle + metadata)
# -------------------------------------------------------------------


def _run_ml_signals(symbols: List[str]):
    if not symbols:
        print("No symbols selected. Use option 1 in the main menu to search/add.")
        _press_enter()
        return

    model_path = input("Enter path to trained model bundle (.joblib): ").strip()
    if not model_path:
        print("No model path provided; cancelled.")
        _press_enter()
        return

    try:
        prob_cutoff = float(input("Probability cutoff for BUY [default: 0.6]: ").strip() or "0.6")
    except ValueError:
        print("Invalid number. Using 0.6.")
        prob_cutoff = 0.6

    print("\nRunning ML signals...")
    df = generate_ml_signals_for_universe(
        model_path=model_path,
        universe=symbols,
        prob_cutoff=prob_cutoff,
    )

    if df.empty:
        print("No ML signals today.")
    else:
        print("\nML Signals:")
        print(df)
        out_path = "today_signals_ml_interactive.csv"
        df.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}")

    _press_enter()


# -------------------------------------------------------------------
# Ticker "search" (simple local universe)
# -------------------------------------------------------------------


def _search_tickers() -> str | None:
    """
    Search Yahoo Finance for a symbol and let the user pick one.

    This uses the same backend as `quant-signal search-symbol`, but in interactive mode.
    """
    query = input("Enter company name or symbol (or empty to cancel): ").strip()
    if not query:
        return None

    exch_input = input("Optional exchange filter [NSE/BSE or ENTER for any]: ").strip().upper()
    exchange_filter: str | None = None
    if exch_input in ("NSE", "NSI"):
        exchange_filter = "NSE"
    elif exch_input in ("BSE",):
        exchange_filter = "BSE"
    elif exch_input:
        print(f"Unrecognized exchange '{exch_input}', ignoring filter.")

    cfg = YahooSearchConfig(region="IN", quotes_count=20)

    try:
        df = search_yahoo_symbols(query, cfg=cfg, exchange_filter=exchange_filter)
    except Exception as exc:
        print(f"Error while searching Yahoo: {exc}")
        return None

    if df.empty:
        print(f"No matches found for '{query}'.")
        return None

    # Show results
    print("\nMatches:")
    for i, row in df.iterrows():
        symbol = row["symbol"]
        exchange = row["exchange"]
        shortname = row.get("shortname", "") or ""
        print(f"  {i+1}. {symbol:12} [{exchange}]  {shortname}")

    while True:
        choice = input("\nSelect number (or ENTER to cancel): ").strip()
        if choice == "":
            return None
        if not choice.isdigit():
            print("Please enter a valid number.")
            continue

        idx = int(choice) - 1
        if 0 <= idx < len(df):
            chosen = df.iloc[idx]["symbol"]
            print(f"Selected: {chosen}")
            return str(chosen)
        else:
            print("Number out of range.")

# -------------------------------------------------------------------
# Main interactive shell
# -------------------------------------------------------------------


def run_shell():
    """
    Main interactive menu-driven session.

    This is a TUI (text UI) on top of the production-grade core modules.
    """
    selected_symbols: List[str] = []

    while True:
        _clear_screen()
        print("=== Quant-Mill Interactive ===\n")
        print("Current symbols:", ", ".join(selected_symbols) if selected_symbols else "(none)")
        print("\nMenu:")
        print("  1. Search & add ticker by name/symbol")
        print("  2. Clear selected symbols")
        print("  3. SMA crossover tools")
        print("  4. ML signal scan (using trained model)")
        print("  5. Exit\n")

        choice = input("Choose an option [1-5]: ").strip()

        if choice == "1":
            symbol = _search_tickers()
            if symbol and symbol not in selected_symbols:
                selected_symbols.append(symbol)
                print(f"\nAdded {symbol} to selected symbols.")
            elif symbol:
                print(f"\n{symbol} is already in the list.")
            _press_enter()

        elif choice == "2":
            selected_symbols.clear()
            print("Cleared selected symbols.")
            _press_enter()

        elif choice == "3":
            _sma_menu(selected_symbols)

        elif choice == "4":
            _run_ml_signals(selected_symbols)

        elif choice == "5":
            print("Goodbye.")
            break

        else:
            print("Invalid choice.")
            _press_enter()
