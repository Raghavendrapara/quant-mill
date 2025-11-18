# src/quant_signal/interactive.py

from __future__ import annotations

from typing import List

from quant_signal.config import TICKER_UNIVERSE
from quant_signal.signals.generator import generate_signals
from quant_signal.signals.ml_signals import generate_ml_signals_for_universe


def _clear_screen():
    # very simple clear; works ok on most terminals
    print("\n" * 80)


def _press_enter_to_continue():
    input("\nPress ENTER to return to the main menu...")


def search_tickers() -> str | None:
    """
    Ask user for a partial name/symbol and let them pick a matching ticker.
    Returns the chosen symbol, or None if cancelled.
    """
    query = input("Enter company name or symbol (or empty to cancel): ").strip()
    if not query:
        return None

    q = query.lower()
    matches = [
        t for t in TICKER_UNIVERSE
        if q in t["symbol"].lower() or q in t["name"].lower()
    ]

    if not matches:
        print(f"No matches found for '{query}'.")
        return None

    print("\nMatches:")
    for i, t in enumerate(matches, start=1):
        print(f"  {i}. {t['symbol']:12}  {t['name']}")

    while True:
        choice = input("\nSelect number (or ENTER to cancel): ").strip()
        if choice == "":
            return None
        if not choice.isdigit():
            print("Please enter a valid number.")
            continue

        idx = int(choice)
        if 1 <= idx <= len(matches):
            chosen = matches[idx - 1]["symbol"]
            print(f"Selected: {chosen}")
            return chosen
        else:
            print("Number out of range.")


def run_sma_signals(symbols: List[str]):
    if not symbols:
        print("No symbols selected yet. Use option 1 to search and add.")
        return

    print("\nRunning SMA crossover signals for:", ", ".join(symbols))
    df = generate_signals(symbols)
    if df.empty:
        print("No SMA signals today.")
    else:
        print("\nSignals:")
        print(df)
        df.to_csv("today_signals_sma.csv", index=False)
        print("\nSaved to today_signals_sma.csv")


def run_ml_signals(symbols: List[str]):
    if not symbols:
        print("No symbols selected yet. Use option 1 to search and add.")
        return

    model_path = input("Enter path to trained model (.joblib): ").strip()
    if not model_path:
        print("No model path provided; cancelled.")
        return

    start = input("Start date [default: 2015-01-01]: ").strip() or "2015-01-01"

    # simple numeric inputs with defaults
    def _float_input(prompt: str, default: float) -> float:
        s = input(f"{prompt} [default: {default}]: ").strip()
        if not s:
            return default
        try:
            return float(s)
        except ValueError:
            print("Invalid number; using default.")
            return default

    horizon = int(_float_input("Forward horizon (days)", 5))
    threshold = _float_input("Forward return threshold", 0.02)
    prob_cutoff = _float_input("Probability cutoff for BUY", 0.6)

    print("\nRunning ML signals...")
    df = generate_ml_signals_for_universe(
        model_path=model_path,
        universe=symbols,
        start=start,
        horizon=horizon,
        threshold=threshold,
        prob_cutoff=prob_cutoff,
    )
    if df.empty:
        print("No ML signals today.")
    else:
        print("\nML Signals:")
        print(df)
        df.to_csv("today_signals_ml.csv", index=False)
        print("\nSaved to today_signals_ml.csv")


def run_shell():
    """
    Start an interactive CLI session with menus.
    """
    selected_symbols: List[str] = []

    while True:
        _clear_screen()
        print("=== Quant-Signal Interactive ===\n")
        print("Current symbols:", ", ".join(selected_symbols) if selected_symbols else "(none)")
        print("\nMenu:")
        print("  1. Search & add ticker by name/symbol")
        print("  2. Clear selected symbols")
        print("  3. Run SMA crossover signals on selected symbols")
        print("  4. Run ML signals on selected symbols")
        print("  5. Exit\n")

        choice = input("Choose an option [1-5]: ").strip()

        if choice == "1":
            symbol = search_tickers()
            if symbol and symbol not in selected_symbols:
                selected_symbols.append(symbol)
                print(f"\nAdded {symbol} to selected symbols.")
            elif symbol:
                print(f"\n{symbol} is already in the list.")
            _press_enter_to_continue()

        elif choice == "2":
            selected_symbols.clear()
            print("Cleared selected symbols.")
            _press_enter_to_continue()

        elif choice == "3":
            run_sma_signals(selected_symbols)
            _press_enter_to_continue()

        elif choice == "4":
            run_ml_signals(selected_symbols)
            _press_enter_to_continue()

        elif choice == "5":
            print("Goodbye.")
            break

        else:
            print("Invalid choice.")
            _press_enter_to_continue()
