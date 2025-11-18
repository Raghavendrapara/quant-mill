import click
from quant_signal.signals.generator import generate_signals
from quant_signal.models.train import train_random_forest
from quant_signal.signals.ml_signals import generate_ml_signals_for_universe
from quant_signal.interactive import run_shell


@click.group()
def cli():
    """Quant-Mill command line interface."""
    pass


@cli.command()
@click.option("--symbols", "-s", multiple=True, required=True,
              help="List of tickers e.g. --symbols TCS.NS RELIANCE.NS")
def signals(symbols):
    """Rule-based SMA crossover signals."""
    df = generate_signals(list(symbols))
    if df.empty:
        click.echo("No signals today (SMA).")
    else:
        click.echo(df)
        df.to_csv("today_signals_sma.csv", index=False)
        click.echo("Saved to today_signals_sma.csv")


@cli.command("train-ml")
@click.option("--symbol", "-s", required=True, help="Single symbol to train on (e.g. TCS.NS)")
@click.option("--start", default="2015-01-01")
@click.option("--horizon", default=5, show_default=True, help="Forward return horizon in days")
@click.option("--threshold", default=0.02, show_default=True, help="Forward return threshold")
def train_ml_cmd(symbol, start, horizon, threshold):
    """Train RandomForest ML model for one symbol."""
    train_random_forest(symbol, start=start, horizon=horizon, threshold=threshold)


@cli.command("signals-ml")
@click.option("--symbols", "-s", multiple=True, required=True,
              help="Universe of tickers")
@click.option("--model-path", required=True, help="Path to trained model .joblib")
@click.option("--start", default="2015-01-01")
@click.option("--horizon", default=5, show_default=True)
@click.option("--threshold", default=0.02, show_default=True)
@click.option("--prob-cutoff", default=0.6, show_default=True)
def signals_ml_cmd(symbols, model_path, start, horizon, threshold, prob_cutoff):
    """ML-based signals using a trained RandomForest model."""
    df = generate_ml_signals_for_universe(
        model_path, list(symbols),
        start=start,
        horizon=horizon,
        threshold=threshold,
        prob_cutoff=prob_cutoff,
    )
    if df.empty:
        click.echo("No ML signals today.")
    else:
        click.echo(df)
        df.to_csv("today_signals_ml.csv", index=False)
        click.echo("Saved to today_signals_ml.csv")

@cli.command("interactive")
def interactive_cmd():
    """Start interactive menu-driven session."""
    run_shell()