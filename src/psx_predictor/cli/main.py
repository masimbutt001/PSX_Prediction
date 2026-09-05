"""Main entrypoint for the PSX Predictor Typer CLI."""

import json
from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from psx_predictor import __version__
from psx_predictor.config.loader import load_config
from psx_predictor.storage.duckdb_client import DuckDBClient
from psx_predictor.storage.parquet_io import read_parquet
from psx_predictor.storage.paths import ensure_directories, get_storage_paths
from psx_predictor.storage.schemas import PriceDatasetValidator

app = typer.Typer(
    name="psx",
    help="Pakistan Stock Exchange (PSX) Prediction and Analytics Platform CLI",
    add_completion=False,
    no_args_is_help=True,
)
config_app = typer.Typer(help="Manage and inspect platform configuration")
storage_app = typer.Typer(help="Manage and inspect local analytical storage & DuckDB")
data_app = typer.Typer(help="Inspect, validate, and query market datasets")

app.add_typer(config_app, name="config")
app.add_typer(storage_app, name="storage")
app.add_typer(data_app, name="data")

console = Console()


@app.command()
def version() -> None:
    """Show the currently installed PSX Predictor version."""
    console.print(f"[bold green]PSX Predictor[/bold green] version [cyan]{__version__}[/cyan]")


@config_app.command("show")
def config_show(
    config_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--config-dir",
            "-c",
            help="Path to custom directory containing configuration YAML files",
        ),
    ] = None,
) -> None:
    """Display active application settings and configured PSX stock universe."""
    try:
        cfg = load_config(config_dir)
    except Exception as e:
        console.print(f"[bold red]Error loading configuration:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    # 1. Application Settings Panel
    settings_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    settings_table.add_column("Key", style="bold cyan")
    settings_table.add_column("Value", style="white")

    settings_table.add_row("Root Data Dir", str(cfg.settings.data_dir))
    settings_table.add_row("Raw Data Dir", str(cfg.settings.raw_dir))
    settings_table.add_row("Processed Data Dir", str(cfg.settings.processed_dir))
    settings_table.add_row("Features Dir", str(cfg.settings.features_dir))
    settings_table.add_row("Predictions Dir", str(cfg.settings.predictions_dir))
    settings_table.add_row("Models Dir", str(cfg.settings.models_dir))
    settings_table.add_row("Reports Dir", str(cfg.settings.reports_dir))
    settings_table.add_row("Log Level", cfg.settings.log_level)
    settings_table.add_row("Timezone", cfg.settings.timezone)

    console.print(
        Panel(
            settings_table,
            title="[bold green]PSX Predictor — Active Settings[/bold green]",
            border_style="green",
        )
    )

    # 2. Stock Universe Table
    stock_table = Table(
        title="Configured PSX Stock Universe",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    stock_table.add_column("#", justify="right", style="dim")
    stock_table.add_column("Symbol", style="bold yellow")
    stock_table.add_column("Name", style="white")
    stock_table.add_column("Sector", style="cyan")
    stock_table.add_column("Yahoo Ticker", style="blue")
    stock_table.add_column("Status", justify="center")

    for idx, s in enumerate(cfg.stocks, 1):
        status = "[bold green]ENABLED[/bold green]" if s.enabled else "[dim red]DISABLED[/dim red]"
        stock_table.add_row(str(idx), s.symbol, s.name, s.sector, s.yahoo_ticker, status)

    console.print(stock_table)

    total = len(cfg.stocks)
    enabled = len(cfg.get_enabled_stocks())
    console.print(
        f"[dim]Universe summary: [bold]{enabled}[/bold] active / "
        f"[bold]{total}[/bold] total stocks configured.[/dim]\n"
    )


@storage_app.command("init")
def storage_init(
    base_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--base-dir",
            "-d",
            help="Base directory for analytical data storage (defaults to config settings)",
        ),
    ] = None,
) -> None:
    """Initialize all standard local analytical storage directories."""
    cfg = load_config()
    target_dir = base_dir or cfg.settings.data_dir

    paths = ensure_directories(target_dir)

    table = Table(title="Initialized Storage Directories", box=box.ROUNDED)
    table.add_column("Storage Key", style="cyan")
    table.add_column("Path", style="green")
    table.add_column("Status", style="bold green", justify="center")

    for key, path in paths.items():
        table.add_row(key, str(path), "READY")

    console.print(table)
    console.print(f"[bold green]Storage successfully initialized at:[/bold green] {target_dir}\n")


@storage_app.command("status")
def storage_status(
    base_dir: Annotated[
        Optional[Path],
        typer.Option("--base-dir", "-d", help="Base directory for analytical data storage"),
    ] = None,
) -> None:
    """Check storage directory structure and DuckDB query engine status."""
    cfg = load_config()
    target_dir = base_dir or cfg.settings.data_dir
    paths = get_storage_paths(target_dir)

    table = Table(title=f"Storage Status — '{target_dir}'", box=box.ROUNDED)
    table.add_column("Subsystem", style="cyan")
    table.add_column("Path", style="dim")
    table.add_column("Exists", justify="center")
    table.add_column("Parquet Files", justify="right")

    for key, p in paths.items():
        exists = p.exists()
        parquet_count = len(list(p.glob("*.parquet"))) if exists and p.is_dir() else 0
        exists_str = "[green]YES[/green]" if exists else "[red]NO[/red]"
        table.add_row(key, str(p), exists_str, str(parquet_count))

    console.print(table)

    # Verify DuckDB connectivity
    try:
        with DuckDBClient() as db:
            ver = db.query("SELECT version() as ver").iloc[0]["ver"]
            console.print(f"[bold green]DuckDB Engine:[/bold green] Connected (version {ver})")
    except Exception as e:
        console.print(f"[bold red]DuckDB Engine Error:[/bold red] {e}")


@data_app.command("inspect")
def data_inspect(
    file_path: Annotated[
        Optional[Path],
        typer.Option("--file", "-f", help="Path to Parquet file to inspect"),
    ] = None,
    fixture_path: Annotated[
        Optional[Path],
        typer.Option("--fixture", help="Path to JSON fixture file to validate and inspect"),
    ] = None,
) -> None:
    """Inspect and validate an OHLCV dataset or test fixture."""
    if file_path is None and fixture_path is None:
        console.print("[bold red]Error:[/bold red] Specify either --file or --fixture")
        raise typer.Exit(code=1)

    df: pd.DataFrame
    if fixture_path is not None:
        if not fixture_path.exists():
            console.print(f"[bold red]Fixture file not found:[/bold red] {fixture_path}")
            raise typer.Exit(code=1)
        with open(fixture_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        raw_df = pd.DataFrame(raw_data)
        console.print(f"Validating {len(raw_df)} records from fixture: {fixture_path}")
        df = PriceDatasetValidator.validate_dataframe(raw_df)
    else:
        assert file_path is not None
        df = read_parquet(file_path)

    # Show dataset summary
    summary_table = Table(box=box.ROUNDED, title="Dataset Summary")
    summary_table.add_column("Metric", style="bold cyan")
    summary_table.add_column("Value", style="white")

    summary_table.add_row("Total Records", str(len(df)))
    symbols = df["symbol"].unique().tolist() if "symbol" in df.columns else []
    summary_table.add_row("Symbols", ", ".join(symbols))
    if "trade_date" in df.columns:
        summary_table.add_row("Start Date", str(df["trade_date"].min()))
        summary_table.add_row("End Date", str(df["trade_date"].max()))

    console.print(summary_table)

    # Show sample rows
    preview = df.head(5)
    preview_table = Table(title="Sample Rows Preview (First 5)", box=box.SIMPLE_HEAVY)
    for col in preview.columns:
        preview_table.add_column(col, style="dim" if col == "index" else "white")

    for _, row in preview.iterrows():
        row_vals = [str(val) for val in row.values]
        preview_table.add_row(*row_vals)

    console.print(preview_table)


if __name__ == "__main__":
    app()
