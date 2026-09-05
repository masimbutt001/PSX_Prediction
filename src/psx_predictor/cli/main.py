"""Main entrypoint for the PSX Predictor Typer CLI."""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from psx_predictor import __version__
from psx_predictor.config.loader import load_config

app = typer.Typer(
    name="psx",
    help="Pakistan Stock Exchange (PSX) Prediction and Analytics Platform CLI",
    add_completion=False,
    no_args_is_help=True,
)
config_app = typer.Typer(help="Manage and inspect platform configuration")
app.add_typer(config_app, name="config")

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

    # 3. Summary Footer
    total = len(cfg.stocks)
    enabled = len(cfg.get_enabled_stocks())
    console.print(
        f"[dim]Universe summary: [bold]{enabled}[/bold] active / "
        f"[bold]{total}[/bold] total stocks configured.[/dim]\n"
    )


if __name__ == "__main__":
    app()
