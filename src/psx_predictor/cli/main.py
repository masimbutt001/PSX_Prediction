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
features_app = typer.Typer(help="Calculate and manage engineered feature stores")
model_app = typer.Typer(help="Train, evaluate, and benchmark predictive models")

app.add_typer(config_app, name="config")
app.add_typer(storage_app, name="storage")
app.add_typer(data_app, name="data")
app.add_typer(features_app, name="features")
app.add_typer(model_app, name="model")

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


@data_app.command("fetch")
def data_fetch(
    symbol: Annotated[
        str,
        typer.Option("--symbol", "-s", help="PSX Ticker symbol to fetch (e.g. OGDC)"),
    ],
    source: Annotated[
        str,
        typer.Option(
            "--source",
            help="Ingestion source: 'composite', 'scs', 'yahoo', or 'dps'",
        ),
    ] = "composite",
    days: Annotated[
        int,
        typer.Option("--days", "-d", help="Lookback window in days (default: 30)"),
    ] = 30,
    start_date: Annotated[
        Optional[str],
        typer.Option("--start-date", help="Custom start date (YYYY-MM-DD)"),
    ] = None,
    end_date: Annotated[
        Optional[str],
        typer.Option("--end-date", help="Custom end date (YYYY-MM-DD)"),
    ] = None,
    save: Annotated[
        bool,
        typer.Option("--save/--no-save", help="Whether to save normalized data to storage"),
    ] = True,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Fetch and validate without writing to disk"),
    ] = False,
) -> None:
    """Fetch historical market data from SCS Trade, Yahoo Finance, or PSX DPS."""
    import datetime

    from psx_predictor.collectors.base import BaseCollector
    from psx_predictor.collectors.composite import CompositeCollector
    from psx_predictor.collectors.dps_collector import DPSCollector
    from psx_predictor.collectors.scs_collector import SCSTradeCollector
    from psx_predictor.collectors.yahoo_collector import YahooCollector
    from psx_predictor.storage.parquet_io import write_parquet_atomic

    # Resolve dates
    resolved_end = datetime.date.fromisoformat(end_date) if end_date else datetime.date.today()
    resolved_start = (
        datetime.date.fromisoformat(start_date)
        if start_date
        else (resolved_end - datetime.timedelta(days=days))
    )

    clean_sym = symbol.strip().upper()
    console.print(
        f"Fetching [bold yellow]{clean_sym}[/bold yellow] via provider [cyan]'{source}'[/cyan] "
        f"({resolved_start} to {resolved_end})..."
    )

    collector: BaseCollector
    src_lower = source.lower()
    if src_lower == "scs":
        collector = SCSTradeCollector()
    elif src_lower == "yahoo":
        collector = YahooCollector()
    elif src_lower == "dps":
        collector = DPSCollector()
    elif src_lower == "composite":
        collector = CompositeCollector()
    else:
        console.print(
            f"[bold red]Unknown source:[/bold red] '{source}'. "
            "Choose 'composite', 'scs', 'yahoo', or 'dps'."
        )
        raise typer.Exit(code=1)

    try:
        df = collector.fetch_historical(
            symbol=clean_sym,
            start_date=resolved_start,
            end_date=resolved_end,
            save_raw_payload=not dry_run,
        )
    except Exception as e:
        console.print(f"[bold red]Fetch error:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    # Summary table
    table = Table(title=f"Fetch Results for {clean_sym} ({collector.source_name})", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold white")

    table.add_row("Sessions Retrieved", str(len(df)))
    table.add_row("Date Bounds", f"{df['trade_date'].min()} to {df['trade_date'].max()}")
    table.add_row("Latest Close", f"PKR {df['close'].iloc[-1]:.2f}")
    table.add_row("Upper Lock Sessions", str(int(df["is_upper_lock"].sum())))
    table.add_row("Lower Lock Sessions", str(int(df["is_lower_lock"].sum())))

    console.print(table)

    # Save to processed storage if requested
    if save and not dry_run:
        cfg = load_config()
        paths = get_storage_paths(cfg.settings.data_dir)
        dest_file = paths["processed_prices"] / f"{clean_sym}.parquet"
        write_parquet_atomic(df, dest_file)
        console.print(f"[bold green]Saved processed Parquet to:[/bold green] {dest_file}")
    elif dry_run:
        console.print(
            "[dim yellow]Dry run active: No files saved to processed storage.[/dim yellow]"
        )

    # Preview
    preview = df.tail(5)
    preview_table = Table(title=f"Recent Sessions ({clean_sym})", box=box.SIMPLE_HEAVY)
    display_cols = [
        c
        for c in [
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "is_upper_lock",
            "is_lower_lock",
        ]
        if c in preview.columns
    ]
    for col in display_cols:
        preview_table.add_column(col)

    for _, row in preview.iterrows():
        row_vals = [str(row[col]) for col in display_cols]
        preview_table.add_row(*row_vals)

    console.print(preview_table)


@data_app.command("bootstrap")
def data_bootstrap(
    years: Annotated[
        int,
        typer.Option("--years", "-y", help="Historical lookback window in years"),
    ] = 5,
    symbols: Annotated[
        Optional[str],
        typer.Option(
            "--symbols",
            "-s",
            help="Comma-separated symbols (e.g. 'OGDC,PPL'). If omitted, runs all enabled.",
        ),
    ] = None,
    source: Annotated[
        str,
        typer.Option(
            "--source",
            help="Data source to use: 'composite', 'scs', 'yahoo', or 'dps'",
        ),
    ] = "composite",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate without writing files to disk"),
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite/--no-overwrite", help="Overwrite existing processed datasets"),
    ] = True,
) -> None:
    """Bootstrap multi-year historical market data for configured PSX symbols."""
    from psx_predictor.collectors.base import BaseCollector
    from psx_predictor.collectors.composite import CompositeCollector
    from psx_predictor.collectors.dps_collector import DPSCollector
    from psx_predictor.collectors.scs_collector import SCSTradeCollector
    from psx_predictor.collectors.yahoo_collector import YahooCollector
    from psx_predictor.processing.pipeline import DataBootstrapPipeline

    collector: BaseCollector
    src_lower = source.lower()
    if src_lower == "scs":
        collector = SCSTradeCollector()
    elif src_lower == "yahoo":
        collector = YahooCollector()
    elif src_lower == "dps":
        collector = DPSCollector()
    elif src_lower == "composite":
        collector = CompositeCollector()
    else:
        console.print(
            f"[bold red]Unknown source:[/bold red] '{source}'. "
            "Choose 'composite', 'scs', 'yahoo', or 'dps'."
        )
        raise typer.Exit(code=1)

    target_syms = [s.strip().upper() for s in symbols.split(",") if s.strip()] if symbols else None

    console.print(
        f"Starting historical bootstrap: [cyan]{years} years[/cyan] lookback | "
        f"Source: [yellow]'{collector.source_name}'[/yellow] | "
        f"Mode: {'[magenta]DRY-RUN[/magenta]' if dry_run else '[green]PERSIST[/green]'}"
    )

    pipeline = DataBootstrapPipeline(collector=collector)
    universe_result = pipeline.bootstrap_universe(
        symbols=target_syms,
        years=years,
        dry_run=dry_run,
        overwrite=overwrite,
    )

    # Render summary table
    table = Table(title="Historical Bootstrap Execution Results", box=box.ROUNDED)
    table.add_column("Symbol", style="bold cyan")
    table.add_column("Status", justify="center")
    table.add_column("Records", justify="right")
    table.add_column("Date Span", justify="center")
    table.add_column("Locks (U/L)", justify="center")
    table.add_column("Dividends", justify="right")
    table.add_column("Duration", justify="right")

    for sym, res in universe_result.symbol_results.items():
        if res.status == "SUCCESS":
            status_style = "[bold green]SUCCESS[/bold green]"
        elif res.status == "SKIPPED":
            status_style = "[dim yellow]SKIPPED[/dim yellow]"
        else:
            status_style = f"[bold red]FAILED[/bold red] ({res.error_message})"

        span_str = f"{res.start_date} -> {res.end_date}" if res.start_date else "N/A"
        locks_str = f"{res.upper_locks} / {res.lower_locks}"
        table.add_row(
            sym,
            status_style,
            str(res.record_count),
            span_str,
            locks_str,
            str(res.dividend_events),
            f"{res.duration_seconds:.2f}s",
        )

    console.print(table)

    summary_panel = Panel(
        f"[bold]Total Symbols:[/bold] {universe_result.total_symbols}  |  "
        f"[bold green]Succeeded:[/bold green] {universe_result.successful_symbols}  |  "
        f"[bold yellow]Skipped:[/bold yellow] {universe_result.skipped_symbols}  |  "
        f"[bold red]Failed:[/bold red] {universe_result.failed_symbols}\n"
        f"[bold cyan]Total Records Ingested:[/bold cyan] {universe_result.total_records:,}",
        title="Bootstrap Summary",
        border_style="cyan",
    )
    console.print(summary_panel)

    if dry_run:
        console.print("[dim yellow]Dry run active: No files saved to disk.[/dim yellow]")
    else:
        manifest_file = pipeline.storage_paths["processed_prices"] / "bootstrap_manifest.json"
        if manifest_file.exists():
            console.print(f"[bold green]Saved run manifest to:[/bold green] {manifest_file}")


@data_app.command("audit")
def data_audit(
    symbols: Annotated[
        Optional[str],
        typer.Option(
            "--symbols",
            "-s",
            help="Comma-separated symbols to audit. If omitted, audits all processed datasets.",
        ),
    ] = None,
    max_gap_days: Annotated[
        int,
        typer.Option(
            "--max-gap-days",
            "-g",
            help="Calendar days threshold to flag an anomalous gap (default: 4)",
        ),
    ] = 4,
    json_output: Annotated[
        Optional[Path],
        typer.Option("--json-output", help="Optional path to write audit report as JSON"),
    ] = None,
) -> None:
    """Audit processed datasets for completeness, trading calendar gaps, and anomalies."""
    from psx_predictor.processing.quality_report import DataQualityAuditor

    target_syms = [s.strip().upper() for s in symbols.split(",") if s.strip()] if symbols else None
    auditor = DataQualityAuditor()

    # Single symbol audit
    if target_syms and len(target_syms) == 1:
        sym = target_syms[0]
        report = auditor.audit_symbol(sym, max_gap_days=max_gap_days)
        console.print(report.to_rich_table())

        if report.gaps:
            gap_table = Table(
                title=f"Detected Gaps (> {max_gap_days} days) for {sym}", box=box.SIMPLE_HEAVY
            )
            gap_table.add_column("Gap Interval", style="cyan")
            gap_table.add_column("Calendar Days", justify="right")
            gap_table.add_column("Missing Weekdays", justify="right", style="bold red")

            for g in report.gaps:
                gap_table.add_row(
                    f"{g.start_date} to {g.end_date}", str(g.calendar_days), str(g.missing_weekdays)
                )
            console.print(gap_table)

        if json_output:
            with open(json_output, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)
            console.print(f"[bold green]Saved JSON report to:[/bold green] {json_output}")
        return

    # Multi-symbol universe audit
    universe_report = auditor.audit_universe(symbols=target_syms, max_gap_days=max_gap_days)
    console.print(universe_report.to_rich_table())

    summary_panel = Panel(
        f"[bold]Total Audited:[/bold] {universe_report.total_symbols}  |  "
        f"[bold green]Healthy:[/bold green] {universe_report.healthy_count}  |  "
        f"[bold yellow]Warning:[/bold yellow] {universe_report.warning_count}  |  "
        f"[bold red]Critical:[/bold red] {universe_report.critical_count}\n"
        f"[bold cyan]Total Analyzed Sessions:[/bold cyan] {universe_report.total_sessions:,}",
        title="Audit Universe Summary",
        border_style="cyan",
    )
    console.print(summary_panel)

    if json_output:
        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(universe_report.to_dict(), f, indent=2)
        console.print(f"[bold green]Saved JSON report to:[/bold green] {json_output}")


@data_app.command("update")
def data_update(
    symbols: Annotated[
        Optional[str],
        typer.Option(
            "--symbols",
            "-s",
            help="Comma-separated symbols to update (e.g. 'OGDC,PPL'). "
            "If omitted, runs all enabled.",
        ),
    ] = None,
    source: Annotated[
        str,
        typer.Option(
            "--source",
            help="Data source to query: 'composite', 'scs', 'yahoo', or 'dps'",
        ),
    ] = "composite",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Fetch and validate delta without writing to disk"),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force", "-f", help="Force querying provider even if latest date matches today"
        ),
    ] = False,
    auto_bootstrap: Annotated[
        bool,
        typer.Option(
            "--auto-bootstrap/--no-auto-bootstrap",
            help="Automatically bootstrap missing datasets",
        ),
    ] = True,
) -> None:
    """Incrementally update local market datasets with newest trading sessions."""
    from psx_predictor.collectors.base import BaseCollector
    from psx_predictor.collectors.composite import CompositeCollector
    from psx_predictor.collectors.dps_collector import DPSCollector
    from psx_predictor.collectors.scs_collector import SCSTradeCollector
    from psx_predictor.collectors.yahoo_collector import YahooCollector
    from psx_predictor.processing.updater import IncrementalUpdater

    collector: BaseCollector
    src_lower = source.lower()
    if src_lower == "scs":
        collector = SCSTradeCollector()
    elif src_lower == "yahoo":
        collector = YahooCollector()
    elif src_lower == "dps":
        collector = DPSCollector()
    elif src_lower == "composite":
        collector = CompositeCollector()
    else:
        console.print(
            f"[bold red]Unknown source:[/bold red] '{source}'. "
            "Choose 'composite', 'scs', 'yahoo', or 'dps'."
        )
        raise typer.Exit(code=1)

    target_syms = [s.strip().upper() for s in symbols.split(",") if s.strip()] if symbols else None

    console.print(
        f"Starting incremental update | Source: [yellow]'{collector.source_name}'[/yellow] | "
        f"Mode: {'[magenta]DRY-RUN[/magenta]' if dry_run else '[green]PERSIST[/green]'}"
    )

    updater = IncrementalUpdater(collector=collector)
    universe_result = updater.update_universe(
        symbols=target_syms,
        dry_run=dry_run,
        force=force,
        auto_bootstrap=auto_bootstrap,
    )

    # Render summary table
    table = Table(title="Incremental Daily Update Execution Results", box=box.ROUNDED)
    table.add_column("Symbol", style="bold cyan")
    table.add_column("Status", justify="center")
    table.add_column("Prev Latest", justify="center")
    table.add_column("New Latest", justify="center")
    table.add_column("Added", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Duration", justify="right")

    for sym, res in universe_result.symbol_results.items():
        if res.status == "UPDATED":
            status_style = "[bold green]UPDATED[/bold green]"
        elif res.status == "ALREADY_UP_TO_DATE":
            status_style = "[cyan]UP TO DATE[/cyan]"
        elif res.status == "BOOTSTRAPPED":
            status_style = "[bold magenta]BOOTSTRAPPED[/bold magenta]"
        else:
            status_style = f"[bold red]FAILED[/bold red] ({res.error_message})"

        prev_str = res.previous_latest_date or "None"
        new_str = res.new_latest_date or "None"
        table.add_row(
            sym,
            status_style,
            prev_str,
            new_str,
            str(res.records_added),
            str(res.total_records),
            f"{res.duration_seconds:.2f}s",
        )

    console.print(table)

    summary_panel = Panel(
        f"[bold]Total Symbols:[/bold] {universe_result.total_symbols}  |  "
        f"[bold green]Updated:[/bold green] {universe_result.updated_symbols}  |  "
        f"[cyan]Up To Date:[/cyan] {universe_result.up_to_date_symbols}  |  "
        f"[bold magenta]Bootstrapped:[/bold magenta] {universe_result.bootstrapped_symbols}  |  "
        f"[bold red]Failed:[/bold red] {universe_result.failed_symbols}\n"
        f"[bold cyan]Total New Records Added:[/bold cyan] {universe_result.total_records_added:,}",
        title="Update Summary",
        border_style="cyan",
    )
    console.print(summary_panel)

    if dry_run:
        console.print("[dim yellow]Dry run active: No files modified on disk.[/dim yellow]")
    else:
        manifest_file = updater.storage_paths["processed_prices"] / "update_manifest.json"
        if manifest_file.exists():
            console.print(f"[bold green]Saved update manifest to:[/bold green] {manifest_file}")


@features_app.command("build")
def features_build(
    symbols: Annotated[
        Optional[str],
        typer.Option(
            "--symbols",
            "-s",
            help="Comma-separated symbols to compute features for (e.g. 'OGDC,PPL'). "
            "If omitted, processes all available datasets in processed storage.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Calculate features without saving to disk"),
    ] = False,
    with_targets: Annotated[
        bool,
        typer.Option(
            "--with-targets/--without-targets",
            help="Append supervised prediction targets (direction, 3-class, return)",
        ),
    ] = True,
) -> None:
    """Compute vectorized technical indicators for processed market datasets."""
    from psx_predictor.features.builder import TechnicalFeatureBuilder

    target_syms = [s.strip().upper() for s in symbols.split(",") if s.strip()] if symbols else None

    console.print(
        "Starting technical feature generation | "
        f"Targets: {'[green]INCLUDED[/green]' if with_targets else '[dim]EXCLUDED[/dim]'} | "
        f"Mode: {'[magenta]DRY-RUN[/magenta]' if dry_run else '[green]PERSIST[/green]'}"
    )

    builder = TechnicalFeatureBuilder()
    universe_result = builder.build_universe(
        symbols=target_syms,
        save=not dry_run,
        with_targets=with_targets,
    )

    # Render summary table
    table = Table(title="Technical Feature Generation Results", box=box.ROUNDED)
    table.add_column("Symbol", style="bold cyan")
    table.add_column("Status", justify="center")
    table.add_column("Sessions", justify="right")
    table.add_column("Features", justify="right")
    table.add_column("Date Span", justify="center")
    table.add_column("Duration", justify="right")

    for sym, res in universe_result.symbol_results.items():
        if res.status == "SUCCESS":
            status_style = "[bold green]SUCCESS[/bold green]"
        else:
            status_style = f"[bold red]FAILED[/bold red] ({res.error_message})"

        span_str = f"{res.start_date} -> {res.end_date}" if res.start_date else "N/A"
        table.add_row(
            sym,
            status_style,
            str(res.total_records),
            str(res.feature_count),
            span_str,
            f"{res.duration_seconds:.2f}s",
        )

    console.print(table)

    summary_panel = Panel(
        f"[bold]Total Symbols:[/bold] {universe_result.total_symbols}  |  "
        f"[bold green]Successful:[/bold green] {universe_result.successful_symbols}  |  "
        f"[bold red]Failed:[/bold red] {universe_result.failed_symbols}\n"
        f"[bold cyan]Total Feature Rows Generated:[/bold cyan] {universe_result.total_records:,}",
        title="Features Summary",
        border_style="cyan",
    )
    console.print(summary_panel)

    if dry_run:
        console.print("[dim yellow]Dry run active: No files saved to disk.[/dim yellow]")
    else:
        manifest_file = builder.storage_paths["features_technical"] / "feature_manifest.json"
        if manifest_file.exists():
            console.print(f"[bold green]Saved feature manifest to:[/bold green] {manifest_file}")


@model_app.command("train")
def model_train(
    symbol: Annotated[
        str,
        typer.Option("--symbol", "-s", help="PSX ticker symbol (e.g. OGDC)"),
    ],
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Model family to evaluate: 'all', 'baselines', or 'logistic'",
        ),
    ] = "all",
    target: Annotated[
        str,
        typer.Option(
            "--target",
            "-t",
            help="Prediction target column (e.g. 'target_next_day_dir')",
        ),
    ] = "target_next_day_dir",
    split_ratio: Annotated[
        float,
        typer.Option(
            "--split-ratio",
            "-r",
            help="Chronological train split ratio between 0.1 and 0.95 (default: 0.8)",
        ),
    ] = 0.8,
    save: Annotated[
        bool,
        typer.Option(
            "--save/--no-save",
            help="Persist fitted model artifacts to analytical storage",
        ),
    ] = False,
) -> None:
    """Train and evaluate baseline and linear models on strict chronological split."""
    from psx_predictor.models.evaluation import display_evaluation_table
    from psx_predictor.models.trainer import ModelTrainer

    clean_sym = symbol.strip().upper()
    console.print(
        f"Initiating model benchmark for [bold cyan]{clean_sym}[/bold cyan] | "
        f"Family: [yellow]{model}[/yellow] | Target: [magenta]{target}[/magenta] | "
        f"Train Split: [green]{split_ratio:.0%}[/green]"
    )

    trainer = ModelTrainer()
    try:
        results = trainer.train_and_evaluate(
            symbol=clean_sym,
            model_type=model.lower(),
            target_col=target,
            train_ratio=split_ratio,
            save_models=save,
        )
    except FileNotFoundError as fnf:
        console.print(f"[bold red]Data Error:[/bold red] {fnf}")
        raise typer.Exit(code=1) from fnf
    except Exception as exc:
        console.print(f"[bold red]Training Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    display_evaluation_table(results=results, symbol=clean_sym, console=console)


@app.command("backtest")
def backtest_command(
    symbol: Annotated[
        str,
        typer.Option("--symbol", "-s", help="PSX ticker symbol to backtest (e.g. OGDC)"),
    ],
    strategy: Annotated[
        str,
        typer.Option(
            "--strategy",
            help="Trading strategy: 'sma_crossover', 'naive_persistence', or 'logistic'",
        ),
    ] = "sma_crossover",
    initial_cash: Annotated[
        float,
        typer.Option(
            "--initial-cash",
            help="Starting cash portfolio allocation in PKR (default: 1,000,000)",
        ),
    ] = 1_000_000.0,
    rf_rate: Annotated[
        float,
        typer.Option(
            "--rf-rate",
            help="Annual risk-free benchmark rate (default: 0.15 for 15% SBP policy rate)",
        ),
    ] = 0.15,
) -> None:
    """Simulate trading strategy with realistic PSX transaction frictions and circuit locks."""
    from psx_predictor.backtesting.metrics import display_backtest_report
    from psx_predictor.backtesting.runner import BacktestRunner

    clean_sym = symbol.strip().upper()
    console.print(
        f"Initiating backtest for [bold cyan]{clean_sym}[/bold cyan] | "
        f"Strategy: [yellow]{strategy}[/yellow] | Capital: [green]PKR {initial_cash:,.0f}[/green]"
    )

    runner = BacktestRunner()
    try:
        sim_result, metrics = runner.run_strategy(
            symbol=clean_sym,
            strategy=strategy,
            initial_cash=initial_cash,
            risk_free_rate=rf_rate,
        )
    except FileNotFoundError as fnf:
        console.print(f"[bold red]Data Error:[/bold red] {fnf}")
        raise typer.Exit(code=1) from fnf
    except Exception as exc:
        console.print(f"[bold red]Backtest Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    display_backtest_report(metrics=metrics, result=sim_result, console=console)


if __name__ == "__main__":
    app()
