"""Daily EOD pipeline orchestrator and daemon runner."""

import datetime
import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from loguru import logger
from rich import box
from rich.console import Console
from rich.table import Table

from psx_predictor.config.loader import load_config
from psx_predictor.scheduler.jobs import (
    DailyPipelineStep,
    PipelineRunReport,
    PipelineTaskResult,
    create_cron_triggers,
)
from psx_predictor.storage.paths import ensure_directories


class DailyPipelineRunner:
    """Orchestrates end-to-end daily operational routine for the PSX Predictor platform."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        config_dir: Optional[Path] = None,
        log_path: Optional[Path] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        console: Optional[Console] = None,
    ) -> None:
        self.config = load_config(config_dir)
        target_dir = data_dir or self.config.settings.data_dir
        self.paths = ensure_directories(target_dir)
        self.log_file = log_path or (self.paths["root"] / "pipeline.log")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.console = console or Console()
        self._target_symbols: Optional[list[str]] = None

    def _execute_with_retry(
        self,
        action: Callable[[], Any],
        step_name: str,
        max_retries: Optional[int] = None,
        initial_delay: Optional[float] = None,
    ) -> Any:
        """Execute callable with exponential backoff on transient errors."""
        retries = max_retries if max_retries is not None else self.max_retries
        delay = initial_delay if initial_delay is not None else self.retry_delay
        last_exc: Optional[Exception] = None
        for attempt in range(retries):
            try:
                return action()
            except Exception as exc:
                last_exc = exc
                wait_time = delay * (2**attempt)
                logger.warning(
                    f"Step '{step_name}' failed attempt {attempt + 1}/{retries}: {exc}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)

        if last_exc:
            raise last_exc
        raise RuntimeError(f"Step '{step_name}' failed after {retries} attempts.")

    def run_step(self, step: DailyPipelineStep, dry_run: bool = False) -> PipelineTaskResult:
        """Execute a single pipeline step in topological order."""
        t0 = time.perf_counter()
        logger.info(f"Executing pipeline step: {step.value} (dry_run={dry_run})")

        try:
            if step == DailyPipelineStep.UPDATE_PRICES:
                msg = self._step_update_prices(dry_run=dry_run)
            elif step == DailyPipelineStep.COLLECT_NEWS:
                msg = self._step_collect_news(dry_run=dry_run)
            elif step == DailyPipelineStep.UPDATE_MACRO:
                msg = self._step_update_macro(dry_run=dry_run)
            elif step == DailyPipelineStep.BUILD_FEATURES:
                msg = self._step_build_features(dry_run=dry_run)
            elif step == DailyPipelineStep.GENERATE_PREDICTIONS:
                msg = self._step_generate_predictions(dry_run=dry_run)
            elif step == DailyPipelineStep.AUDIT_PREDICTIONS:
                msg = self._step_audit_predictions(dry_run=dry_run)
            else:
                raise ValueError(f"Unknown pipeline step: {step}")

            duration = time.perf_counter() - t0
            return PipelineTaskResult(
                step=step,
                status="SUCCESS",
                duration_seconds=duration,
                message=msg,
            )

        except Exception as exc:
            duration = time.perf_counter() - t0
            err_msg = str(exc)
            logger.error(f"Error in step '{step.value}': {err_msg}")
            return PipelineTaskResult(
                step=step,
                status="FAILED",
                duration_seconds=duration,
                message=f"Step execution failed: {err_msg}",
                error=err_msg,
            )

    def _step_update_prices(self, dry_run: bool) -> str:
        """Step 1: Fetch today's closing prices for all enabled stocks."""
        symbols = self._target_symbols or [s.symbol for s in self.config.get_enabled_stocks()]
        if dry_run:
            return f"[DRY-RUN] Simulated price update for {len(symbols)} enabled stocks"

        from psx_predictor.data.collector import MarketDataCollector
        from psx_predictor.processing.updater import IncrementalUpdater

        collector = MarketDataCollector()
        updater = IncrementalUpdater(storage_paths=self.paths, collector=collector)
        res = self._execute_with_retry(
            lambda: updater.update_universe(symbols=symbols),
            step_name="update_prices",
        )
        return f"Updated {res.successful_updates}/{res.total_symbols} stock price series"

    def _step_collect_news(self, dry_run: bool) -> str:
        """Step 2: Collect latest financial news & corporate announcements and score sentiment."""
        if dry_run:
            return "[DRY-RUN] Simulated news collection and FinBERT sentiment scoring"

        from psx_predictor.news.dps_announcements import DPSAnnouncementsCollector
        from psx_predictor.news.processor import NewsNLPProcessor
        from psx_predictor.news.rss_collector import RSSNewsCollector

        rss_col = RSSNewsCollector(storage_paths=self.paths)
        dps_col = DPSAnnouncementsCollector(storage_paths=self.paths)
        processor = NewsNLPProcessor(storage_paths=self.paths)

        rss_res = self._execute_with_retry(
            lambda: rss_col.fetch_all(dry_run=False),
            step_name="collect_news_rss",
        )
        dps_res = self._execute_with_retry(
            lambda: dps_col.fetch_announcements(count=50, dry_run=False),
            step_name="collect_news_dps",
        )
        proc_res = processor.process_all(dry_run=False)

        return (
            f"Ingested {rss_res.new_articles_added} RSS, {dps_res.new_announcements} DPS, "
            f"created {proc_res.new_signals_created} signals"
        )

    def _step_update_macro(self, dry_run: bool) -> str:
        """Step 3: Synchronize USD/PKR, Brent crude, KIBOR, and policy rates."""
        if dry_run:
            return "[DRY-RUN] Simulated macroeconomic indicator synchronization"

        from psx_predictor.macro.ingestor import MacroRetrospectiveIngestor

        ingestor = MacroRetrospectiveIngestor(paths=self.paths)
        res = self._execute_with_retry(
            lambda: ingestor.ingest_all(dry_run=False),
            step_name="update_macro",
        )
        return f"Synchronized macroeconomic indicators: {len(res.indicators)} series updated"

    def _step_build_features(self, dry_run: bool) -> str:
        """Step 4: Rebuild technical and multi-modal combined feature vectors."""
        symbols = self._target_symbols or [s.symbol for s in self.config.get_enabled_stocks()]
        if dry_run:
            return f"[DRY-RUN] Simulated feature generation for {len(symbols)} stocks"

        from psx_predictor.features.builder import TechnicalFeatureBuilder
        from psx_predictor.features.merger import MultiModalFeatureMerger

        builder = TechnicalFeatureBuilder(storage_paths=self.paths)
        builder.build_universe(symbols=symbols, save=True, with_targets=True)

        merger = MultiModalFeatureMerger(storage_paths=self.paths)
        merge_res = merger.merge_universe(symbols=symbols, save=True)

        return f"Built technical & combined features for {len(merge_res['symbols'])} stocks"

    def _step_generate_predictions(self, dry_run: bool) -> str:
        """Step 5: Run model inference across active universe and record forecasts."""
        symbols = self._target_symbols or [s.symbol for s in self.config.get_enabled_stocks()]
        if dry_run:
            return f"[DRY-RUN] Simulated inference for {len(symbols)} stocks"

        from psx_predictor.predictions.predictor import LivePredictor

        predictor = LivePredictor(storage_paths=self.paths)
        count = 0
        for sym in symbols:
            try:
                predictor.generate_prediction(sym, log_to_registry=True)
                count += 1
            except Exception as exc:
                logger.warning(f"Could not generate prediction for {sym}: {exc}")

        return f"Generated and registered forward predictions for {count}/{len(symbols)} stocks"

    def _step_audit_predictions(self, dry_run: bool) -> str:
        """Step 6: Reconcile yesterday's predictions against today's realized returns."""
        if dry_run:
            return "[DRY-RUN] Simulated prediction outcome reconciliation and Brier scoring"

        from psx_predictor.predictions.auditor import PredictionAuditor

        auditor = PredictionAuditor(storage_paths=self.paths)
        _, summary = auditor.reconcile()
        brier_str = f"{summary.brier_score:.4f}" if summary.brier_score is not None else "N/A"
        return (
            f"Audited {summary.total_records} predictions ({summary.resolved_records} resolved): "
            f"Hit Rate {summary.hit_rate * 100:.1f}%, Brier {brier_str}"
        )

    def run_daily_pipeline(
        self,
        dry_run: bool = False,
        continue_on_error: bool = False,
        symbols: Optional[list[str]] = None,
    ) -> PipelineRunReport:
        """Execute end-to-end daily operational routine in strict topological sequence."""
        if symbols is not None:
            self._target_symbols = symbols
        start_time = datetime.datetime.now(datetime.timezone.utc)
        start_perf = time.perf_counter()
        tasks: list[PipelineTaskResult] = []
        aborted = False

        steps = [
            DailyPipelineStep.UPDATE_PRICES,
            DailyPipelineStep.COLLECT_NEWS,
            DailyPipelineStep.UPDATE_MACRO,
            DailyPipelineStep.BUILD_FEATURES,
            DailyPipelineStep.GENERATE_PREDICTIONS,
            DailyPipelineStep.AUDIT_PREDICTIONS,
        ]

        self.console.print(
            f"Starting Daily EOD Pipeline Orchestrator | "
            f"Mode: {'[magenta]DRY-RUN[/magenta]' if dry_run else '[green]EXECUTE[/green]'}"
        )

        for step in steps:
            if aborted:
                tasks.append(
                    PipelineTaskResult(
                        step=step,
                        status="SKIPPED",
                        duration_seconds=0.0,
                        message="Skipped due to prior step failure.",
                    )
                )
                continue

            result = self.run_step(step, dry_run=dry_run)
            tasks.append(result)

            if result.status == "FAILED" and not continue_on_error:
                aborted = True

        end_time = datetime.datetime.now(datetime.timezone.utc)
        total_duration = time.perf_counter() - start_perf
        overall_success = all(t.status == "SUCCESS" for t in tasks)

        report = PipelineRunReport(
            started_at=start_time.isoformat(),
            ended_at=end_time.isoformat(),
            duration_seconds=total_duration,
            success=overall_success,
            tasks=tasks,
        )

        # Append execution record to pipeline.log
        self._append_log(report)

        # Print visual Rich summary table
        self._display_summary(report)

        return report

    def _append_log(self, report: PipelineRunReport) -> None:
        """Write execution audit record to data/pipeline.log."""
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(report.to_dict()) + "\n")
        except Exception as exc:
            logger.warning(f"Could not append to pipeline log {self.log_file}: {exc}")

    def _display_summary(self, report: PipelineRunReport) -> None:
        """Display formatted execution summary table to console."""
        table = Table(
            title=f"Daily EOD Pipeline Execution Summary (Total: {report.duration_seconds:.2f}s)",
            box=box.ROUNDED,
        )
        table.add_column("Order", justify="center", style="cyan")
        table.add_column("Step Name", style="white")
        table.add_column("Status", justify="center")
        table.add_column("Duration", justify="right", style="yellow")
        table.add_column("Details", style="dim")

        for idx, t in enumerate(report.tasks, 1):
            if t.status == "SUCCESS":
                status_str = "[bold green]SUCCESS[/bold green]"
            elif t.status == "FAILED":
                status_str = "[bold red]FAILED[/bold red]"
            else:
                status_str = "[dim]SKIPPED[/dim]"

            table.add_row(
                str(idx),
                t.step.value,
                status_str,
                f"{t.duration_seconds:.2f}s",
                t.message[:75],
            )

        self.console.print(table)
        if report.success:
            self.console.print(
                "[bold green]Daily EOD pipeline completed successfully![/bold green]"
            )
        else:
            self.console.print("[bold red]Daily EOD pipeline encountered errors.[/bold red]")

    def start_daemon(self, blocking: bool = True) -> Any:
        """Initialize and run the long-running APScheduler process."""
        tz = self.config.settings.timezone
        triggers = create_cron_triggers(timezone=tz)

        scheduler: Any = BlockingScheduler() if blocking else BackgroundScheduler()

        scheduler.add_job(
            self.run_daily_pipeline,
            trigger=triggers[0],
            id="psx_eod_mon_thu",
            name="PSX EOD Pipeline (Mon-Thu 16:00 PKT)",
            replace_existing=True,
        )
        scheduler.add_job(
            self.run_daily_pipeline,
            trigger=triggers[1],
            id="psx_eod_fri",
            name="PSX EOD Pipeline (Fri 17:00 PKT)",
            replace_existing=True,
        )

        self.console.print(
            f"Started PSX Predictor Scheduler Daemon | Timezone: [cyan]{tz}[/cyan]\n"
            f"Scheduled Triggers:\n"
            f"  - Mon–Thu: [yellow]16:00 PKT[/yellow]\n"
            f"  - Fri:     [yellow]17:00 PKT[/yellow]\n"
            f"Press Ctrl+C to terminate."
        )

        if blocking:
            try:
                scheduler.start()
            except (KeyboardInterrupt, SystemExit):
                self.console.print("\n[yellow]Scheduler daemon stopped by user.[/yellow]")
        else:
            scheduler.start()

        return scheduler
