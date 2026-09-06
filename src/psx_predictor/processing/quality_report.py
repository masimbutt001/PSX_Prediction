"""Market data quality assurance, calendar gap detection, and health auditing."""

import datetime
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from loguru import logger
from rich import box
from rich.table import Table

from psx_predictor.config.loader import load_config
from psx_predictor.storage.parquet_io import DatasetNotFoundError, read_parquet
from psx_predictor.storage.paths import get_storage_paths


@dataclass
class DataGap:
    """Represents an anomalous gap between consecutive trading sessions."""

    start_date: str  # Last session before gap
    end_date: str  # First session after gap
    calendar_days: int
    missing_weekdays: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SymbolQualityReport:
    """Quality and health metrics for an individual stock dataset."""

    symbol: str
    total_sessions: int
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    calendar_span_days: int = 0
    expected_weekdays: int = 0
    coverage_ratio: float = 0.0
    gaps: list[DataGap] = field(default_factory=list)
    upper_locks: int = 0
    lower_locks: int = 0
    dividend_events: int = 0
    total_dividends: float = 0.0
    split_events: int = 0
    zero_volume_sessions: int = 0
    health_status: str = "UNKNOWN"  # 'HEALTHY', 'WARNING', 'CRITICAL'
    health_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "total_sessions": self.total_sessions,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "calendar_span_days": self.calendar_span_days,
            "expected_weekdays": self.expected_weekdays,
            "coverage_ratio": round(self.coverage_ratio, 4),
            "gaps": [g.to_dict() for g in self.gaps],
            "upper_locks": self.upper_locks,
            "lower_locks": self.lower_locks,
            "dividend_events": self.dividend_events,
            "total_dividends": round(self.total_dividends, 2),
            "split_events": self.split_events,
            "zero_volume_sessions": self.zero_volume_sessions,
            "health_status": self.health_status,
            "health_summary": self.health_summary,
        }

    def to_rich_table(self) -> Table:
        """Render a formatted Rich Table summarizing symbol health."""
        table = Table(title=f"Data Quality Report: {self.symbol}", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold white")

        status_color = (
            "green"
            if self.health_status == "HEALTHY"
            else "yellow"
            if self.health_status == "WARNING"
            else "red"
        )
        table.add_row("Health Status", f"[{status_color}]{self.health_status}[/{status_color}]")
        table.add_row("Summary", self.health_summary)
        table.add_row("Total Sessions", str(self.total_sessions))
        table.add_row("Trading Span", f"{self.start_date} to {self.end_date}")
        table.add_row("Coverage Ratio", f"{self.coverage_ratio:.1%}")
        table.add_row("Detected Gaps (> 4 days)", str(len(self.gaps)))
        table.add_row("Upper Lock Sessions (+7.5%)", str(self.upper_locks))
        table.add_row("Lower Lock Sessions (-7.5%)", str(self.lower_locks))
        table.add_row(
            "Dividend Distributions", f"{self.dividend_events} (PKR {self.total_dividends:.2f})"
        )
        table.add_row("Stock Split Events", str(self.split_events))
        table.add_row("Zero Volume Sessions", str(self.zero_volume_sessions))

        return table


@dataclass
class UniverseQualityReport:
    """Aggregated quality report across the entire universe of stocks."""

    total_symbols: int
    healthy_count: int = 0
    warning_count: int = 0
    critical_count: int = 0
    total_sessions: int = 0
    symbol_reports: dict[str, SymbolQualityReport] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_symbols": self.total_symbols,
            "healthy_count": self.healthy_count,
            "warning_count": self.warning_count,
            "critical_count": self.critical_count,
            "total_sessions": self.total_sessions,
            "symbol_reports": {s: r.to_dict() for s, r in self.symbol_reports.items()},
        }

    def to_rich_table(self) -> Table:
        """Render a comparative overview table of the entire universe."""
        table = Table(title="PSX Universe Data Quality Audit", box=box.ROUNDED)
        table.add_column("Symbol", style="bold cyan")
        table.add_column("Status", justify="center")
        table.add_column("Sessions", justify="right")
        table.add_column("Span", justify="center")
        table.add_column("Coverage", justify="right")
        table.add_column("Gaps", justify="right")
        table.add_column("Locks (U/L)", justify="center")
        table.add_column("Dividends", justify="right")

        for sym, r in sorted(self.symbol_reports.items()):
            status_style = (
                "[bold green]HEALTHY[/bold green]"
                if r.health_status == "HEALTHY"
                else "[bold yellow]WARNING[/bold yellow]"
                if r.health_status == "WARNING"
                else "[bold red]CRITICAL[/bold red]"
            )
            span_str = f"{r.start_date} -> {r.end_date}" if r.start_date else "N/A"
            locks_str = f"{r.upper_locks} / {r.lower_locks}"
            div_str = (
                f"{r.dividend_events} (PKR {r.total_dividends:.1f})"
                if r.dividend_events > 0
                else "-"
            )
            table.add_row(
                sym,
                status_style,
                str(r.total_sessions),
                span_str,
                f"{r.coverage_ratio:.1%}",
                str(len(r.gaps)),
                locks_str,
                div_str,
            )

        return table


class DataQualityAuditor:
    """Evaluator for market dataset completeness, anomalies, and health."""

    def __init__(self, storage_paths: Optional[dict[str, Path]] = None) -> None:
        """Initialize auditor with storage paths.

        Args:
            storage_paths: Storage paths dictionary.
        """
        if storage_paths is None:
            cfg = load_config()
            self.storage_paths = get_storage_paths(cfg.settings.data_dir)
        else:
            self.storage_paths = storage_paths

    def audit_dataframe(
        self,
        symbol: str,
        df: pd.DataFrame,
        max_gap_days: int = 4,
    ) -> SymbolQualityReport:
        """Audit an in-memory OHLCV DataFrame for completeness and anomalies.

        Args:
            symbol: Ticker symbol.
            df: Normalized OHLCV DataFrame.
            max_gap_days: Consecutive calendar days threshold to flag as gap (default: 4).

        Returns:
            SymbolQualityReport with detailed metrics.
        """
        clean_sym = symbol.strip().upper()
        if df.empty or "trade_date" not in df.columns:
            return SymbolQualityReport(
                symbol=clean_sym,
                total_sessions=0,
                health_status="CRITICAL",
                health_summary="Dataset is empty or missing trade_date column",
            )

        # Sort chronologically
        work_df = df.sort_values(by="trade_date").reset_index(drop=True)
        dates: list[datetime.date] = [
            d
            if isinstance(d, datetime.date) and not isinstance(d, datetime.datetime)
            else d.date()
            if hasattr(d, "date")
            else datetime.date.fromisoformat(str(d)[:10])
            for d in work_df["trade_date"]
        ]

        start_date = dates[0]
        end_date = dates[-1]
        calendar_span = (end_date - start_date).days + 1

        # Calculate expected weekdays
        expected_weekdays = sum(
            1
            for d in (start_date + datetime.timedelta(n) for n in range(calendar_span))
            if d.weekday() < 5
        )
        total_sessions = len(dates)
        # PSX has ~15-20 public holidays per year; expected coverage vs weekdays is ~90-95%
        coverage_ratio = total_sessions / max(expected_weekdays, 1)

        # Detect anomalous calendar gaps (> max_gap_days)
        gaps: list[DataGap] = []
        for i in range(1, len(dates)):
            gap_days = (dates[i] - dates[i - 1]).days
            if gap_days > max_gap_days:
                # Count missing weekdays in the gap
                gap_start = dates[i - 1]
                gap_end = dates[i]
                missing_weekdays = sum(
                    1
                    for d in (gap_start + datetime.timedelta(n) for n in range(1, gap_days))
                    if d.weekday() < 5
                )
                # Only flag as gap if there are at least 3 missing weekdays
                # (excluding standard holidays and long weekends)
                if missing_weekdays >= 3:
                    gaps.append(
                        DataGap(
                            start_date=str(gap_start),
                            end_date=str(gap_end),
                            calendar_days=gap_days,
                            missing_weekdays=missing_weekdays,
                        )
                    )

        upper_locks = int(work_df["is_upper_lock"].sum()) if "is_upper_lock" in work_df else 0
        lower_locks = int(work_df["is_lower_lock"].sum()) if "is_lower_lock" in work_df else 0

        div_events = 0
        total_divs = 0.0
        if "dividend_amount" in work_df:
            div_events = int((work_df["dividend_amount"] > 0).sum())
            total_divs = float(work_df["dividend_amount"].sum())

        split_events = 0
        if "split_ratio" in work_df:
            split_events = int((work_df["split_ratio"] != 1.0).sum())

        zero_vol = 0
        if "volume" in work_df:
            zero_vol = int((work_df["volume"] == 0).sum())

        # Determine health status
        critical_gaps = [g for g in gaps if g.missing_weekdays >= 10]
        if total_sessions < 10 or len(critical_gaps) >= 2 or coverage_ratio < 0.60:
            health = "CRITICAL"
            summary = (
                f"Severe data gaps ({len(critical_gaps)} critical gaps >= 10 weekdays) "
                f"or low coverage ({coverage_ratio:.1%})."
            )
        elif len(gaps) > 0 or coverage_ratio < 0.85:
            health = "WARNING"
            summary = (
                f"Detected {len(gaps)} potential gap(s) totaling "
                f"{sum(g.missing_weekdays for g in gaps)} missing weekdays."
            )
        else:
            health = "HEALTHY"
            summary = "Continuous daily price history with expected holiday distribution."

        return SymbolQualityReport(
            symbol=clean_sym,
            total_sessions=total_sessions,
            start_date=str(start_date),
            end_date=str(end_date),
            calendar_span_days=calendar_span,
            expected_weekdays=expected_weekdays,
            coverage_ratio=coverage_ratio,
            gaps=gaps,
            upper_locks=upper_locks,
            lower_locks=lower_locks,
            dividend_events=div_events,
            total_dividends=total_divs,
            split_events=split_events,
            zero_volume_sessions=zero_vol,
            health_status=health,
            health_summary=summary,
        )

    def audit_symbol(
        self,
        symbol: str,
        max_gap_days: int = 4,
    ) -> SymbolQualityReport:
        """Audit a symbol from the processed Parquet storage directory.

        Args:
            symbol: Ticker symbol.
            max_gap_days: Threshold for anomalous gap detection.

        Returns:
            SymbolQualityReport for the stored dataset.
        """
        clean_sym = symbol.strip().upper()
        file_path = self.storage_paths["processed_prices"] / f"{clean_sym}.parquet"

        try:
            df = read_parquet(file_path)
            return self.audit_dataframe(clean_sym, df, max_gap_days=max_gap_days)
        except DatasetNotFoundError:
            return SymbolQualityReport(
                symbol=clean_sym,
                total_sessions=0,
                health_status="CRITICAL",
                health_summary=f"Processed Parquet file not found at: {file_path}",
            )
        except Exception as e:
            return SymbolQualityReport(
                symbol=clean_sym,
                total_sessions=0,
                health_status="CRITICAL",
                health_summary=f"Error reading dataset: {e}",
            )

    def audit_universe(
        self,
        symbols: Optional[list[str]] = None,
        max_gap_days: int = 4,
    ) -> UniverseQualityReport:
        """Audit an entire universe of symbols from processed Parquet storage.

        Args:
            symbols: Optional list of symbols. If None, audits all existing .parquet files.
            max_gap_days: Threshold for anomalous gap detection.

        Returns:
            UniverseQualityReport containing all audited symbols.
        """
        processed_dir = self.storage_paths["processed_prices"]

        if symbols is None or not symbols:
            parquet_files = list(processed_dir.glob("*.parquet"))
            target_symbols = [f.stem for f in parquet_files]
        else:
            target_symbols = [s.strip().upper() for s in symbols if s.strip()]

        target_symbols.sort()
        logger.info(f"Auditing universe of {len(target_symbols)} symbols: {target_symbols}")

        universe_report = UniverseQualityReport(total_symbols=len(target_symbols))

        for sym in target_symbols:
            report = self.audit_symbol(sym, max_gap_days=max_gap_days)
            universe_report.symbol_reports[sym] = report
            universe_report.total_sessions += report.total_sessions

            if report.health_status == "HEALTHY":
                universe_report.healthy_count += 1
            elif report.health_status == "WARNING":
                universe_report.warning_count += 1
            else:
                universe_report.critical_count += 1

        return universe_report
