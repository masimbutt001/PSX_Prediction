"""Prediction audit engine reconciling past model forecasts against realized market closes."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from psx_predictor.config.loader import load_config
from psx_predictor.predictions.registry import PredictionRegistry
from psx_predictor.storage.parquet_io import read_parquet
from psx_predictor.storage.paths import ensure_directories


@dataclass(frozen=True)
class AuditSummary:
    """Summary statistics from a prediction reconciliation audit."""

    total_records: int
    resolved_records: int
    pending_records: int
    hit_rate: float
    brier_score: Optional[float]
    symbol_filter: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "resolved_records": self.resolved_records,
            "pending_records": self.pending_records,
            "hit_rate": round(self.hit_rate, 4),
            "brier_score": round(self.brier_score, 4) if self.brier_score is not None else None,
            "symbol_filter": self.symbol_filter,
        }


class PredictionAuditor:
    """Audits and reconciles pending prediction records against realized market outcomes."""

    def __init__(self, storage_paths: Optional[dict[str, Path]] = None) -> None:
        if storage_paths is None:
            config = load_config()
            self.storage_paths = ensure_directories(config.settings.data_dir)
        else:
            self.storage_paths = storage_paths

        self.registry = PredictionRegistry(storage_paths=self.storage_paths)

    def reconcile(self, symbol: Optional[str] = None) -> tuple[pd.DataFrame, AuditSummary]:
        """Reconcile unverified forecasts with actual market closing prices.

        Args:
            symbol: Optional ticker filter.

        Returns:
            Tuple of (all_predictions_dataframe, AuditSummary).
        """
        all_df = self.registry.get_predictions(symbol=symbol)
        if all_df.empty:
            logger.info("Prediction registry is empty. No predictions to audit.")
            return all_df, AuditSummary(
                total_records=0,
                resolved_records=0,
                pending_records=0,
                hit_rate=0.0,
                brier_score=None,
                symbol_filter=symbol,
            )

        updated_rows = 0
        symbols_to_check = all_df["symbol"].unique()

        for sym in symbols_to_check:
            proc_file = self.storage_paths["processed_prices"] / f"{sym}.parquet"
            if not proc_file.exists():
                logger.warning(f"[{sym}] Processed price file not found: {proc_file}")
                continue

            price_df = read_parquet(proc_file)
            if price_df.empty:
                continue

            # Ensure trade_date is string YYYY-MM-DD
            price_df["date_str"] = price_df["trade_date"].astype(str)
            price_df = price_df.sort_values("trade_date").reset_index(drop=True)

            # Match unresolved records for this symbol
            mask = (all_df["symbol"] == sym) & (all_df["realized_outcome"].isna())
            unresolved_indices = all_df[mask].index

            for idx in unresolved_indices:
                tgt_date = str(all_df.loc[idx, "target_date"])

                # Check if target date exists in realized market sessions
                match_indices = price_df[price_df["date_str"] == tgt_date].index
                if len(match_indices) > 0:
                    target_row_idx = match_indices[0]
                    if target_row_idx > 0:
                        prev_close = float(price_df.loc[target_row_idx - 1, "close"])
                        curr_close = float(price_df.loc[target_row_idx, "close"])
                        realized_return = (curr_close - prev_close) / prev_close

                        # Check prediction correctness
                        signal = str(all_df.loc[idx, "signal"]).upper()
                        is_correct: bool
                        if signal == "BUY":
                            is_correct = bool(realized_return > 0)
                        elif signal == "SELL":
                            is_correct = bool(realized_return <= 0)
                        else:  # HOLD / NEUTRAL
                            is_correct = bool(abs(realized_return) <= 0.0075)

                        all_df.loc[idx, "realized_outcome"] = round(realized_return, 6)
                        all_df.loc[idx, "is_correct"] = is_correct
                        updated_rows += 1

        if updated_rows > 0:
            logger.info(f"Reconciled {updated_rows} newly verified prediction outcomes.")
            self.registry.update_predictions(all_df)

        # Compute audit statistics
        resolved_mask = all_df["realized_outcome"].notna()
        resolved_df = all_df[resolved_mask]
        n_resolved = len(resolved_df)
        n_pending = len(all_df) - n_resolved

        if n_resolved > 0:
            hit_rate = float(resolved_df["is_correct"].mean())

            # Brier score calculation: mean squared error of probability vs outcome
            # Outcome = 1 if realized_outcome > 0 else 0
            binary_outcomes = np.where(resolved_df["realized_outcome"] > 0, 1.0, 0.0)
            probs = resolved_df["up_probability"].to_numpy()
            brier_score = float(np.mean((probs - binary_outcomes) ** 2))
        else:
            hit_rate = 0.0
            brier_score = None

        summary = AuditSummary(
            total_records=len(all_df),
            resolved_records=n_resolved,
            pending_records=n_pending,
            hit_rate=hit_rate,
            brier_score=brier_score,
            symbol_filter=symbol,
        )

        return all_df, summary


def display_audit_report(
    summary: AuditSummary,
    predictions_df: pd.DataFrame,
    console: Optional[Console] = None,
) -> None:
    """Print an aesthetic Rich summary table of audited market predictions."""
    c = console or Console()

    sym_str = f" ({summary.symbol_filter})" if summary.symbol_filter else ""
    title = f"PSX Live Prediction Registry Audit{sym_str}"

    summary_table = Table(
        box=box.ROUNDED,
        title=title,
        header_style="bold cyan",
        show_lines=True,
    )
    summary_table.add_column("Metric", style="bold white")
    summary_table.add_column("Value", style="bold green", justify="right")
    summary_table.add_column("Benchmark / Interpretation", style="dim", justify="left")

    summary_table.add_row(
        "Total Registered Forecasts",
        str(summary.total_records),
        "Cumulative logged predictions across universe",
    )
    summary_table.add_row(
        "Reconciled Forecasts",
        str(summary.resolved_records),
        "Historical sessions whose closing prices have occurred",
    )
    summary_table.add_row(
        "Pending / Unclosed Forecasts",
        str(summary.pending_records),
        "Active forecasts awaiting upcoming market close",
    )

    hit_color = "green" if summary.hit_rate >= 0.50 else "yellow"
    summary_table.add_row(
        "Directional Hit Rate",
        f"[{hit_color}]{summary.hit_rate:.1%}[/{hit_color}]",
        "Accuracy of directional signal vs realized outcome (Baseline: 50.0%)",
    )

    brier_str = f"{summary.brier_score:.4f}" if summary.brier_score is not None else "N/A"
    summary_table.add_row(
        "Brier Calibration Score",
        brier_str,
        "Mean squared probability error (0.00 = perfect, 0.25 = uninformative)",
    )

    c.print()
    c.print(summary_table)

    # Display recent 10 prediction rows
    if not predictions_df.empty:
        records_table = Table(
            box=box.SIMPLE_HEAVY,
            title="Recent Prediction Registry Log (Last 10 Records)",
            header_style="bold magenta",
        )
        records_table.add_column("Symbol", style="bold cyan")
        records_table.add_column("Target Date", justify="center")
        records_table.add_column("Model", style="white")
        records_table.add_column("P(Up)", justify="right")
        records_table.add_column("Signal", justify="center")
        records_table.add_column("Realized Return", justify="right")
        records_table.add_column("Outcome", justify="center")

        recent_df = predictions_df.tail(10)
        for _, row in recent_df.iterrows():
            p_up = f"{row['up_probability']:.1%}"
            sig = str(row["signal"])
            sig_styled = (
                f"[bold green]{sig}[/bold green]"
                if sig == "BUY"
                else (f"[bold red]{sig}[/bold red]" if sig == "SELL" else f"[dim]{sig}[/dim]")
            )

            ret = row.get("realized_outcome")
            if pd.isna(ret) or ret is None:
                ret_str = "[dim]Pending[/dim]"
                res_str = "[dim]PENDING[/dim]"
            else:
                ret_val = float(ret)
                ret_color = "green" if ret_val > 0 else "red"
                ret_str = f"[{ret_color}]{ret_val:+.2%}[/{ret_color}]"
                is_corr = bool(row.get("is_correct", False))
                res_str = (
                    "[bold green]CORRECT[/bold green]" if is_corr else "[bold red]WRONG[/bold red]"
                )

            records_table.add_row(
                str(row["symbol"]),
                str(row["target_date"]),
                str(row["model_name"]),
                p_up,
                sig_styled,
                ret_str,
                res_str,
            )

        c.print()
        c.print(records_table)

    c.print(
        Panel(
            "[dim]Audit Logic: Realized outcome is calculated as "
            "(Close_t - Close_{t-1}) / Close_{t-1}. "
            "Forecasts are registered before session open and reconciled after market close.[/dim]",
            border_style="dim",
        )
    )
