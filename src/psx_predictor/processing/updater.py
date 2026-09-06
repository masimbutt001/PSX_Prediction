"""Idempotent incremental daily market data updater and state manager."""

import datetime
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from loguru import logger

from psx_predictor.collectors.base import BaseCollector
from psx_predictor.collectors.composite import CompositeCollector
from psx_predictor.config.loader import load_config
from psx_predictor.processing.pipeline import (
    DataBootstrapPipeline,
    compute_continuous_adjusted_close,
)
from psx_predictor.storage.parquet_io import (
    DatasetNotFoundError,
    read_parquet,
    write_parquet_atomic,
)
from psx_predictor.storage.paths import ensure_directories, get_storage_paths
from psx_predictor.storage.schemas import PriceDatasetValidator


@dataclass
class SymbolUpdateResult:
    """Detailed update metrics for an individual stock ticker."""

    symbol: str
    status: str  # 'UPDATED', 'ALREADY_UP_TO_DATE', 'BOOTSTRAPPED', 'FAILED'
    previous_latest_date: Optional[str] = None
    new_latest_date: Optional[str] = None
    records_added: int = 0
    total_records: int = 0
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    file_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UniverseUpdateResult:
    """Aggregated results across the updated stock universe."""

    total_symbols: int
    updated_symbols: int = 0
    up_to_date_symbols: int = 0
    bootstrapped_symbols: int = 0
    failed_symbols: int = 0
    total_records_added: int = 0
    start_time: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    end_time: Optional[str] = None
    symbol_results: dict[str, SymbolUpdateResult] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_symbols": self.total_symbols,
            "updated_symbols": self.updated_symbols,
            "up_to_date_symbols": self.up_to_date_symbols,
            "bootstrapped_symbols": self.bootstrapped_symbols,
            "failed_symbols": self.failed_symbols,
            "total_records_added": self.total_records_added,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "symbol_results": {s: r.to_dict() for s, r in self.symbol_results.items()},
        }


class IncrementalUpdater:
    """Synchronizes local market datasets by fetching only new incremental sessions."""

    def __init__(
        self,
        storage_paths: Optional[dict[str, Path]] = None,
        collector: Optional[BaseCollector] = None,
    ) -> None:
        """Initialize updater with storage paths and market data collector.

        Args:
            storage_paths: Storage directory mapping. Defaults to app config paths.
            collector: Data collector instance. Defaults to CompositeCollector.
        """
        if storage_paths is None:
            cfg = load_config()
            self.storage_paths = get_storage_paths(cfg.settings.data_dir)
        else:
            self.storage_paths = storage_paths

        ensure_directories(self.storage_paths["root"])
        self.collector = collector or CompositeCollector()

    def get_latest_trade_date(self, symbol: str) -> Optional[datetime.date]:
        """Query the latest existing local trading date for a symbol.

        Args:
            symbol: Ticker symbol (e.g. 'OGDC').

        Returns:
            Latest trading date in the dataset, or None if dataset does not exist.
        """
        clean_sym = symbol.strip().upper()
        target_file = self.storage_paths["processed_prices"] / f"{clean_sym}.parquet"

        if not target_file.exists():
            return None

        try:
            # Read only the trade_date column for performance
            df = read_parquet(target_file, columns=["trade_date"])
            if df.empty or "trade_date" not in df.columns:
                return None

            max_val = df["trade_date"].max()
            if isinstance(max_val, datetime.date) and not isinstance(max_val, datetime.datetime):
                return max_val
            if hasattr(max_val, "date"):
                return max_val.date()
            return datetime.date.fromisoformat(str(max_val)[:10])
        except Exception as e:
            logger.warning(f"[{clean_sym}] Failed to read latest date: {e}")
            return None

    def update_symbol(
        self,
        symbol: str,
        end_date: Optional[datetime.date] = None,
        dry_run: bool = False,
        force: bool = False,
        auto_bootstrap: bool = True,
    ) -> SymbolUpdateResult:
        """Incrementally update a single stock ticker with new trading sessions.

        Args:
            symbol: PSX ticker symbol.
            end_date: Upper boundary date for update (defaults to today).
            dry_run: If True, executes logic without modifying Parquet on disk.
            force: If True, queries provider even if latest date matches today.
            auto_bootstrap: If True, bootstraps dataset if no local file exists.

        Returns:
            SymbolUpdateResult with detailed operation metrics.
        """
        start_ts = time.time()
        clean_sym = symbol.strip().upper()
        target_file = self.storage_paths["processed_prices"] / f"{clean_sym}.parquet"
        resolved_end = end_date or datetime.date.today()

        # 1. Handle missing local dataset
        if not target_file.exists():
            if auto_bootstrap:
                logger.info(f"[{clean_sym}] No local dataset found. Auto-bootstrapping...")
                pipeline = DataBootstrapPipeline(
                    storage_paths=self.storage_paths,
                    collector=self.collector,
                )
                boot_res = pipeline.bootstrap_symbol(
                    symbol=clean_sym,
                    start_date=resolved_end - datetime.timedelta(days=int(5 * 365.25)),
                    end_date=resolved_end,
                    dry_run=dry_run,
                )
                duration = round(time.time() - start_ts, 3)
                if boot_res.status == "SUCCESS":
                    return SymbolUpdateResult(
                        symbol=clean_sym,
                        status="BOOTSTRAPPED",
                        previous_latest_date=None,
                        new_latest_date=boot_res.end_date,
                        records_added=boot_res.record_count,
                        total_records=boot_res.record_count,
                        duration_seconds=duration,
                        file_path=str(target_file) if not dry_run else None,
                    )
                return SymbolUpdateResult(
                    symbol=clean_sym,
                    status="FAILED",
                    duration_seconds=duration,
                    error_message=f"Auto-bootstrap failed: {boot_res.error_message}",
                )
            return SymbolUpdateResult(
                symbol=clean_sym,
                status="FAILED",
                duration_seconds=round(time.time() - start_ts, 3),
                error_message=f"Dataset not found at {target_file} and auto_bootstrap=False",
            )

        # 2. Read existing dataset & resolve T_max
        try:
            existing_df = read_parquet(target_file)
        except DatasetNotFoundError:
            return SymbolUpdateResult(
                symbol=clean_sym,
                status="FAILED",
                duration_seconds=round(time.time() - start_ts, 3),
                error_message=f"Dataset missing at {target_file}",
            )

        latest_date = self.get_latest_trade_date(clean_sym)
        if latest_date is None:
            return SymbolUpdateResult(
                symbol=clean_sym,
                status="FAILED",
                duration_seconds=round(time.time() - start_ts, 3),
                error_message=f"Unable to resolve latest trade_date for {clean_sym}",
            )

        fetch_start = latest_date + datetime.timedelta(days=1)
        prev_latest_str = str(latest_date)

        # 3. Check if already up-to-date
        if fetch_start > resolved_end and not force:
            logger.info(f"[{clean_sym}] Already up to date (latest: {prev_latest_str})")
            return SymbolUpdateResult(
                symbol=clean_sym,
                status="ALREADY_UP_TO_DATE",
                previous_latest_date=prev_latest_str,
                new_latest_date=prev_latest_str,
                records_added=0,
                total_records=len(existing_df),
                duration_seconds=round(time.time() - start_ts, 3),
                file_path=str(target_file),
            )

        logger.info(
            f"[{clean_sym}] Querying incremental delta from {fetch_start} to {resolved_end} "
            f"via '{self.collector.source_name}'"
        )

        try:
            # 4. Fetch delta
            delta_df = self.collector.fetch_historical(
                symbol=clean_sym,
                start_date=fetch_start,
                end_date=resolved_end,
                save_raw_payload=not dry_run,
            )

            # If no sessions occurred between fetch_start and resolved_end (e.g. weekend / holiday)
            if delta_df.empty:
                logger.info(f"[{clean_sym}] Provider returned 0 new records. Up to date.")
                return SymbolUpdateResult(
                    symbol=clean_sym,
                    status="ALREADY_UP_TO_DATE",
                    previous_latest_date=prev_latest_str,
                    new_latest_date=prev_latest_str,
                    records_added=0,
                    total_records=len(existing_df),
                    duration_seconds=round(time.time() - start_ts, 3),
                    file_path=str(target_file),
                )

            # 5. Validate new delta records
            validated_delta = PriceDatasetValidator.validate_dataframe(delta_df)

            # 6. Merge, deduplicate on (symbol, trade_date), and sort
            merged_df = pd.concat([existing_df, validated_delta], ignore_index=True)
            deduped_df = merged_df.drop_duplicates(subset=["symbol", "trade_date"], keep="last")
            sorted_df = deduped_df.sort_values(by="trade_date").reset_index(drop=True)

            # 7. Retroactive continuous adjusted close recomputation
            # If corporate action occurs in delta (or historical), re-run adjustment
            has_dividends = (
                "dividend_amount" in sorted_df.columns and (sorted_df["dividend_amount"] > 0).any()
            )
            has_splits = (
                "split_ratio" in sorted_df.columns and (sorted_df["split_ratio"] != 1.0).any()
            )
            if has_dividends or has_splits:
                sorted_df["adjusted_close"] = compute_continuous_adjusted_close(sorted_df)

            records_added = len(sorted_df) - len(existing_df)
            new_latest_str = str(sorted_df["trade_date"].max())

            # 8. Persist atomically
            if not dry_run and records_added > 0:
                write_parquet_atomic(sorted_df, target_file)
                logger.success(
                    f"[{clean_sym}] Appended {records_added} new records (total: {len(sorted_df)})"
                )

            duration = round(time.time() - start_ts, 3)
            status_out = "UPDATED" if records_added > 0 else "ALREADY_UP_TO_DATE"

            return SymbolUpdateResult(
                symbol=clean_sym,
                status=status_out,
                previous_latest_date=prev_latest_str,
                new_latest_date=new_latest_str,
                records_added=records_added,
                total_records=len(sorted_df),
                duration_seconds=duration,
                file_path=str(target_file) if not dry_run else None,
            )

        except Exception as e:
            duration = round(time.time() - start_ts, 3)
            logger.error(f"[{clean_sym}] Incremental update failed: {e}")
            return SymbolUpdateResult(
                symbol=clean_sym,
                status="FAILED",
                previous_latest_date=prev_latest_str,
                duration_seconds=duration,
                error_message=str(e),
            )

    def update_universe(
        self,
        symbols: Optional[list[str]] = None,
        end_date: Optional[datetime.date] = None,
        dry_run: bool = False,
        force: bool = False,
        auto_bootstrap: bool = True,
    ) -> UniverseUpdateResult:
        """Incrementally update the entire stock universe with failure isolation.

        Args:
            symbols: Optional list of tickers. Defaults to all enabled stocks in stocks.yaml.
            end_date: Upper boundary date for update (defaults to today).
            dry_run: If True, runs without disk modification.
            force: If True, forces provider query even if up to date.
            auto_bootstrap: If True, bootstraps any missing tickers.

        Returns:
            UniverseUpdateResult summarizing universe update run.
        """
        # Resolve symbols
        if symbols is None or not symbols:
            cfg = load_config()
            target_symbols = [s.symbol for s in cfg.get_enabled_stocks()]
        else:
            target_symbols = [s.strip().upper() for s in symbols if s.strip()]

        logger.info(f"Starting incremental update run for {len(target_symbols)} symbols")

        universe_result = UniverseUpdateResult(total_symbols=len(target_symbols))

        for idx, sym in enumerate(target_symbols, 1):
            logger.info(f"[{idx}/{len(target_symbols)}] Updating {sym}...")
            # Failure isolation
            try:
                res = self.update_symbol(
                    symbol=sym,
                    end_date=end_date,
                    dry_run=dry_run,
                    force=force,
                    auto_bootstrap=auto_bootstrap,
                )
            except Exception as e:
                logger.critical(f"Unhandled exception during incremental update of {sym}: {e}")
                res = SymbolUpdateResult(
                    symbol=sym,
                    status="FAILED",
                    error_message=f"Critical unhandled error: {e}",
                )

            universe_result.symbol_results[sym] = res

            if res.status == "UPDATED":
                universe_result.updated_symbols += 1
                universe_result.total_records_added += res.records_added
            elif res.status == "ALREADY_UP_TO_DATE":
                universe_result.up_to_date_symbols += 1
            elif res.status == "BOOTSTRAPPED":
                universe_result.bootstrapped_symbols += 1
                universe_result.total_records_added += res.records_added
            else:
                universe_result.failed_symbols += 1

        universe_result.end_time = datetime.datetime.now().isoformat()

        # Save update manifest if not dry-run
        if not dry_run:
            manifest_path = self.storage_paths["processed_prices"] / "update_manifest.json"
            try:
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(universe_result.to_dict(), f, indent=2)
                logger.info(f"Saved update manifest to {manifest_path}")
            except Exception as e:
                logger.warning(f"Failed to write update manifest: {e}")

        logger.info(
            f"Universe update finished. Updated: {universe_result.updated_symbols}, "
            f"Up to date: {universe_result.up_to_date_symbols}, "
            f"Bootstrapped: {universe_result.bootstrapped_symbols}, "
            f"Failed: {universe_result.failed_symbols}, "
            f"Records added: {universe_result.total_records_added}"
        )

        return universe_result
