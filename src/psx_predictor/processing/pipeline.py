"""Historical market data bootstrap pipeline and corporate actions adjustment."""

import datetime
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from loguru import logger

from psx_predictor.collectors.base import BaseCollector
from psx_predictor.collectors.composite import CompositeCollector
from psx_predictor.config.loader import load_config
from psx_predictor.storage.parquet_io import write_parquet_atomic
from psx_predictor.storage.paths import ensure_directories, get_storage_paths
from psx_predictor.storage.schemas import PriceDatasetValidator


def compute_continuous_adjusted_close(df: pd.DataFrame) -> pd.Series:
    """Compute continuous backward-adjusted closing prices honoring dividends and splits.

    Uses standard financial backward adjustment:
    For any trading session t with cash dividend D_t announced against close C_{t-1},
    the price adjustment factor for prior periods is:
        factor_div = 1.0 - (D_t / C_{t-1})
    For stock splits with split ratio S_t:
        factor_split = 1.0 / S_t

    Args:
        df: Chronologically sorted OHLCV DataFrame with columns:
            ['trade_date', 'close', 'dividend_amount', 'split_ratio'].

    Returns:
        pd.Series containing the continuous adjusted close price.
    """
    if df.empty or "close" not in df.columns:
        return pd.Series(dtype=float)

    # Ensure chronological sort
    work_df = df.copy()
    if not work_df["trade_date"].is_monotonic_increasing:
        work_df = work_df.sort_values(by="trade_date").reset_index(drop=True)

    n = len(work_df)
    adj_close = work_df["close"].astype(float).values.copy()
    dividends = (
        work_df["dividend_amount"].fillna(0.0).values if "dividend_amount" in work_df else [0.0] * n
    )
    splits = work_df["split_ratio"].fillna(1.0).values if "split_ratio" in work_df else [1.0] * n

    # Backward pass from newest (n-1) to oldest (0)
    cum_factor = 1.0
    for i in range(n - 1, -1, -1):
        # Current day's adjusted close is close * cum_factor
        adj_close[i] = round(adj_close[i] * cum_factor, 4)

        # Update cum_factor for all prior days if day i had corporate actions
        div = float(dividends[i])
        split = float(splits[i])

        if split > 0 and abs(split - 1.0) > 1e-4:
            cum_factor *= 1.0 / split

        if div > 0 and i > 0:
            prev_close = float(work_df["close"].iloc[i - 1])
            if prev_close > 0:
                div_factor = max(1.0 - (div / prev_close), 0.0001)
                cum_factor *= div_factor

    return pd.Series(adj_close, index=df.index, name="adjusted_close")


@dataclass
class SymbolBootstrapResult:
    """Ingestion and validation metrics for a single stock ticker."""

    symbol: str
    status: str  # 'SUCCESS', 'SKIPPED', 'FAILED'
    record_count: int = 0
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    upper_locks: int = 0
    lower_locks: int = 0
    dividend_events: int = 0
    split_events: int = 0
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    file_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UniverseBootstrapResult:
    """Aggregated results across the entire bootstrapped stock universe."""

    total_symbols: int
    successful_symbols: int = 0
    skipped_symbols: int = 0
    failed_symbols: int = 0
    total_records: int = 0
    start_time: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    end_time: Optional[str] = None
    symbol_results: dict[str, SymbolBootstrapResult] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_symbols": self.total_symbols,
            "successful_symbols": self.successful_symbols,
            "skipped_symbols": self.skipped_symbols,
            "failed_symbols": self.failed_symbols,
            "total_records": self.total_records,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "symbol_results": {sym: res.to_dict() for sym, res in self.symbol_results.items()},
        }


class DataBootstrapPipeline:
    """Orchestrator for bulk historical market data ingestion and normalization."""

    def __init__(
        self,
        storage_paths: Optional[dict[str, Path]] = None,
        collector: Optional[BaseCollector] = None,
    ) -> None:
        """Initialize pipeline with storage paths and market data collector.

        Args:
            storage_paths: Storage directory mapping. If None, resolves from app config.
            collector: Data collector instance. If None, uses CompositeCollector.
        """
        if storage_paths is None:
            cfg = load_config()
            self.storage_paths = get_storage_paths(cfg.settings.data_dir)
        else:
            self.storage_paths = storage_paths

        ensure_directories(self.storage_paths["root"])
        self.collector = collector or CompositeCollector()

    def bootstrap_symbol(
        self,
        symbol: str,
        start_date: datetime.date,
        end_date: Optional[datetime.date] = None,
        dry_run: bool = False,
        overwrite: bool = True,
    ) -> SymbolBootstrapResult:
        """Bootstrap historical market data for an individual stock.

        Args:
            symbol: PSX ticker symbol (e.g. 'OGDC').
            start_date: Historical start date.
            end_date: Historical end date (defaults to today).
            dry_run: If True, executes validation without writing Parquet.
            overwrite: If False and processed Parquet exists, skips fetch.

        Returns:
            SymbolBootstrapResult with detailed ingestion metrics.
        """
        import time

        start_ts = time.time()
        clean_sym = symbol.strip().upper()
        target_file = self.storage_paths["processed_prices"] / f"{clean_sym}.parquet"

        # Check skip condition
        if not overwrite and target_file.exists() and not dry_run:
            logger.info(f"[{clean_sym}] Skipping existing dataset at {target_file}")
            return SymbolBootstrapResult(
                symbol=clean_sym,
                status="SKIPPED",
                file_path=str(target_file),
                duration_seconds=round(time.time() - start_ts, 3),
            )

        resolved_end = end_date or datetime.date.today()
        logger.info(
            f"[{clean_sym}] Bootstrapping from {start_date} to {resolved_end} "
            f"via '{self.collector.source_name}'"
        )

        try:
            # 1. Fetch historical raw data
            df = self.collector.fetch_historical(
                symbol=clean_sym,
                start_date=start_date,
                end_date=resolved_end,
                save_raw_payload=not dry_run,
            )

            if df.empty:
                raise ValueError(f"No records retrieved for {clean_sym}")

            # 2. Strict schema validation & deduplication
            validated_df = PriceDatasetValidator.validate_dataframe(df)

            # 3. Compute/verify continuous adjusted close
            if (
                "dividend_amount" in validated_df.columns
                and validated_df["dividend_amount"].sum() > 0
            ) or (
                "split_ratio" in validated_df.columns and (validated_df["split_ratio"] != 1.0).any()
            ):
                validated_df["adjusted_close"] = compute_continuous_adjusted_close(validated_df)

            # 4. Atomically persist to processed storage
            if not dry_run:
                write_parquet_atomic(validated_df, target_file)
                logger.success(
                    f"[{clean_sym}] Stored {len(validated_df)} sessions in {target_file.name}"
                )

            duration = round(time.time() - start_ts, 3)
            min_date = str(validated_df["trade_date"].min())
            max_date = str(validated_df["trade_date"].max())
            upper_locks = int(validated_df["is_upper_lock"].sum())
            lower_locks = int(validated_df["is_lower_lock"].sum())
            div_events = int((validated_df["dividend_amount"] > 0).sum())
            split_events = int((validated_df["split_ratio"] != 1.0).sum())

            return SymbolBootstrapResult(
                symbol=clean_sym,
                status="SUCCESS",
                record_count=len(validated_df),
                start_date=min_date,
                end_date=max_date,
                upper_locks=upper_locks,
                lower_locks=lower_locks,
                dividend_events=div_events,
                split_events=split_events,
                duration_seconds=duration,
                file_path=str(target_file) if not dry_run else None,
            )

        except Exception as e:
            duration = round(time.time() - start_ts, 3)
            logger.error(f"[{clean_sym}] Bootstrap failed: {e}")
            return SymbolBootstrapResult(
                symbol=clean_sym,
                status="FAILED",
                duration_seconds=duration,
                error_message=str(e),
            )

    def bootstrap_universe(
        self,
        symbols: Optional[list[str]] = None,
        years: int = 5,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
        dry_run: bool = False,
        overwrite: bool = True,
    ) -> UniverseBootstrapResult:
        """Bootstrap historical market data across an entire universe of symbols.

        Enforces strict failure isolation: an error in one symbol will not stop
        or corrupt the ingestion of other symbols in the batch.

        Args:
            symbols: Optional list of tickers. If None, loads enabled tickers from stocks.yaml.
            years: Lookback window in years if start_date is not specified.
            start_date: Optional explicit start date.
            end_date: Optional explicit end date (default: today).
            dry_run: If True, validates data without writing files.
            overwrite: If False, skips symbols that already have processed Parquets.

        Returns:
            UniverseBootstrapResult summarizing universe execution.
        """
        resolved_end = end_date or datetime.date.today()
        if start_date:
            resolved_start = start_date
        else:
            # Approximate leap years with 365.25 days/year
            resolved_start = resolved_end - datetime.timedelta(days=int(years * 365.25))

        # Resolve symbol universe
        if symbols is None or not symbols:
            cfg = load_config()
            target_symbols = [s.symbol for s in cfg.get_enabled_stocks()]
        else:
            target_symbols = [s.strip().upper() for s in symbols if s.strip()]

        logger.info(
            f"Starting bootstrap universe run for {len(target_symbols)} symbols: "
            f"{target_symbols} ({resolved_start} to {resolved_end})"
        )

        universe_result = UniverseBootstrapResult(total_symbols=len(target_symbols))

        for idx, sym in enumerate(target_symbols, 1):
            logger.info(f"[{idx}/{len(target_symbols)}] Ingesting {sym}...")
            # Failure isolation: wrap individual symbol bootstrap
            try:
                res = self.bootstrap_symbol(
                    symbol=sym,
                    start_date=resolved_start,
                    end_date=resolved_end,
                    dry_run=dry_run,
                    overwrite=overwrite,
                )
            except Exception as e:
                logger.critical(f"Unhandled exception during bootstrap of {sym}: {e}")
                res = SymbolBootstrapResult(
                    symbol=sym,
                    status="FAILED",
                    error_message=f"Critical unhandled error: {e}",
                )

            universe_result.symbol_results[sym] = res

            if res.status == "SUCCESS":
                universe_result.successful_symbols += 1
                universe_result.total_records += res.record_count
            elif res.status == "SKIPPED":
                universe_result.skipped_symbols += 1
            else:
                universe_result.failed_symbols += 1

        universe_result.end_time = datetime.datetime.now().isoformat()

        # Save run manifest if not dry-run
        if not dry_run:
            manifest_path = self.storage_paths["processed_prices"] / "bootstrap_manifest.json"
            try:
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(universe_result.to_dict(), f, indent=2)
                logger.info(f"Saved bootstrap manifest to {manifest_path}")
            except Exception as e:
                logger.warning(f"Failed to write manifest to {manifest_path}: {e}")

        logger.info(
            f"Bootstrap finished. Succeeded: {universe_result.successful_symbols}, "
            f"Failed: {universe_result.failed_symbols}, Records: {universe_result.total_records}"
        )

        return universe_result
