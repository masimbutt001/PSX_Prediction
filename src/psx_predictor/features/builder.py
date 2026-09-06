"""Feature builder orchestrator for transforming processed market data into feature datasets."""

import datetime
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from loguru import logger

from psx_predictor.config.loader import load_config
from psx_predictor.features.targets import compute_all_prediction_targets
from psx_predictor.features.technical import compute_all_technical_features
from psx_predictor.storage.parquet_io import (
    DatasetNotFoundError,
    read_parquet,
    write_parquet_atomic,
)
from psx_predictor.storage.paths import ensure_directories, get_storage_paths


@dataclass
class FeatureBuildResult:
    """Outcome metrics for feature generation on an individual ticker."""

    symbol: str
    status: str  # 'SUCCESS', 'FAILED'
    total_records: int = 0
    feature_count: int = 0
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    file_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureUniverseResult:
    """Aggregated outcome across all built feature sets."""

    total_symbols: int
    successful_symbols: int = 0
    failed_symbols: int = 0
    total_records: int = 0
    start_time: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    end_time: Optional[str] = None
    symbol_results: dict[str, FeatureBuildResult] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_symbols": self.total_symbols,
            "successful_symbols": self.successful_symbols,
            "failed_symbols": self.failed_symbols,
            "total_records": self.total_records,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "symbol_results": {s: r.to_dict() for s, r in self.symbol_results.items()},
        }


class TechnicalFeatureBuilder:
    """Transforms processed historical OHLCV data into engineered technical feature stores."""

    def __init__(self, storage_paths: Optional[dict[str, Path]] = None) -> None:
        """Initialize builder with analytical storage paths.

        Args:
            storage_paths: Storage paths dictionary. Defaults to configured paths.
        """
        if storage_paths is None:
            cfg = load_config()
            self.storage_paths = get_storage_paths(cfg.settings.data_dir)
        else:
            self.storage_paths = storage_paths

        ensure_directories(self.storage_paths["root"])

    def build_for_symbol(
        self,
        symbol: str,
        save: bool = True,
        with_targets: bool = True,
        threshold: float = 0.0075,
        target_horizon: int = 5,
    ) -> pd.DataFrame:
        """Compute technical features and optional targets for a single stock.

        Args:
            symbol: Ticker symbol (e.g. 'OGDC').
            save: Whether to atomically serialize to Parquet (default: True).
            with_targets: Whether to append supervised prediction targets (default: True).
            threshold: 3-class breakout return threshold (default: 0.0075).
            target_horizon: Multi-session return horizon (default: 5).

        Returns:
            DataFrame enriched with technical features and targets.

        Raises:
            DatasetNotFoundError: If processed price dataset does not exist.
        """
        clean_sym = symbol.strip().upper()
        source_file = self.storage_paths["processed_prices"] / f"{clean_sym}.parquet"

        if not source_file.exists():
            raise DatasetNotFoundError(
                f"Processed price dataset not found for '{clean_sym}' at {source_file}. "
                "Run 'psx data bootstrap' or 'psx data fetch' first."
            )

        logger.info(f"[{clean_sym}] Loading processed data from {source_file}...")
        df = read_parquet(source_file)

        logger.info(
            f"[{clean_sym}] Computing vectorized technical features for {len(df)} sessions..."
        )
        features_df = compute_all_technical_features(df)

        if with_targets:
            logger.info(f"[{clean_sym}] Computing supervised prediction targets...")
            targets_df = compute_all_prediction_targets(
                df, threshold=threshold, horizon=target_horizon
            )
            features_df = pd.concat([features_df, targets_df], axis=1)

        if save:
            dest_file = (
                self.storage_paths["features_technical"] / f"{clean_sym}_tech_features.parquet"
            )
            write_parquet_atomic(features_df, dest_file)
            logger.success(f"[{clean_sym}] Saved {len(features_df)} records to {dest_file}")

        return features_df

    def build_universe(
        self,
        symbols: Optional[list[str]] = None,
        save: bool = True,
        with_targets: bool = True,
        threshold: float = 0.0075,
        target_horizon: int = 5,
    ) -> FeatureUniverseResult:
        """Generate technical feature sets across the specified or processed universe.

        Args:
            symbols: Optional list of tickers. If None, discovers all *.parquet in processed_prices.
            save: Whether to save features to Parquet (default: True).
            with_targets: Whether to append supervised prediction targets (default: True).
            threshold: 3-class breakout return threshold (default: 0.0075).
            target_horizon: Multi-session return horizon (default: 5).

        Returns:
            FeatureUniverseResult summarizing feature generation metrics.
        """
        # Resolve target symbols
        if symbols is None or not symbols:
            processed_dir = self.storage_paths["processed_prices"]
            parquet_files = list(processed_dir.glob("*.parquet"))
            target_symbols = [f.stem for f in parquet_files]
        else:
            target_symbols = [s.strip().upper() for s in symbols if s.strip()]

        target_symbols.sort()
        logger.info(f"Starting technical feature generation for {len(target_symbols)} symbols")

        universe_result = FeatureUniverseResult(total_symbols=len(target_symbols))

        for idx, sym in enumerate(target_symbols, 1):
            start_ts = time.time()
            logger.info(f"[{idx}/{len(target_symbols)}] Generating features for {sym}...")

            # Strict failure isolation per ticker
            try:
                features_df = self.build_for_symbol(
                    sym,
                    save=save,
                    with_targets=with_targets,
                    threshold=threshold,
                    target_horizon=target_horizon,
                )
                duration = round(time.time() - start_ts, 3)

                start_date = (
                    str(features_df["trade_date"].min()) if "trade_date" in features_df else None
                )
                end_date = (
                    str(features_df["trade_date"].max()) if "trade_date" in features_df else None
                )
                dest_file = (
                    self.storage_paths["features_technical"] / f"{sym}_tech_features.parquet"
                )

                res = FeatureBuildResult(
                    symbol=sym,
                    status="SUCCESS",
                    total_records=len(features_df),
                    feature_count=len(features_df.columns),
                    start_date=start_date,
                    end_date=end_date,
                    duration_seconds=duration,
                    file_path=str(dest_file) if save else None,
                )
                universe_result.successful_symbols += 1
                universe_result.total_records += len(features_df)

            except Exception as e:
                duration = round(time.time() - start_ts, 3)
                logger.error(f"[{sym}] Feature generation failed: {e}")
                res = FeatureBuildResult(
                    symbol=sym,
                    status="FAILED",
                    duration_seconds=duration,
                    error_message=str(e),
                )
                universe_result.failed_symbols += 1

            universe_result.symbol_results[sym] = res

        universe_result.end_time = datetime.datetime.now().isoformat()

        # Save manifest
        if save:
            manifest_path = self.storage_paths["features_technical"] / "feature_manifest.json"
            try:
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(universe_result.to_dict(), f, indent=2)
                logger.info(f"Saved feature manifest to {manifest_path}")
            except Exception as e:
                logger.warning(f"Failed to write feature manifest: {e}")

        logger.info(
            f"Feature universe build completed. Successful: {universe_result.successful_symbols}, "
            f"Failed: {universe_result.failed_symbols}, Records: {universe_result.total_records}"
        )

        return universe_result
