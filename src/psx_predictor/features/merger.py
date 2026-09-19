"""Point-in-time multi-modal feature merger combining technical, news, and macroeconomic series."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from loguru import logger

from psx_predictor.config.loader import load_config
from psx_predictor.features.builder import TechnicalFeatureBuilder
from psx_predictor.macro.storage import MacroStorage
from psx_predictor.storage.parquet_io import (
    DatasetNotFoundError,
    read_parquet,
    write_parquet_atomic,
)
from psx_predictor.storage.paths import ensure_directories

NEWS_FEATURE_COLUMNS: list[str] = [
    "sentiment_24h",
    "sentiment_72h",
    "news_count_24h",
    "news_count_72h",
    "positive_count_24h",
    "negative_count_24h",
    "has_earnings_announcement_today",
    "has_dividend_announcement_today",
    "has_discovery_announcement_today",
]

MACRO_FEATURE_COLUMNS: list[str] = [
    "usd_pkr",
    "usd_pkr_return_1d",
    "sbp_policy_rate",
    "kibor_6m",
    "cpi_yoy",
    "brent_crude",
    "brent_return_1d",
    "kse100_index",
    "kse100_return_1d",
]

INTERACTION_FEATURE_COLUMNS: list[str] = [
    "rel_kse100_ret_1d",
    "policy_rate_spread",
    "real_interest_rate",
]


@dataclass
class MergeResult:
    """Outcome metrics for multi-modal feature merging on an individual ticker."""

    symbol: str
    status: str  # 'SUCCESS' or 'FAILED'
    total_records: int = 0
    total_features: int = 0
    tech_features_count: int = 0
    news_features_count: int = 0
    macro_features_count: int = 0
    interaction_features_count: int = 0
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    file_path: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MultiModalFeatureMerger:
    """Merges price technical indicators, market-aligned news sentiment, and macro series."""

    def __init__(self, storage_paths: Optional[dict[str, Path]] = None) -> None:
        """Initialize merger with resolved storage paths."""
        if storage_paths is None:
            config = load_config()
            self.storage_paths = ensure_directories(config.settings.data_dir)
        else:
            self.storage_paths = storage_paths

        ensure_directories(self.storage_paths["root"])
        self.tech_builder = TechnicalFeatureBuilder(self.storage_paths)
        self.macro_storage = MacroStorage(storage_paths=self.storage_paths)

    def merge_symbol(self, symbol: str, save: bool = True) -> pd.DataFrame:
        """Merge all data modalities for a single ticker with zero look-ahead bias.

        Args:
            symbol: Ticker symbol (e.g. 'OGDC').
            save: Whether to persist to 'data/features/combined/{symbol}_combined.parquet'.

        Returns:
            Merged point-in-time DataFrame.
        """
        clean_sym = symbol.strip().upper()
        logger.info(f"[{clean_sym}] Starting multi-modal point-in-time feature merge...")

        # 1. Load or build technical features with supervised targets
        tech_path = (
            self.storage_paths["features_technical"] / f"{clean_sym}_tech_features.parquet"
        )
        if not tech_path.exists():
            logger.info(f"[{clean_sym}] Technical features not found at {tech_path}. Building...")
            df_tech = self.tech_builder.build_for_symbol(clean_sym, save=True, with_targets=True)
        else:
            df_tech = read_parquet(tech_path)

        if df_tech.empty:
            raise ValueError(f"Technical feature dataset for '{clean_sym}' is empty.")

        merged = df_tech.copy()

        # Ensure session_date column formatted as ISO YYYY-MM-DD
        if "trade_date" in merged.columns:
            merged["session_date"] = pd.to_datetime(merged["trade_date"]).dt.strftime("%Y-%m-%d")
        elif "date" in merged.columns:
            merged["session_date"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d")
            merged["trade_date"] = pd.to_datetime(merged["date"]).dt.date

        # 2. Load aligned news sentiment features
        news_df = self._load_news_features(clean_sym)
        if not news_df.empty:
            # Drop symbol column from news before merging to avoid collisions
            cols_to_merge = [c for c in news_df.columns if c != "symbol"]
            merged = pd.merge(
                merged,
                news_df[cols_to_merge],
                on="session_date",
                how="left",
            )
        else:
            # Populate empty news columns
            for col in NEWS_FEATURE_COLUMNS:
                merged[col] = 0.0

        # Fill missing news entries with neutral defaults
        for col in NEWS_FEATURE_COLUMNS:
            if col not in merged.columns:
                merged[col] = 0.0
            else:
                merged[col] = merged[col].fillna(0.0)

        # 3. Load processed daily macroeconomic features
        macro_df = self._load_macro_features()
        if not macro_df.empty:
            merged = self._merge_macro_retrospective(merged, macro_df)
        else:
            logger.warning(f"[{clean_sym}] Macro features unavailable. Filling defaults.")
            for col in MACRO_FEATURE_COLUMNS:
                merged[col] = 0.0

        # 4. Compute cross-modal interaction & relative features
        merged = self._compute_interaction_features(merged)

        # 5. Persist merged dataset atomically
        if save:
            dest_dir = self.storage_paths["features_combined"]
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / f"{clean_sym}_combined.parquet"
            write_parquet_atomic(merged, dest_file)
            logger.success(
                f"[{clean_sym}] Atomically persisted {len(merged)} combined rows "
                f"({len(merged.columns)} cols) to {dest_file}"
            )

        return merged

    def _load_news_features(self, symbol: str) -> pd.DataFrame:
        """Load session-aligned news features for the symbol."""
        news_file = self.storage_paths["features_news"] / f"{symbol}.parquet"
        if news_file.exists():
            df = read_parquet(news_file)
            if not df.empty and "session_date" in df.columns:
                df["session_date"] = pd.to_datetime(df["session_date"]).dt.strftime("%Y-%m-%d")
                return df

        # Fallback to collective news features file
        collective_file = self.storage_paths["features_news"] / "daily_news_features.parquet"
        if collective_file.exists():
            df = read_parquet(collective_file)
            if not df.empty and "symbol" in df.columns and "session_date" in df.columns:
                sym_df = df[df["symbol"].str.upper() == symbol.upper()].copy()
                sym_df["session_date"] = pd.to_datetime(sym_df["session_date"]).dt.strftime(
                    "%Y-%m-%d"
                )
                return sym_df

        return pd.DataFrame()

    def _load_macro_features(self) -> pd.DataFrame:
        """Load processed daily macro features, building if missing."""
        macro_file = self.storage_paths["processed_macro"] / "macro_daily.parquet"
        if not macro_file.exists():
            logger.info("Daily macro features missing. Updating macro storage...")
            try:
                self.macro_storage.update()
            except Exception as e:
                logger.error(f"Failed to automatically update macro storage: {e}")
                return pd.DataFrame()

        if macro_file.exists():
            df = read_parquet(macro_file)
            if not df.empty and "session_date" in df.columns:
                df["session_date"] = pd.to_datetime(df["session_date"]).dt.strftime("%Y-%m-%d")
                return df
        return pd.DataFrame()

    def _merge_macro_retrospective(
        self, target_df: pd.DataFrame, macro_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge macro series retrospectively with zero look-ahead leakage."""
        t_df = target_df.copy()
        m_df = macro_df.copy()

        t_df["_join_dt"] = pd.to_datetime(t_df["session_date"])
        m_df["_join_dt"] = pd.to_datetime(m_df["session_date"])

        t_df = t_df.sort_values("_join_dt").reset_index(drop=True)
        m_df = m_df.sort_values("_join_dt").reset_index(drop=True)

        # Retrospective backward merge: matches latest macro_release <= session_date
        merged = pd.merge_asof(
            t_df,
            m_df.drop(columns=["session_date"]),
            on="_join_dt",
            direction="backward",
        )

        merged = merged.drop(columns=["_join_dt"])

        # Forward fill and backward fill any edge missing macro values
        for col in MACRO_FEATURE_COLUMNS:
            if col in merged.columns:
                merged[col] = merged[col].ffill().bfill().fillna(0.0)

        return merged

    def _compute_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute relative market return and interest rate spread features."""
        result = df.copy()

        # 1. Relative return vs. KSE-100 benchmark (Asset return minus index return)
        if "log_ret_1d" in result.columns and "kse100_return_1d" in result.columns:
            result["rel_kse100_ret_1d"] = result["log_ret_1d"] - result["kse100_return_1d"]
        else:
            result["rel_kse100_ret_1d"] = 0.0

        # 2. Policy rate spread (KIBOR minus SBP Policy Rate)
        if "kibor_6m" in result.columns and "sbp_policy_rate" in result.columns:
            result["policy_rate_spread"] = result["kibor_6m"] - result["sbp_policy_rate"]
        else:
            result["policy_rate_spread"] = 0.0

        # 3. Real interest rate (SBP Policy Rate minus CPI inflation)
        if "sbp_policy_rate" in result.columns and "cpi_yoy" in result.columns:
            result["real_interest_rate"] = result["sbp_policy_rate"] - result["cpi_yoy"]
        else:
            result["real_interest_rate"] = 0.0

        return result

    def merge_universe(
        self, symbols: Optional[list[str]] = None, save: bool = True
    ) -> dict[str, Any]:
        """Merge multi-modal features across the specified universe.

        Args:
            symbols: Optional list of ticker symbols. If None, scans processed_prices.
            save: Whether to persist each merged dataset.

        Returns:
            Dictionary containing execution statistics and individual ticker results.
        """
        if symbols is None:
            price_files = list(self.storage_paths["processed_prices"].glob("*.parquet"))
            clean_symbols = [f.stem.upper() for f in price_files]
        else:
            clean_symbols = [s.strip().upper() for s in symbols]

        logger.info(f"Beginning multi-modal merge across {len(clean_symbols)} symbols...")
        results: dict[str, MergeResult] = {}
        total_records = 0

        for sym in clean_symbols:
            try:
                merged_df = self.merge_symbol(sym, save=save)
                rec_count = len(merged_df)
                total_records += rec_count

                # Count features per modality
                news_count = sum(1 for c in NEWS_FEATURE_COLUMNS if c in merged_df.columns)
                macro_count = sum(1 for c in MACRO_FEATURE_COLUMNS if c in merged_df.columns)
                inter_count = sum(
                    1 for c in INTERACTION_FEATURE_COLUMNS if c in merged_df.columns
                )
                tech_count = len(merged_df.columns) - (news_count + macro_count + inter_count)

                start_d = str(merged_df["session_date"].iloc[0]) if not merged_df.empty else None
                end_d = str(merged_df["session_date"].iloc[-1]) if not merged_df.empty else None
                dest_file = (
                    str(self.storage_paths["features_combined"] / f"{sym}_combined.parquet")
                    if save
                    else None
                )

                results[sym] = MergeResult(
                    symbol=sym,
                    status="SUCCESS",
                    total_records=rec_count,
                    total_features=len(merged_df.columns),
                    tech_features_count=tech_count,
                    news_features_count=news_count,
                    macro_features_count=macro_count,
                    interaction_features_count=inter_count,
                    start_date=start_d,
                    end_date=end_d,
                    file_path=dest_file,
                )
            except DatasetNotFoundError as e:
                logger.warning(f"[{sym}] Skipped: {e}")
                results[sym] = MergeResult(symbol=sym, status="SKIPPED", error_message=str(e))
            except Exception as e:
                logger.error(f"[{sym}] Merge failed: {e}")
                results[sym] = MergeResult(symbol=sym, status="FAILED", error_message=str(e))

        success_count = sum(1 for r in results.values() if r.status == "SUCCESS")
        return {
            "total_requested": len(clean_symbols),
            "successful": success_count,
            "failed": len(clean_symbols) - success_count,
            "total_records": total_records,
            "symbols": {k: v.to_dict() for k, v in results.items()},
        }
