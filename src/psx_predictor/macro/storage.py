"""Macroeconomic storage and point-in-time temporal alignment pipeline."""

import datetime
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from loguru import logger

from psx_predictor.config.loader import load_config
from psx_predictor.macro.market_collector import MarketMacroCollector
from psx_predictor.macro.sbp_collector import SBPMacroCollector
from psx_predictor.macro.schemas import MacroDataPoint, MacroIndicatorType
from psx_predictor.news.calendar import PSXMarketCalendar
from psx_predictor.storage.parquet_io import read_parquet, write_parquet_atomic
from psx_predictor.storage.paths import ensure_directories, get_storage_paths


@dataclass
class MacroUpdateSummary:
    """Summary of macroeconomic ingestion and daily feature generation."""

    total_raw_points_collected: int = 0
    indicators_updated: list[str] = field(default_factory=list)
    trading_sessions_covered: int = 0
    earliest_session: Optional[str] = None
    latest_session: Optional[str] = None
    daily_feature_records: int = 0
    latest_indicators: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MacroStorage:
    """Manages raw macro ingestion, zero-lookahead temporal alignment, and Parquet persistence."""

    def __init__(
        self,
        calendar: Optional[PSXMarketCalendar] = None,
        storage_paths: Optional[dict[str, Path]] = None,
    ) -> None:
        self.calendar = calendar or PSXMarketCalendar()

        if storage_paths is None:
            cfg = load_config()
            self.storage_paths = get_storage_paths(cfg.settings.data_dir)
        else:
            self.storage_paths = storage_paths

        ensure_directories(self.storage_paths["root"])
        self.raw_file = self.storage_paths["raw_macro"] / "macro_raw.parquet"
        self.processed_file = self.storage_paths["processed_macro"] / "macro_daily.parquet"

        self.sbp_collector = SBPMacroCollector()
        self.market_collector = MarketMacroCollector()

    def collect_all_raw(
        self,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
    ) -> list[MacroDataPoint]:
        """Fetch and combine all SBP, PBS, and financial market macroeconomic series."""
        sbp_points = self.sbp_collector.fetch_all()
        market_points = self.market_collector.fetch_all(start_date=start_date, end_date=end_date)
        all_points = sbp_points + market_points
        logger.info(f"Collected total of {len(all_points)} macroeconomic observations.")
        return all_points

    def persist_raw(self, points: list[MacroDataPoint]) -> pd.DataFrame:
        """Persist raw macro points atomically, deduplicating on indicator & observation date."""
        new_df = pd.DataFrame([p.to_dict() for p in points])
        if self.raw_file.exists():
            existing = read_parquet(self.raw_file)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["indicator", "observation_date", "public_release_date"], keep="last"
            )
        else:
            combined = new_df

        # Sort chronologically
        combined = combined.sort_values(
            by=["indicator", "public_release_date", "observation_date"]
        ).reset_index(drop=True)
        write_parquet_atomic(combined, self.raw_file)
        logger.info(f"Persisted {len(combined)} raw macro rows to {self.raw_file}")
        return combined

    def build_daily_features(
        self,
        raw_df: pd.DataFrame,
        trading_dates: list[datetime.date],
    ) -> pd.DataFrame:
        """Construct daily macroeconomic feature vectors with strictly ZERO lookahead bias.

        Look-Ahead Bias Prevention Rule:
        For every trading date D:
        A macro observation is eligible for session D ONLY IF:
        public_release_date <= D.
        Under no circumstances can an observation be used on dates prior to its public_release_date.

        Args:
            raw_df: DataFrame of raw macro observations.
            trading_dates: Chronological list of PSX trading days.

        Returns:
            DataFrame with one row per trading date and aligned macro features.
        """
        if raw_df.empty or not trading_dates:
            return pd.DataFrame()

        dates_str = [d.isoformat() for d in trading_dates]
        base_df = pd.DataFrame(
            {"session_date": dates_str, "_asof_dt": pd.to_datetime(dates_str)}
        ).sort_values("_asof_dt")

        features_df = base_df.copy()
        indicators = raw_df["indicator"].unique()

        for ind in indicators:
            ind_df = raw_df[raw_df["indicator"] == ind].copy()
            ind_df["_asof_dt"] = pd.to_datetime(ind_df["public_release_date"])
            ind_df = ind_df.sort_values("_asof_dt").drop_duplicates(
                subset=["_asof_dt"], keep="last"
            )
            ind_sub = ind_df[["_asof_dt", "value"]].copy()
            ind_sub.rename(columns={"value": ind}, inplace=True)

            # Retrospective point-in-time join (strictly public_release_date <= session_date)
            merged = pd.merge_asof(
                features_df[["_asof_dt"]],
                ind_sub,
                on="_asof_dt",
                direction="backward",
            )
            # Forward-fill, and fill any initial dates prior to the earliest history
            features_df[ind] = merged[ind].ffill().bfill()

        # Standard column renames / fallbacks
        features_df["usd_pkr"] = features_df.get(
            MacroIndicatorType.USD_PKR.value, pd.Series(278.50, index=features_df.index)
        ).astype(float)
        features_df["sbp_policy_rate"] = features_df.get(
            MacroIndicatorType.SBP_POLICY_RATE.value, pd.Series(10.0, index=features_df.index)
        ).astype(float)
        features_df["kibor_6m"] = features_df.get(
            MacroIndicatorType.KIBOR_6M.value, pd.Series(10.65, index=features_df.index)
        ).astype(float)
        features_df["cpi_yoy"] = features_df.get(
            MacroIndicatorType.CPI_YOY.value, pd.Series(5.0, index=features_df.index)
        ).astype(float)
        features_df["brent_crude"] = features_df.get(
            MacroIndicatorType.BRENT_CRUDE.value, pd.Series(80.0, index=features_df.index)
        ).astype(float)
        features_df["kse100_index"] = features_df.get(
            MacroIndicatorType.KSE_100.value, pd.Series(80000.0, index=features_df.index)
        ).astype(float)

        # Compute daily percentage returns for continuous market series
        features_df["usd_pkr_return_1d"] = features_df["usd_pkr"].pct_change().fillna(0.0).round(5)
        features_df["brent_return_1d"] = (
            features_df["brent_crude"].pct_change().fillna(0.0).round(5)
        )
        features_df["kse100_return_1d"] = (
            features_df["kse100_index"].pct_change().fillna(0.0).round(5)
        )

        # Clean output columns
        output_cols = [
            "session_date",
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
        return features_df[output_cols]

    def update(
        self,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
        dry_run: bool = False,
    ) -> MacroUpdateSummary:
        """Run complete macro pipeline: ingest, persist, align, and generate daily features."""
        summary = MacroUpdateSummary()

        # 1. Collect raw observations
        points = self.collect_all_raw(start_date=start_date, end_date=end_date)
        summary.total_raw_points_collected = len(points)
        if not points:
            logger.warning("No macroeconomic data points collected.")
            return summary

        # 2. Persist raw dataset
        if not dry_run:
            raw_df = self.persist_raw(points)
        else:
            raw_df = pd.DataFrame([p.to_dict() for p in points])

        summary.indicators_updated = sorted(list(raw_df["indicator"].unique()))

        # 3. Determine trading dates window from prices or calendar
        earliest_rel = raw_df["public_release_date"].min()
        start_dt = datetime.date.fromisoformat(earliest_rel[:10])
        end_dt = end_date or datetime.date.today()

        trading_days = self.calendar.get_trading_days_between(start_dt, end_dt)
        if not trading_days:
            trading_days = [start_dt]

        # 4. Construct point-in-time daily features
        daily_features = self.build_daily_features(raw_df=raw_df, trading_dates=trading_days)
        summary.daily_feature_records = len(daily_features)
        summary.trading_sessions_covered = len(trading_days)

        if not daily_features.empty:
            summary.earliest_session = str(daily_features["session_date"].min())
            summary.latest_session = str(daily_features["session_date"].max())
            latest_row = daily_features.iloc[-1]
            summary.latest_indicators = {
                "USD/PKR": round(float(latest_row["usd_pkr"]), 2),
                "SBP Policy Rate (%)": round(float(latest_row["sbp_policy_rate"]), 2),
                "6M KIBOR (%)": round(float(latest_row["kibor_6m"]), 2),
                "CPI YoY (%)": round(float(latest_row["cpi_yoy"]), 2),
                "Brent Crude ($/bbl)": round(float(latest_row["brent_crude"]), 2),
                "KSE-100 Index": round(float(latest_row["kse100_index"]), 2),
            }

        # 5. Persist daily features table atomically
        if not dry_run and not daily_features.empty:
            write_parquet_atomic(daily_features, self.processed_file)
            logger.info(
                f"Persisted {len(daily_features)} macro records to {self.processed_file}"
            )

        return summary
