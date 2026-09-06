"""Composite multi-source collector orchestrating provider fallback and gap filling."""

import datetime
from typing import Any, Optional

import pandas as pd
from loguru import logger

from psx_predictor.collectors.base import (
    BaseCollector,
    CollectorError,
    calculate_circuit_locks,
)
from psx_predictor.collectors.dps_collector import DPSCollector
from psx_predictor.collectors.scs_collector import SCSTradeCollector
from psx_predictor.collectors.yahoo_collector import YahooCollector
from psx_predictor.storage.schemas import PriceDatasetValidator


class CompositeCollector(BaseCollector):
    """Orchestrates multi-source historical ingestion with automatic failover and gap backfill."""

    def __init__(
        self,
        collectors: Optional[list[BaseCollector]] = None,
    ) -> None:
        """Initialize composite collector with prioritized list of providers.

        Default priority:
        1. SCSTradeCollector (scstrade.com)
        2. YahooCollector (Yahoo Finance .KA)
        3. DPSCollector (PSX Data Portal)
        """
        self.collectors = collectors or [
            SCSTradeCollector(),
            YahooCollector(),
            DPSCollector(),
        ]

    @property
    def source_name(self) -> str:
        return "composite"

    def fetch_raw(
        self,
        symbol: str,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> Any:
        """Composite collector handles fetch_historical directly across sub-collectors."""
        raise NotImplementedError("CompositeCollector orchestrates via fetch_historical()")

    def normalize(self, raw_data: Any, symbol: str) -> pd.DataFrame:
        """Composite collector normalizes via underlying sub-collectors."""
        raise NotImplementedError("CompositeCollector normalizes via underlying sub-collectors")

    def fetch_historical(
        self,
        symbol: str,
        start_date: datetime.date,
        end_date: Optional[datetime.date] = None,
        save_raw_payload: bool = True,
    ) -> pd.DataFrame:
        """Fetch historical data across prioritized providers with automatic failover.

        Args:
            symbol: PSX Ticker Symbol (e.g. 'OGDC').
            start_date: Earliest trading date.
            end_date: Latest trading date (defaults to today).
            save_raw_payload: Whether sub-collectors save raw payloads.

        Returns:
            Clean, validated, deduplicated, chronologically sorted DataFrame.
        """
        end_date = end_date or datetime.date.today()
        if start_date > end_date:
            raise ValueError(
                f"start_date ({start_date}) cannot be later than end_date ({end_date})"
            )

        errors: list[str] = []
        collected_df: Optional[pd.DataFrame] = None

        for collector in self.collectors:
            try:
                logger.info(f"[composite] Trying fetch for {symbol} via '{collector.source_name}'")
                df = collector.fetch_historical(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    save_raw_payload=save_raw_payload,
                )

                if df is not None and not df.empty:
                    collected_df = df
                    logger.info(
                        f"[composite] Successfully retrieved {len(df)} sessions for {symbol} "
                        f"using '{collector.source_name}'"
                    )
                    break

            except Exception as e:
                err_msg = f"[{collector.source_name}] failed for {symbol}: {e}"
                logger.warning(f"[composite] {err_msg}")
                errors.append(err_msg)

        if collected_df is None or collected_df.empty:
            raise CollectorError(
                f"All providers failed for symbol '{symbol}'. Summary: " + "; ".join(errors)
            )

        # Check for date gaps if expected date range is substantial (> 10 days)
        # Attempt backfill for missing dates if secondary collectors exist
        backfilled_df = self._backfill_gaps(
            df=collected_df,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            save_raw=save_raw_payload,
        )

        with_locks = calculate_circuit_locks(backfilled_df)
        return PriceDatasetValidator.validate_dataframe(with_locks)

    def _backfill_gaps(
        self,
        df: pd.DataFrame,
        symbol: str,
        start_date: datetime.date,
        end_date: datetime.date,
        save_raw: bool,
    ) -> pd.DataFrame:
        """Detect date boundaries and query secondary providers to fill missing bounds."""
        min_date = df["trade_date"].min()
        max_date = df["trade_date"].max()

        delta_start = (min_date - start_date).days
        delta_end = (end_date - max_date).days

        # If data is missing > 5 days from requested start or end
        if delta_start > 5 or delta_end > 5:
            logger.info(
                f"[composite] Data bounds ({min_date} to {max_date}) narrower than requested "
                f"({start_date} to {end_date}). Attempting gap resolution..."
            )
            for secondary in self.collectors:
                try:
                    sec_df = secondary.fetch_historical(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        save_raw_payload=save_raw,
                    )
                    if not sec_df.empty:
                        # Combine and deduplicate
                        merged = pd.concat([df, sec_df], ignore_index=True)
                        merged = merged.drop_duplicates(
                            subset=["symbol", "trade_date"], keep="first"
                        )
                        df = merged.sort_values(by="trade_date").reset_index(drop=True)
                        logger.info(
                            f"[composite] Merged {len(df)} total sessions after backfill from "
                            f"'{secondary.source_name}'"
                        )
                        break
                except Exception:
                    continue

        return df
