"""Abstract base class and exceptions for all market data collectors."""

import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from loguru import logger

from psx_predictor.config.loader import load_config
from psx_predictor.storage.paths import get_storage_paths
from psx_predictor.storage.schemas import PriceDatasetValidator


class CollectorError(Exception):
    """Base exception for all collector-related errors."""


class SymbolNotFoundError(CollectorError):
    """Raised when an external provider does not recognize the requested symbol."""


class RateLimitError(CollectorError):
    """Raised when an external provider throttles requests."""


class NetworkTimeoutError(CollectorError):
    """Raised when a request to an external provider times out."""


class EmptyResponseError(CollectorError):
    """Raised when an external provider returns an empty payload."""


def calculate_circuit_locks(df: pd.DataFrame) -> pd.DataFrame:
    """Compute PSX circuit breaker upper and lower price locks.

    On the Pakistan Stock Exchange, stocks have a daily circuit limit of +/-7.5%
    (or 1 PKR, whichever is higher). A stock is considered 'locked' when it moves
    by the limit threshold and trades virtually flat across High, Low, and Close.

    Args:
        df: DataFrame sorted chronologically with columns 'close', 'high', 'low'.

    Returns:
        DataFrame with 'is_upper_lock' and 'is_lower_lock' boolean columns populated.
    """
    if df.empty or "close" not in df.columns:
        return df

    out = df.copy()
    if "is_upper_lock" not in out.columns:
        out["is_upper_lock"] = False
    if "is_lower_lock" not in out.columns:
        out["is_lower_lock"] = False

    if len(out) < 2:
        return out

    # Calculate previous session close
    prev_close = out["close"].shift(1)

    # 7.4% threshold accounts for rounding and tick sizes near 7.5%
    pct_change = (out["close"] - prev_close) / prev_close
    abs_change = out["close"] - prev_close

    eps = 1e-4
    is_flat_range = (out["high"] - out["low"]).abs() < (out["close"] * 0.005 + eps)

    upper_lock_pct = (pct_change >= 0.074) & is_flat_range
    upper_lock_pkr = (abs_change >= 0.99) & is_flat_range & (out["close"] <= 15.0)

    lower_lock_pct = (pct_change <= -0.074) & is_flat_range
    lower_lock_pkr = (abs_change <= -0.99) & is_flat_range & (out["close"] <= 15.0)

    out["is_upper_lock"] = (upper_lock_pct | upper_lock_pkr).fillna(False).astype(bool)
    out["is_lower_lock"] = (lower_lock_pct | lower_lock_pkr).fillna(False).astype(bool)

    return out


class BaseCollector(ABC):
    """Abstract interface defining the contract for all price and market data collectors."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Unique identifier of this collector data source."""

    @abstractmethod
    def fetch_raw(
        self,
        symbol: str,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> Any:
        """Fetch raw unparsed payload from external provider."""

    @abstractmethod
    def normalize(self, raw_data: Any, symbol: str) -> pd.DataFrame:
        """Normalize raw provider data into a DataFrame conforming to OHLCVRecord schema."""

    def save_raw(
        self,
        raw_data: Any,
        symbol: str,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Persist immutable raw download to data/raw/prices/.

        Args:
            raw_data: Raw provider payload (JSON string, dict, or bytes).
            symbol: Ticker symbol.
            output_dir: Optional target directory. Defaults to standard raw_prices path.

        Returns:
            Path to saved raw file.
        """
        if output_dir is None:
            cfg = load_config()
            paths = get_storage_paths(cfg.settings.data_dir)
            output_dir = paths["raw_prices"]

        output_dir.mkdir(parents=True, exist_ok=True)
        now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol.upper()}_{self.source_name}_{now_str}.json"
        dest_path = output_dir / filename

        # Support json string or serializable objects
        import json

        with open(dest_path, "w", encoding="utf-8") as f:
            if isinstance(raw_data, str):
                f.write(raw_data)
            else:
                json.dump(raw_data, f, indent=2, default=str)

        logger.debug(f"[{self.source_name}] Saved raw data for {symbol} to {dest_path}")
        return dest_path

    def fetch_historical(
        self,
        symbol: str,
        start_date: datetime.date,
        end_date: Optional[datetime.date] = None,
        save_raw_payload: bool = True,
    ) -> pd.DataFrame:
        """Orchestrate fetch, raw persistence, normalization, and validation.

        Args:
            symbol: Ticker symbol (e.g. 'OGDC').
            start_date: Earliest trading date to fetch.
            end_date: Latest trading date to fetch (defaults to today).
            save_raw_payload: Whether to save immutable raw download.

        Returns:
            Clean, validated, deduplicated, chronologically sorted DataFrame.
        """
        end_date = end_date or datetime.date.today()
        if start_date > end_date:
            raise ValueError(
                f"start_date ({start_date}) cannot be later than end_date ({end_date})"
            )

        logger.info(
            f"[{self.source_name}] Fetching historical data for {symbol} "
            f"from {start_date} to {end_date}"
        )

        raw_data = self.fetch_raw(symbol=symbol, start_date=start_date, end_date=end_date)
        if raw_data is None:
            raise EmptyResponseError(
                f"No data returned from {self.source_name} for symbol {symbol}"
            )

        if save_raw_payload:
            try:
                self.save_raw(raw_data=raw_data, symbol=symbol)
            except Exception as e:
                logger.warning(f"[{self.source_name}] Failed to save raw data: {e}")

        normalized_df = self.normalize(raw_data=raw_data, symbol=symbol)
        if normalized_df.empty:
            raise EmptyResponseError(
                f"Normalized dataset for {symbol} from {self.source_name} is empty"
            )

        # Apply circuit breaker detection
        with_locks_df = calculate_circuit_locks(normalized_df)

        # Validate with PriceDatasetValidator
        validated_df = PriceDatasetValidator.validate_dataframe(with_locks_df)
        logger.info(
            f"[{self.source_name}] Successfully retrieved {len(validated_df)} "
            f"validated sessions for {symbol}"
        )
        return validated_df
