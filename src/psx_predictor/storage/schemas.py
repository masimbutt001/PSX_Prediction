"""Pydantic schemas and validation rules for market OHLCV price datasets."""

import datetime
from typing import Any

import pandas as pd
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class OHLCVRecord(BaseModel):
    """Normalized schema for a single daily OHLCV trading session record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(..., description="Normalized uppercase PSX ticker symbol")
    trade_date: datetime.date = Field(..., description="Trading session date")
    open: float = Field(..., gt=0.0, description="Session opening price in PKR")
    high: float = Field(..., gt=0.0, description="Session highest executed price in PKR")
    low: float = Field(..., gt=0.0, description="Session lowest executed price in PKR")
    close: float = Field(..., gt=0.0, description="Session unadjusted closing price in PKR")
    adjusted_close: float = Field(
        ..., gt=0.0, description="Dividend and split-adjusted closing price"
    )
    volume: int = Field(..., ge=0, description="Total traded volume (shares)")
    dividend_amount: float = Field(
        default=0.0, ge=0.0, description="Cash dividend announced on this ex-date"
    )
    split_ratio: float = Field(
        default=1.0, gt=0.0, description="Stock split ratio (e.g. 2.0 for 2-for-1)"
    )
    is_upper_lock: bool = Field(
        default=False, description="Whether session hit the upper circuit breaker (+7.5% / 1 PKR)"
    )
    is_lower_lock: bool = Field(
        default=False, description="Whether session hit the lower circuit breaker (-7.5% / 1 PKR)"
    )

    @field_validator("symbol", mode="before")
    @classmethod
    def validate_symbol(cls, value: Any) -> str:
        """Normalize ticker symbol to uppercase."""
        if not isinstance(value, str):
            raise ValueError("Ticker symbol must be a string")
        cleaned = value.strip().upper()
        if not cleaned:
            raise ValueError("Ticker symbol cannot be empty")
        if not cleaned.isalnum():
            raise ValueError(f"Ticker symbol must be alphanumeric, got: {value}")
        return cleaned

    @field_validator("trade_date", mode="before")
    @classmethod
    def validate_trade_date(cls, value: Any) -> datetime.date:
        """Convert string or pandas Timestamp to standard date."""
        if isinstance(value, datetime.date) and not isinstance(value, datetime.datetime):
            return value
        if isinstance(value, (datetime.datetime, pd.Timestamp)):
            return value.date()
        if isinstance(value, str):
            return datetime.date.fromisoformat(value.strip())
        raise ValueError(f"Unsupported trade_date format: {value}")

    @model_validator(mode="after")
    def validate_price_invariants(self) -> "OHLCVRecord":
        """Verify mathematical integrity of OHLC price boundaries."""
        # Allow small floating point tolerance (1e-6)
        eps = 1e-6
        if self.high < (self.low - eps):
            raise ValueError(f"High price ({self.high}) cannot be lower than Low ({self.low})")
        if self.high < (self.open - eps):
            raise ValueError(f"High price ({self.high}) cannot be lower than Open ({self.open})")
        if self.high < (self.close - eps):
            raise ValueError(f"High price ({self.high}) cannot be lower than Close ({self.close})")
        if self.low > (self.open + eps):
            raise ValueError(f"Low price ({self.low}) cannot be higher than Open ({self.open})")
        if self.low > (self.close + eps):
            raise ValueError(f"Low price ({self.low}) cannot be higher than Close ({self.close})")
        return self


class PriceDatasetValidator:
    """Batch validator and sanitizer for pandas OHLCV datasets."""

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Validate, deduplicate, and sort an OHLCV dataframe.

        Args:
            df: Raw input DataFrame with required OHLCV columns.

        Returns:
            Cleaned, validated, and chronologically sorted DataFrame.
        """
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "trade_date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "adjusted_close",
                    "volume",
                    "dividend_amount",
                    "split_ratio",
                    "is_upper_lock",
                    "is_lower_lock",
                ]
            )

        records: list[dict[str, Any]] = []
        raw_rows = df.to_dict(orient="records")

        for idx, row in enumerate(raw_rows):
            try:
                rec = OHLCVRecord(**row)
                records.append(rec.model_dump())
            except Exception as e:
                logger.warning(f"Row {idx} rejected due to validation failure: {e}")

        if not records:
            raise ValueError("All rows failed OHLCV schema validation")

        validated_df = pd.DataFrame(records)

        # Deduplicate on (symbol, trade_date), keeping the last entry
        deduped_df = validated_df.drop_duplicates(subset=["symbol", "trade_date"], keep="last")

        # Sort chronologically
        sorted_df = deduped_df.sort_values(by=["symbol", "trade_date"]).reset_index(drop=True)
        return sorted_df
