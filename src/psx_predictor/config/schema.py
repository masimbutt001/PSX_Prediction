"""Pydantic schemas for PSX Predictor application configuration and stock universe."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StockConfig(BaseModel):
    """Configuration model representing a single PSX listed stock."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(..., description="PSX stock ticker symbol (e.g. OGDC)")
    name: str = Field(..., description="Full registered company name")
    sector: str = Field(..., description="PSX industry sector")
    yahoo_ticker: str = Field(
        ..., description="Corresponding ticker on Yahoo Finance (e.g. OGDC.KA)"
    )
    enabled: bool = Field(
        default=True, description="Whether this stock is active for tracking/modeling"
    )

    @field_validator("symbol", mode="before")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        """Validate and normalize ticker symbol to uppercase."""
        if not isinstance(value, str):
            raise ValueError("Stock symbol must be a string")
        cleaned = value.strip().upper()
        if not cleaned:
            raise ValueError("Stock symbol cannot be empty")
        if not cleaned.isalnum():
            raise ValueError(f"Stock symbol must be alphanumeric, got: {value}")
        return cleaned


class SettingsConfig(BaseModel):
    """Application-wide settings including directory paths and logging level."""

    model_config = ConfigDict(extra="ignore")

    data_dir: Path = Field(default=Path("data"), description="Root analytical data directory")
    raw_dir: Path = Field(default=Path("data/raw"), description="Directory for immutable raw data")
    processed_dir: Path = Field(
        default=Path("data/processed"), description="Directory for cleaned/adjusted data"
    )
    features_dir: Path = Field(
        default=Path("data/features"), description="Directory for engineered feature sets"
    )
    predictions_dir: Path = Field(
        default=Path("data/predictions"), description="Directory for logged predictions"
    )
    models_dir: Path = Field(
        default=Path("data/models"), description="Directory for serialized model artifacts"
    )
    reports_dir: Path = Field(
        default=Path("data/reports"), description="Directory for generated reports"
    )
    log_level: str = Field(default="INFO", description="Application logging level")
    timezone: str = Field(default="Asia/Karachi", description="Market local timezone")

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Ensure log level is uppercase and valid."""
        cleaned = value.strip().upper()
        allowed = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
        if cleaned not in allowed:
            raise ValueError(f"Invalid log level: '{value}'. Must be one of {allowed}")
        return cleaned


class AppConfig(BaseModel):
    """Root configuration container combining settings and stock definitions."""

    model_config = ConfigDict(extra="ignore")

    settings: SettingsConfig = Field(default_factory=SettingsConfig)
    stocks: list[StockConfig] = Field(default_factory=list)

    def get_stock(self, symbol: str) -> Optional[StockConfig]:
        """Find a stock configuration by symbol (case-insensitive)."""
        target = symbol.strip().upper()
        for stock in self.stocks:
            if stock.symbol == target:
                return stock
        return None

    def get_enabled_stocks(self) -> list[StockConfig]:
        """Retrieve all currently active and enabled stocks."""
        return [stock for stock in self.stocks if stock.enabled]
