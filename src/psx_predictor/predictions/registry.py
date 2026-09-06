"""Permanent append-only prediction registry storing model forecasts and realized outcomes."""

import datetime
import uuid
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, Field

from psx_predictor.config.loader import load_config
from psx_predictor.storage.parquet_io import read_parquet, write_parquet_atomic
from psx_predictor.storage.paths import ensure_directories


class PredictionRecord(BaseModel):
    """Immutable schema invariant for logged market predictions."""

    prediction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    generated_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    target_date: str  # YYYY-MM-DD
    model_name: str
    model_version: str = "1.0.0"
    up_probability: float = Field(ge=0.0, le=1.0)
    expected_return: Optional[float] = None
    signal: str  # "BUY", "SELL", "HOLD"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    feature_version: str = "1.0.0"
    drivers_json: str = "{}"
    realized_outcome: Optional[float] = None
    is_correct: Optional[bool] = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class PredictionRegistry:
    """Manages atomic append, query, and reconciliation updates to predictions.parquet."""

    def __init__(self, storage_paths: Optional[dict[str, Path]] = None) -> None:
        if storage_paths is None:
            config = load_config()
            self.storage_paths = ensure_directories(config.settings.data_dir)
        else:
            self.storage_paths = storage_paths

        self.predictions_file = self.storage_paths["predictions"] / "predictions.parquet"

    def log_predictions(self, records: list[PredictionRecord]) -> None:
        """Atomically append new prediction records to the registry."""
        if not records:
            return

        new_data = pd.DataFrame([r.to_dict() for r in records])

        if self.predictions_file.exists():
            existing_df = read_parquet(self.predictions_file)
            combined_df = pd.concat([existing_df, new_data], ignore_index=True)
            # Deduplicate by prediction_id if any duplicate
            combined_df = combined_df.drop_duplicates(subset=["prediction_id"], keep="last")
        else:
            combined_df = new_data

        write_parquet_atomic(combined_df, self.predictions_file)

    def log_prediction(self, record: PredictionRecord) -> None:
        """Atomically append a single prediction record."""
        self.log_predictions([record])

    def get_predictions(
        self,
        symbol: Optional[str] = None,
        unresolved_only: bool = False,
    ) -> pd.DataFrame:
        """Query predictions with optional symbol and status filters."""
        if not self.predictions_file.exists():
            return pd.DataFrame()

        df = read_parquet(self.predictions_file)
        if symbol:
            df = df[df["symbol"] == symbol.upper()].reset_index(drop=True)

        if unresolved_only:
            df = df[df["realized_outcome"].isna()].reset_index(drop=True)

        return df

    def update_predictions(self, updated_df: pd.DataFrame) -> None:
        """Update existing prediction records (e.g. after audit reconciliation)."""
        if not self.predictions_file.exists():
            write_parquet_atomic(updated_df, self.predictions_file)
            return

        existing_df = read_parquet(self.predictions_file)

        # Merge or update by prediction_id
        merged = pd.concat([existing_df, updated_df], ignore_index=True)
        # Keep last instance for updated records
        final_df = merged.drop_duplicates(subset=["prediction_id"], keep="last").reset_index(
            drop=True
        )

        write_parquet_atomic(final_df, self.predictions_file)
