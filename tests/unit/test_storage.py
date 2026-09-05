"""Unit tests for Phase 01: Local Storage Layer & DuckDB Catalog."""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from psx_predictor.cli.main import app
from psx_predictor.storage.duckdb_client import DuckDBClient
from psx_predictor.storage.parquet_io import (
    DatasetNotFoundError,
    StorageError,
    read_parquet,
    write_parquet_atomic,
)
from psx_predictor.storage.paths import ensure_directories
from psx_predictor.storage.schemas import OHLCVRecord, PriceDatasetValidator


def test_ensure_directories(tmp_path: Path) -> None:
    """Verify that ensure_directories safely creates all standard analytical subdirectories."""
    data_dir = tmp_path / "custom_data"
    paths = ensure_directories(data_dir)

    assert paths["raw_prices"].exists()
    assert paths["processed_prices"].exists()
    assert paths["features_technical"].exists()
    assert paths["predictions"].exists()
    assert paths["models"].exists()
    assert paths["reports"].exists()


def test_atomic_write_and_read(tmp_path: Path) -> None:
    """Verify atomic write serializes dataframe and read recovers it accurately."""
    target_file = tmp_path / "prices" / "OGDC.parquet"
    df = pd.DataFrame(
        {
            "symbol": ["OGDC", "OGDC"],
            "trade_date": ["2026-01-02", "2026-01-05"],
            "close": [115.0, 117.5],
        }
    )

    write_parquet_atomic(df, target_file)

    assert target_file.exists()
    # Ensure no lingering temporary files remain
    tmp_files = list(target_file.parent.glob("*.tmp"))
    assert len(tmp_files) == 0

    loaded_df = read_parquet(target_file)
    assert len(loaded_df) == 2
    assert list(loaded_df["symbol"]) == ["OGDC", "OGDC"]
    assert list(loaded_df["close"]) == [115.0, 117.5]


def test_atomic_write_preserves_original_on_failure(tmp_path: Path) -> None:
    """Verify that if a write fails, the pre-existing file is NOT corrupted or overwritten."""
    target_file = tmp_path / "test.parquet"
    original_df = pd.DataFrame({"val": [1, 2, 3]})
    write_parquet_atomic(original_df, target_file)

    # Attempt to write with a failing mock on to_parquet
    with patch.object(pd.DataFrame, "to_parquet", side_effect=RuntimeError("Simulated disk error")):
        bad_df = pd.DataFrame({"val": [999]})
        with pytest.raises(StorageError):
            write_parquet_atomic(bad_df, target_file)

    # Verify original file contents remain intact
    restored_df = read_parquet(target_file)
    assert list(restored_df["val"]) == [1, 2, 3]


def test_read_parquet_missing_raises_error(tmp_path: Path) -> None:
    """Verify DatasetNotFoundError is raised when file does not exist."""
    missing_file = tmp_path / "does_not_exist.parquet"
    with pytest.raises(DatasetNotFoundError):
        read_parquet(missing_file)


def test_ohlcv_validation_success() -> None:
    """Verify valid OHLCV records pass validation successfully."""
    record = OHLCVRecord(
        symbol="ogdc",
        trade_date="2026-01-02",
        open=115.0,
        high=118.0,
        low=114.5,
        close=117.0,
        adjusted_close=117.0,
        volume=2500000,
        is_upper_lock=False,
    )
    assert record.symbol == "OGDC"
    assert record.high >= record.low
    assert record.high >= record.open
    assert record.volume == 2500000


def test_ohlcv_validation_rejects_inverted_prices() -> None:
    """Verify OHLC mathematical invariants: High < Low or Low > Open raises error."""
    # High < Low
    with pytest.raises(ValidationError):
        OHLCVRecord(
            symbol="OGDC",
            trade_date="2026-01-02",
            open=115.0,
            high=110.0,  # Invalid: High < Low
            low=114.0,
            close=112.0,
            adjusted_close=112.0,
            volume=1000,
        )

    # Low > Open
    with pytest.raises(ValidationError):
        OHLCVRecord(
            symbol="OGDC",
            trade_date="2026-01-02",
            open=110.0,
            high=115.0,
            low=112.0,  # Invalid: Low > Open
            close=114.0,
            adjusted_close=114.0,
            volume=1000,
        )


def test_ohlcv_validation_rejects_zero_or_negative_price() -> None:
    """Verify zero or negative prices are strictly rejected."""
    with pytest.raises(ValidationError):
        OHLCVRecord(
            symbol="OGDC",
            trade_date="2026-01-02",
            open=0.0,  # Invalid: must be > 0
            high=10.0,
            low=0.0,
            close=5.0,
            adjusted_close=5.0,
            volume=100,
        )


def test_dataset_deduplication() -> None:
    """Verify PriceDatasetValidator deduplicates duplicate dates and sorts by date."""
    raw_df = pd.DataFrame(
        [
            {
                "symbol": "OGDC",
                "trade_date": "2026-01-05",
                "open": 110.0,
                "high": 112.0,
                "low": 109.0,
                "close": 111.0,
                "adjusted_close": 111.0,
                "volume": 1000,
            },
            {
                "symbol": "OGDC",
                "trade_date": "2026-01-02",
                "open": 108.0,
                "high": 110.0,
                "low": 107.0,
                "close": 109.0,
                "adjusted_close": 109.0,
                "volume": 2000,
            },
            # Duplicate of 2026-01-05 (updated volume)
            {
                "symbol": "OGDC",
                "trade_date": "2026-01-05",
                "open": 110.0,
                "high": 112.0,
                "low": 109.0,
                "close": 111.0,
                "adjusted_close": 111.0,
                "volume": 5555,
            },
        ]
    )

    cleaned = PriceDatasetValidator.validate_dataframe(raw_df)
    assert len(cleaned) == 2
    # Verify chronological sort: 2026-01-02 comes first
    assert str(cleaned.iloc[0]["trade_date"]) == "2026-01-02"
    assert str(cleaned.iloc[1]["trade_date"]) == "2026-01-05"
    # Verify deduplication kept the last entry
    assert cleaned.iloc[1]["volume"] == 5555


def test_fixture_file_validation() -> None:
    """Verify the sample_prices.json fixture loads and validates all 5 records."""
    fixture_path = Path("tests/fixtures/sample_prices.json")
    assert fixture_path.exists()

    with open(fixture_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    df = pd.DataFrame(raw_data)
    validated = PriceDatasetValidator.validate_dataframe(df)

    assert len(validated) == 5
    assert set(validated["symbol"].unique()) == {"OGDC", "PPL"}


def test_duckdb_view_and_query(tmp_path: Path) -> None:
    """Verify DuckDB registers Parquet files as views and runs analytical queries."""
    prices_dir = tmp_path / "data" / "processed" / "prices"
    prices_dir.mkdir(parents=True)

    ogdc_df = pd.DataFrame(
        {
            "symbol": ["OGDC", "OGDC"],
            "trade_date": ["2026-01-02", "2026-01-05"],
            "close": [117.2, 120.4],
            "volume": [2450000, 3120000],
        }
    )
    write_parquet_atomic(ogdc_df, prices_dir / "OGDC.parquet")

    ppl_df = pd.DataFrame(
        {
            "symbol": ["PPL", "PPL"],
            "trade_date": ["2026-01-02", "2026-01-05"],
            "close": [86.1, 85.4],
            "volume": [1780000, 1230000],
        }
    )
    write_parquet_atomic(ppl_df, prices_dir / "PPL.parquet")

    with DuckDBClient() as client:
        parquet_glob = prices_dir / "*.parquet"
        client.register_view("processed_prices", parquet_glob)

        result = client.query(
            """
            SELECT symbol, AVG(close) as avg_close, SUM(volume) as total_volume
            FROM processed_prices
            GROUP BY symbol
            ORDER BY symbol
            """
        )

        assert len(result) == 2
        assert list(result["symbol"]) == ["OGDC", "PPL"]
        assert round(result.iloc[0]["avg_close"], 2) == 118.80
        assert result.iloc[0]["total_volume"] == 5570000


def test_cli_storage_commands(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test CLI commands: psx storage init and psx storage status."""
    custom_dir = tmp_path / "cli_storage"

    # 1. Test init
    init_res = cli_runner.invoke(app, ["storage", "init", "--base-dir", str(custom_dir)])
    assert init_res.exit_code == 0
    assert "Initialized Storage Directories" in init_res.output
    assert custom_dir.exists()

    # 2. Test status
    status_res = cli_runner.invoke(app, ["storage", "status", "--base-dir", str(custom_dir)])
    assert status_res.exit_code == 0
    assert "Storage Status" in status_res.output
    assert "DuckDB Engine: Connected" in status_res.output


def test_cli_data_inspect(cli_runner: CliRunner) -> None:
    """Test CLI command: psx data inspect --fixture."""
    res = cli_runner.invoke(
        app,
        ["data", "inspect", "--fixture", "tests/fixtures/sample_prices.json"],
    )
    assert res.exit_code == 0
    assert "Dataset Summary" in res.output
    assert "OGDC" in res.output
    assert "PPL" in res.output
    assert "Sample Rows Preview" in res.output
