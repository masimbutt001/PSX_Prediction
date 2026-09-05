"""Storage subsystem for PSX Predictor: Parquet I/O, schemas, and DuckDB catalog."""

from psx_predictor.storage.duckdb_client import DuckDBClient
from psx_predictor.storage.parquet_io import (
    DatasetNotFoundError,
    StorageError,
    read_parquet,
    write_parquet_atomic,
)
from psx_predictor.storage.paths import ensure_directories, get_storage_paths
from psx_predictor.storage.schemas import OHLCVRecord, PriceDatasetValidator

__all__ = [
    "DatasetNotFoundError",
    "DuckDBClient",
    "OHLCVRecord",
    "PriceDatasetValidator",
    "StorageError",
    "ensure_directories",
    "get_storage_paths",
    "read_parquet",
    "write_parquet_atomic",
]
