"""Atomic read and write operations for Apache Parquet datasets."""

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


class StorageError(Exception):
    """Base exception for storage subsystem errors."""

    pass


class DatasetNotFoundError(StorageError, FileNotFoundError):
    """Raised when an expected dataset or Parquet file cannot be found."""

    pass


def write_parquet_atomic(
    df: pd.DataFrame,
    path: Path,
    compression: str = "zstd",
    index: bool = False,
) -> None:
    """Atomically write a pandas DataFrame to a Parquet file.

    Writes to a temporary file in the destination directory first and then
    atomically moves/renames it into place using os.replace. This prevents
    partially written or corrupted files during crashes or interruptions.

    Args:
        df: DataFrame to serialize.
        path: Target destination Path for the Parquet file.
        compression: Parquet compression codec (default: 'zstd').
        index: Whether to include the DataFrame index in the Parquet file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.stem}_{os.getpid()}.tmp")

    try:
        df.to_parquet(temp_path, engine="pyarrow", compression=compression, index=index)
        os.replace(temp_path, path)
        logger.debug(f"Atomically wrote {len(df)} records to {path}")
    except Exception as e:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        logger.error(f"Failed atomic write to {path}: {e}")
        raise StorageError(f"Atomic write to {path} failed: {e}") from e


def read_parquet(
    path: Path,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Read a Parquet file into a pandas DataFrame with verification.

    Args:
        path: Path to the Parquet file to read.
        columns: Optional list of column names to load.

    Returns:
        Loaded DataFrame.

    Raises:
        DatasetNotFoundError: If the requested file does not exist.
    """
    if not path.exists():
        raise DatasetNotFoundError(f"Parquet file not found at: {path}")

    try:
        return pd.read_parquet(path, engine="pyarrow", columns=columns)
    except Exception as e:
        raise StorageError(f"Failed to read Parquet file at {path}: {e}") from e
