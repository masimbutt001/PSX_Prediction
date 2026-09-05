"""Embedded in-process DuckDB analytical client and virtual view manager."""

from pathlib import Path
from typing import Any, Optional

import duckdb
import pandas as pd
from loguru import logger

from psx_predictor.storage.paths import get_storage_paths


class DuckDBClient:
    """Manages an in-process DuckDB analytical engine connected to Parquet datasets."""

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        """Initialize DuckDB connection.

        Args:
            db_path: Path to database file, or None for an in-memory database.
        """
        target = str(db_path) if db_path else ":memory:"
        self._db_path = target
        self._conn: duckdb.DuckDBPyConnection = duckdb.connect(target)
        logger.debug(f"Initialized DuckDB connection to '{self._db_path}'")

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Access underlying DuckDB connection object."""
        return self._conn

    def register_view(self, view_name: str, parquet_path: str | Path) -> None:
        """Register a virtual SQL view over one or more Parquet files.

        Args:
            view_name: Target SQL view name.
            parquet_path: File path or glob string (e.g. 'data/processed/prices/*.parquet').
        """
        clean_path = str(parquet_path).replace("\\", "/")
        sql = f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet('{clean_path}')"
        try:
            self._conn.execute(sql)
            logger.debug(f"Registered DuckDB view '{view_name}' -> '{clean_path}'")
        except Exception as e:
            logger.warning(f"Could not register view '{view_name}' for '{clean_path}': {e}")

    def register_standard_views(self, base_dir: Optional[Path] = None) -> None:
        """Register all standard analytical storage directories as queryable views."""
        paths = get_storage_paths(base_dir)

        # Map views to Parquet glob patterns
        views = {
            "raw_prices": paths["raw_prices"] / "*.parquet",
            "processed_prices": paths["processed_prices"] / "*.parquet",
            "features_technical": paths["features_technical"] / "*.parquet",
            "features_combined": paths["features_combined"] / "*.parquet",
            "predictions": paths["predictions"] / "*.parquet",
        }

        for view_name, glob_path in views.items():
            # Only register if directory contains parquet files or directory exists
            parent_dir = glob_path.parent
            if parent_dir.exists() and any(parent_dir.glob("*.parquet")):
                self.register_view(view_name, glob_path)

    def query(self, sql: str, params: Optional[dict[str, Any] | list[Any]] = None) -> pd.DataFrame:
        """Execute a SQL query against registered views and Parquet files.

        Args:
            sql: SQL statement to execute.
            params: Optional parameter substitution dictionary or list.

        Returns:
            Query results as a pandas DataFrame.
        """
        try:
            if params is not None:
                return self._conn.execute(sql, params).df()
            return self._conn.execute(sql).df()
        except Exception as e:
            logger.error(f"DuckDB query failed: {e}\nSQL: {sql}")
            raise

    def close(self) -> None:
        """Close database connection."""
        try:
            self._conn.close()
            logger.debug(f"Closed DuckDB connection to '{self._db_path}'")
        except Exception:
            pass

    def __enter__(self) -> "DuckDBClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
