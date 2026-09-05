# Phase 01 — Local Storage Layer & DuckDB Catalog

## 1. Objective
Build a robust, local-first storage foundation before connecting any external network data sources. This layer provides type-safe Parquet reading/writing, DuckDB analytical SQL querying, atomic file writes to prevent data corruption, OHLCV validation schemas, and deduplication logic using local fixture data.

---

## 2. Deliverables & Files to Create
```text
src/psx_predictor/storage/
├── __init__.py
├── paths.py                           # Directory resolution and management
├── parquet_io.py                      # Atomic Parquet read/write helpers
├── duckdb_client.py                   # DuckDB connection manager & views
└── schemas.py                         # Storage-level Pydantic schemas

tests/fixtures/
└── sample_prices.json                 # Offline sample OHLCV fixture data

tests/unit/
└── test_storage.py                    # Storage, atomic writes, and validation tests
```

---

## 3. Detailed Specifications

### 3.1 Directory & Path Manager (`storage/paths.py`)
- Standard paths:
  - `data/raw/{prices, news, announcements, macro}`
  - `data/processed/{prices, corporate_actions}`
  - `data/features/{technical, combined}`
  - `data/predictions/`
  - `data/models/`
- Helper `ensure_directories()` creates all required data folders safely if absent.

### 3.2 Atomic Parquet I/O (`storage/parquet_io.py`)
- `write_parquet_atomic(df: pd.DataFrame, path: Path) -> None`:
  - Writes data first to a temporary file in the same filesystem: `path.with_suffix(".tmp")`.
  - Atomically replaces the destination path using `os.replace`.
  - Ensures that interrupted writes or crashes never leave a corrupt Parquet file.
- `read_parquet(path: Path) -> pd.DataFrame`:
  - Reads Parquet file, raising custom `DatasetNotFoundError` if file does not exist.

### 3.3 Data Validation Rules (`storage/schemas.py`)
- Validates:
  - `symbol` non-empty uppercase string.
  - `trade_date` valid `datetime.date`.
  - `open, high, low, close, adjusted_close` are strictly $> 0`.
  - Invariants: `high >= open`, `high >= close`, `high >= low`, `low <= open`, `low <= close`.
  - `volume >= 0`.
  - `dividend_amount >= 0.0`.
  - `split_ratio > 0.0`.
  - `is_upper_lock`, `is_lower_lock` are boolean flags.
- Rejects and logs invalid rows with clear error context.

### 3.4 DuckDB Analytical Interface (`storage/duckdb_client.py`)
- In-process DuckDB connection.
- Registers Parquet files as queryable virtual views:
  - `CREATE OR REPLACE VIEW processed_prices AS SELECT * FROM 'data/processed/prices/*.parquet'`
- Provides `query(sql: str) -> pd.DataFrame` utility.

---

## 4. Testing Plan
- `test_atomic_write_creates_file()`: Verifies write creates valid Parquet file and cleans up `.tmp`.
- `test_atomic_write_prevents_corruption()`: Simulates failure and verifies original file remains intact.
- `test_ohlcv_validation_rejects_inverted_prices()`: Asserts high < low raises validation error.
- `test_duplicate_rejection()`: Ingestion of identical symbol and trade_date rows deduplicates properly.
- `test_duckdb_view_query()`: Reads multiple sample stock Parquets through DuckDB SQL query.

---

## 5. Validation Commands
```bash
pytest tests/unit/test_storage.py -v
psx data inspect --fixture tests/fixtures/sample_prices.json
ruff check .
```

---

## 6. Phase Completion Exit Criteria
- [ ] All atomic Parquet write and read operations succeed.
- [ ] Validation functions correctly reject invalid OHLC, zero prices, and duplicates.
- [ ] DuckDB successfully queries local Parquet files.
- [ ] Phase 1 completion report documented.
