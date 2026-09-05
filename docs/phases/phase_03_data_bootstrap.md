# Phase 03 — Historical Data Bootstrap

## 1. Objective
Bootstrap the local dataset with multi-year (e.g., 3–5 years) daily historical stock data for all enabled symbols in `config/stocks.yaml`. Execute full validation, normalization, and quality audits, and write processed datasets into `data/processed/prices/{symbol}.parquet`.

---

## 2. Deliverables & Files to Create
```text
src/psx_predictor/processing/
├── __init__.py
├── pipeline.py                        # Ingestion orchestrator
└── quality_report.py                  # Data health and gap detector

tests/unit/
└── test_pipeline.py                   # Data ingestion and quality reporting tests
```

---

## 3. Detailed Specifications

### 3.1 Bootstrap Command (`psx data bootstrap --years 5`)
- Iterates over all enabled symbols in `config/stocks.yaml`.
- For each symbol:
  1. Calls collector for specified lookback period (e.g. 5 years).
  2. Saves raw file to `data/raw/prices/`.
  3. Validates OHLCV constraints and removes corrupted rows.
  4. Deduplicates records on `trade_date`.
  5. Computes continuous `adjusted_close` honoring dividends and splits.
  6. Atomically writes to `data/processed/prices/{symbol}.parquet`.
  7. Isolates failures: If symbol $A$ fails, log error and continue with symbol $B$.

### 3.2 Data Quality Report
- Produces a comprehensive summary table for each ingested stock:
  - First trading date & Last trading date
  - Total trading sessions
  - Duplicate rows removed
  - Missing sessions detected (comparing against expected PSX business days)
  - Dividend distribution count
  - Lock limit sessions count (Upper locks & Lower locks)

---

## 4. Testing Plan
- `test_bootstrap_orchestration()`: Mocks collector and verifies full pipeline writes processed Parquet.
- `test_failure_isolation()`: Mocks failure on second stock and verifies first and third succeed.
- `test_quality_report_detects_gaps()`: Feeds fixture with missing week and verifies report flags gap.

---

## 5. Validation Commands
```bash
pytest tests/unit/test_pipeline.py -v
psx data bootstrap --years 5 --dry-run
ruff check .
```

---

## 6. Phase Completion Exit Criteria
- [ ] Bootstrap pipeline executes cleanly for all configured stocks.
- [ ] Processed Parquet files written to `data/processed/prices/`.
- [ ] Data quality audit passes with zero unhandled nulls or corrupt rows.
- [ ] Phase 3 completion report documented.
