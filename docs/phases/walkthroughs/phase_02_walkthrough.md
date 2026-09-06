# Walkthrough — Phase 02: Historical Price Collector Interface & Multi-Source Engine

**Status:** Completed & Validated  
**Date:** September 2026  
**Specification:** [phase_02_collector.md](../phase_02_collector.md)

---

## 1. Objective Accomplished
Engineered a resilient, multi-source ingestion framework capable of collecting historical daily stock prices from the Pakistan Stock Exchange. The system features:
1. An abstract base interface (`BaseCollector`) with automated payload archiving in `data/raw/prices/`, price normalization, and PSX $\pm 7.5\%$ circuit breaker detection (`calculate_circuit_locks`).
2. A high-performance reverse-engineered collector for **SCS Trade** (`SCSTradeCollector`) extracting data from `POST /MarketStatistics/MS_HistoricalPrices.aspx/chart`.
3. An official exchange collector and scraper for **PSX Data Portal (DPS)** (`DPSCollector`) supporting both the timeseries API and HTML historical table fallback.
4. An international fallback collector for **Yahoo Finance** (`YahooCollector`) mapping PSX tickers to `{SYMBOL}.KA`.
5. A tiered failover and gap-filling aggregator (`CompositeCollector`) coordinating primary and backup data sources with automatic backfilling.
6. An extensible CLI command `psx data fetch` with dry-run support, summary reporting, and atomic Parquet persistence.

---

## 2. Deliverables & Files Created

| File | Purpose |
| :--- | :--- |
| [`src/psx_predictor/collectors/base.py`](../../../src/psx_predictor/collectors/base.py) | Defines `BaseCollector` ABC, custom exception hierarchy (`CollectorError`, `RateLimitError`, `NetworkTimeoutError`, `SymbolNotFoundError`, `EmptyResponseError`), raw payload persistence, schema validation bridge, and PSX circuit breaker lock calculations. |
| [`src/psx_predictor/collectors/scs_collector.py`](../../../src/psx_predictor/collectors/scs_collector.py) | Implementation of `SCSTradeCollector`. Sends JSON POST requests to SCS Trade historical price endpoints, converts ASP.NET `/Date(epoch_ms)/` timestamps, normalizes OHLCV data, and handles network retries with exponential backoff. |
| [`src/psx_predictor/collectors/yahoo_collector.py`](../../../src/psx_predictor/collectors/yahoo_collector.py) | Implementation of `YahooCollector` wrapping `yfinance`. Appends `.KA` suffix, handles splits and dividends, and validates non-zero price series. |
| [`src/psx_predictor/collectors/dps_collector.py`](../../../src/psx_predictor/collectors/dps_collector.py) | Implementation of `DPSCollector`. Fetches EOD timeseries JSON from PSX Data Portal (`dps.psx.com.pk`) and falls back to BeautifulSoup HTML table scraping if timeseries endpoint is unavailable. |
| [`src/psx_predictor/collectors/composite.py`](../../../src/psx_predictor/collectors/composite.py) | Implementation of `CompositeCollector`. Tiers collectors in priority order (`SCS Trade` -> `Yahoo Finance` -> `PSX DPS`), falls back on error, and backfills date gaps. |
| [`src/psx_predictor/collectors/__init__.py`](../../../src/psx_predictor/collectors/__init__.py) | Package initialization exporting public collector classes and exceptions. |
| [`src/psx_predictor/cli/main.py`](../../../src/psx_predictor/cli/main.py) | Added `psx data fetch` command supporting `--symbol`, `--source`, `--days`, `--start-date`, `--end-date`, `--save`, and `--dry-run`. |
| [`tests/unit/test_collectors.py`](../../../tests/unit/test_collectors.py) | 12 comprehensive unit tests using `respx` and `unittest.mock` covering SCS parsing, Yahoo conversion, DPS JSON and HTML parsing, circuit breaker flags, retry logic, and Composite failover/gap backfill. |
| [`tests/integration/test_live_collector.py`](../../../tests/integration/test_live_collector.py) | Live integration test verifying live HTTP communication against SCS Trade and PSX DPS (marked with `@pytest.mark.external`). |

---

## 3. Verification & Test Results

### 3.1 Code Quality & Typing
```powershell
uv run ruff format --check .
# Result: 25 files already formatted

uv run ruff check .
# Result: All checks passed!

uv run mypy src
# Result: Success: no issues found in 17 source files
```

### 3.2 Automated Test Suite Execution
All unit tests execute offline with zero network dependency using mocked HTTP responses:
```powershell
uv run pytest -v
```
**Output:**
```text
============================= test session starts =============================
platform win32 -- Python 3.12.11, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\Development\PSX_Prediction
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.15.1, respx-0.23.1
collected 36 items / 2 deselected / 34 selected

tests\unit\test_collectors.py ............                               [ 35%]
tests\unit\test_config.py ..........                                     [ 64%]
tests\unit\test_storage.py ............                                  [100%]

====================== 34 passed, 2 deselected in 0.62s =======================
```

### 3.3 Live CLI Verification
Tested live fetch against SCS Trade for `OGDC` in dry-run mode:
```powershell
uv run psx data fetch --symbol OGDC --source scs --days 10 --dry-run
```
**Output:**
```text
Fetching OGDC via provider 'scs' (2026-08-27 to 2026-09-06)...
[scs] Fetching historical data for OGDC from 2026-08-27 to 2026-09-06
[scs] Successfully retrieved 7 validated sessions for OGDC
           Fetch Results for OGDC (scs)           
+------------------------------------------------+
| Metric              | Value                    |
|---------------------+--------------------------|
| Sessions Retrieved  | 7                        |
| Date Bounds         | 2026-08-26 to 2026-09-03 |
| Latest Close        | PKR 328.80               |
| Upper Lock Sessions | 0                        |
| Lower Lock Sessions | 0                        |
+------------------------------------------------+
Dry run active: No files saved to processed storage.
```
