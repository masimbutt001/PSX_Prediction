# Phase 02 — Historical Price Collector Interface & Multi-Source Engine

## 1. Objective
Establish an extensible collector interface (`BaseCollector`) and implement a production-quality, multi-source data ingestion engine for Pakistan Stock Exchange (PSX) securities:
1. **SCS Trade Collector (`scstrade.com`):** High-speed, decade-long official PSX historical data endpoint (`MS_HistoricalPrices.aspx/chart`) with volume and price change.
2. **Yahoo Finance Collector (`.KA` tickers):** Fast global provider for OHLCV and corporate action splits.
3. **PSX Data Portal (DPS) Collector (`dps.psx.com.pk`):** Official exchange timeseries JSON endpoint.
4. **Composite Fallback Engine:** Automatically queries the best available source and backfills any missing sessions or gaps across providers.

All external network interactions must be thoroughly mocked for deterministic, offline unit testing.

---

## 2. Deliverables & Files to Create
```text
src/psx_predictor/collectors/
├── __init__.py
├── base.py                            # Abstract BaseCollector class & exceptions
├── scs_collector.py                   # SCS Trade (scstrade.com) historical collector
├── yahoo_collector.py                 # Yahoo Finance .KA collector implementation
├── dps_collector.py                   # PSX Data Portal (dps.psx.com.pk) collector
└── composite.py                       # Composite multi-source fallback orchestrator

tests/unit/
└── test_collectors.py                 # Mocked network, retry, parsing, and fallback tests

tests/integration/
└── test_live_collector.py             # Live integration tests marked with @pytest.mark.external
```

---

## 3. Detailed Specifications

### 3.1 Abstract Base Collector (`collectors/base.py`)
```python
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

class BaseCollector(ABC):
    @property
    @abstractmethod
    def source_name(self) -> str:
        """Identifier of the data provider (e.g., 'scs', 'yahoo', 'dps')."""
        pass

    @abstractmethod
    def fetch_historical(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch raw historical OHLCV data from the external source."""
        pass

    @abstractmethod
    def normalize(self, raw_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalize raw source columns into PSX standard schema."""
        pass

    @abstractmethod
    def save_raw(self, raw_df: pd.DataFrame, symbol: str) -> None:
        """Persist immutable raw download to data/raw/prices/."""
        pass
```

### 3.2 SCS Trade Collector (`collectors/scs_collector.py`)
Ingestion directly from Standard Capital Securities (`https://www.scstrade.com/MarketStatistics/MS_HistoricalPrices.aspx/chart`):
- **Endpoint:** `POST https://www.scstrade.com/MarketStatistics/MS_HistoricalPrices.aspx/chart`
- **Request Headers:**
  - `Content-Type: application/json; charset=utf-8`
  - `User-Agent: Mozilla/5.0`
  - `Referer: https://www.scstrade.com/MarketStatistics/MS_HistoricalPrices.aspx`
- **JSON Payload:**
  ```json
  {
    "par": "OGDC",
    "date1": "01-Jan-2020",
    "date2": "06-Sep-2026"
  }
  ```
- **Response Format:**
  Extracts records from `d` array containing:
  - `trading_Date`: Microsoft JSON Date format (e.g., `"/Date(1705258800000)/"` $\to$ parsed to `datetime.date`)
  - `trading_open`, `trading_high`, `trading_low`, `trading_close`, `trading_vol`, `trading_change`
- **Lock Limits Detection:**
  Computes upper and lower locks:
  - `is_upper_lock`: `trading_close >= prev_close * 1.074` and `trading_high == trading_low == trading_close`
  - `is_lower_lock`: `trading_close <= prev_close * 0.926` and `trading_high == trading_low == trading_close`
- **Persistence:**
  Saves raw payload to `data/raw/prices/{symbol}_scs_{timestamp}.json`.

### 3.3 Yahoo Finance Collector (`collectors/yahoo_collector.py`)
- Maps local symbols to Yahoo symbols (e.g., `OGDC` $\to$ `OGDC.KA`).
- Uses `yfinance` or direct Yahoo Finance Chart API.
- Implements exponential backoff: 3 retries with jitter (1s, 2s, 4s).
- Extracts: `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`, `Dividends`, `Stock Splits`.
- Saves raw payload to `data/raw/prices/{symbol}_yahoo_{timestamp}.parquet`.

### 3.4 PSX Data Portal (DPS) Collector (`collectors/dps_collector.py`)
- Endpoint: `GET https://dps.psx.com.pk/timeseries/eod/{symbol}`
- Fallback: HTML table parsing on `https://dps.psx.com.pk/historical` using `httpx` + `BeautifulSoup`.
- Saves raw payload to `data/raw/prices/{symbol}_dps_{timestamp}.json`.

### 3.5 Composite Fallback Collector (`collectors/composite.py`)
- Configurable source preference:
  - `source="scs"`: Primary local Pakistani institutional broker (high fidelity, deep history).
  - `source="yahoo"`: Yahoo Finance global provider.
  - `source="dps"`: Official PSX exchange portal.
  - `source="composite"` (Default): Queries primary source (SCS Trade / Yahoo); if any date ranges have missing data or zero volume, automatically falls back to secondary sources to fill gaps and merge seamlessly.

---

## 4. Testing Plan
- `test_scs_collector_parses_json_date()`: Mocks SCS response and asserts `"/Date(1705258800000)/"` parses accurately to `2024-01-15`.
- `test_scs_collector_normalizes_to_standard_schema()`: Validates that all standard OHLCV columns are populated.
- `test_yahoo_collector_normalizes_columns()`: Mocks Yahoo response and verifies column mapping.
- `test_dps_collector_parses_eod_timeseries()`: Mocks DPS response and asserts date and price parsing.
- `test_collector_retries_on_http_error()`: Asserts that HTTP 500/503 errors trigger 3 exponential backoff retries.
- `test_composite_collector_fills_gaps()`: Simulates source A having missing dates and verifies composite collector fetches missing dates from source B.
- `test_lock_limit_detection()`: Asserts synthetic $\pm 7.5\%$ sessions set `is_upper_lock` and `is_lower_lock`.

---

## 5. Validation Commands
```bash
# 1. Run all unit tests with mocks
uv run pytest tests/unit/test_collectors.py -v

# 2. Test CLI data fetch with dry-run (SCS Trade)
uv run psx data fetch --symbol OGDC --source scs --days 30 --dry-run

# 3. Test CLI data fetch with dry-run (Yahoo)
uv run psx data fetch --symbol OGDC --source yahoo --days 30 --dry-run

# 4. Code quality & typing checks
uv run ruff check .
uv run mypy src
```

---

## 6. Phase Completion Exit Criteria
- [ ] `BaseCollector` interface clean and decoupled.
- [ ] `SCSTradeCollector`, `YahooCollector`, and `DPSCollector` fully implemented.
- [ ] `CompositeCollector` resolves gaps and provides automatic provider failover.
- [ ] Circuit breaker upper/lower locks correctly calculated.
- [ ] Raw payloads stored under `data/raw/prices/` prior to normalization.
- [ ] 100% offline unit test suite passing.
- [ ] Phase 2 completion report and walkthrough documented.
