# Phase 02 — Historical Price Collector Interface

## 1. Objective
Establish an extensible collector interface (`BaseCollector`) and implement a production-quality historical price collector for PSX securities using Yahoo Finance (`.KA` tickers) as the primary data source and PSX Data Portal (DPS) as a supplemental validator. All external network interactions must be thoroughly mocked for deterministic unit testing.

---

## 2. Deliverables & Files to Create
```text
src/psx_predictor/collectors/
├── __init__.py
├── base.py                            # Abstract BaseCollector class
├── yahoo_collector.py                 # Yahoo Finance .KA collector implementation
└── dps_collector.py                   # PSX Data Portal scraper/collector stub

tests/unit/
└── test_collectors.py                 # Mocked network and retry tests

tests/integration/
└── test_live_collector.py             # Optional live test marked with @pytest.mark.external
```

---

## 3. Detailed Specifications

### 3.1 Abstract Base Collector (`collectors/base.py`)
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd

class BaseCollector(ABC):
    @abstractmethod
    def fetch_historical(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch historical raw OHLCV data from external source."""
        pass

    @abstractmethod
    def normalize(self, raw_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Convert source-specific column names and formats into PSX standard schema."""
        pass

    @abstractmethod
    def save_raw(self, raw_df: pd.DataFrame, symbol: str) -> None:
        """Persist immutable raw download to data/raw/prices/."""
        pass
```

### 3.2 Yahoo Finance Collector (`collectors/yahoo_collector.py`)
- Maps local symbols to Yahoo symbols (e.g., `OGDC` $\to$ `OGDC.KA`).
- Uses `yfinance` or direct `httpx` query to Yahoo Finance Chart API.
- Implements exponential backoff: 3 retries with jitter (1s, 2s, 4s).
- Extracts: `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`, `Dividends`, `Stock Splits`.
- Detects circuit breaker lock events:
  - `is_upper_lock`: `Close >= Prev_Close * 1.074` and `High == Low == Close`.
  - `is_lower_lock`: `Close <= Prev_Close * 0.926` and `High == Low == Close`.
- Persists raw payload to `data/raw/prices/{symbol}_yahoo_{timestamp}.parquet`.

---

## 4. Testing Plan
- `test_collector_retries_on_http_error()`: Uses `unittest.mock` to simulate HTTP 500/503 errors and verifies 3 retries before raising custom `CollectorError`.
- `test_collector_normalizes_columns()`: Verifies source columns are correctly mapped to `symbol, trade_date, open, high, low, close, adjusted_close, volume, dividend_amount, split_ratio`.
- `test_collector_lock_detection()`: Verifies synthetic 7.5% limit moves flag `is_upper_lock` and `is_lower_lock`.
- `test_collector_handles_empty_response()`: Asserts that querying an unlisted symbol raises `SymbolNotFoundError`.

---

## 5. Validation Commands
```bash
pytest tests/unit/test_collectors.py -v
psx data fetch --symbol OGDC --days 30 --dry-run
ruff check .
```

---

## 6. Phase Completion Exit Criteria
- [ ] Base collector interface clean and decoupled.
- [ ] Yahoo Finance collector functions with mocked tests passing 100%.
- [ ] Circuit breaker upper/lower locks correctly calculated.
- [ ] Raw responses persisted before normalization.
- [ ] Phase 2 completion report documented.
