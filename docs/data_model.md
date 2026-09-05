# Data Model & Storage Schema — PSX Stock Market Prediction Platform

## 1. Storage Architecture

The platform uses a tiered, local-first storage layout combining **Parquet** files for high-speed columnar disk persistence and **DuckDB** for analytical SQL queries.

```text
data/
├── raw/                               # Immutable raw downloaded payloads
│   ├── prices/{symbol}_{source}_{timestamp}.parquet (or .json)
│   ├── news/{source}_{timestamp}.json
│   ├── announcements/{symbol}_{timestamp}.json
│   └── macro/{indicator}_{timestamp}.json
│
├── processed/                          # Normalized and validated clean datasets
│   ├── prices/
│   │   └── {symbol}.parquet           # Daily OHLCV + adjusted prices
│   ├── corporate_actions/
│   │   └── {symbol}.parquet           # Dividends, bonus shares, rights
│   ├── news/
│   │   └── processed_news.parquet     # Parsed & deduplicated news articles
│   └── macro/
│       └── macro_indicators.parquet   # Synchronized macroeconomic series
│
├── features/                           # Model-ready feature datasets
│   ├── technical/
│   │   └── {symbol}_tech_features.parquet
│   └── combined/
│       └── {symbol}_dataset_v{version}.parquet
│
├── predictions/                        # Permanent audit log of model predictions
│   └── predictions.parquet
│
└── models/                             # Serialized trained model artifacts
    └── {model_type}_{symbol}_v{version}.joblib
```

---

## 2. Processed Stock Prices Schema (`data/processed/prices/{symbol}.parquet`)

| Column Name | Data Type | Nullable | Description |
| :--- | :--- | :--- | :--- |
| `symbol` | `VARCHAR` | No | PSX Ticker Symbol (e.g., `OGDC`, `PPL`, `HBL`) |
| `trade_date` | `DATE` | No | Date of the trading session |
| `open` | `DOUBLE` | No | Unadjusted opening price in PKR |
| `high` | `DOUBLE` | No | Unadjusted session high in PKR |
| `low` | `DOUBLE` | No | Unadjusted session low in PKR |
| `close` | `DOUBLE` | No | Unadjusted session closing price in PKR |
| `adjusted_close` | `DOUBLE` | No | Price adjusted for dividends and stock splits |
| `volume` | `BIGINT` | No | Total session shares traded |
| `dividend_amount`| `DOUBLE` | No | Dividend per share effective on `trade_date` (default `0.0`) |
| `split_ratio` | `DOUBLE` | No | Split or bonus multiplier effective on `trade_date` (default `1.0`) |
| `is_upper_lock` | `BOOLEAN` | No | `True` if price reached +7.5% upper limit with dry volume |
| `is_lower_lock` | `BOOLEAN` | No | `True` if price hit -7.5% lower limit with dry volume |
| `source` | `VARCHAR` | No | Data origin (`yahoo`, `dps_psx`, etc.) |
| `collected_at` | `TIMESTAMP`| No | UTC timestamp when record was acquired |

### Invariants & Validation Constraints:
- Primary Key: `(symbol, trade_date)` must be unique.
- `high >= open`, `high >= close`, `high >= low`.
- `low <= open`, `low <= close`.
- `open > 0`, `high > 0`, `low > 0`, `close > 0`, `adjusted_close > 0`.
- `volume >= 0`.

---

## 3. Corporate Actions Schema (`data/processed/corporate_actions/{symbol}.parquet`)

| Column Name | Data Type | Nullable | Description |
| :--- | :--- | :--- | :--- |
| `action_id` | `VARCHAR` | No | Unique identifier (e.g., `OGDC_DIV_20241015`) |
| `symbol` | `VARCHAR` | No | PSX Ticker Symbol |
| `action_type` | `VARCHAR` | No | `DIVIDEND`, `BONUS_ISSUE`, or `RIGHTS_ISSUE` |
| `announcement_date`| `DATE` | No | Date company announced the action to PSX |
| `ex_date` | `DATE` | No | Ex-entitlement date when price adjusts |
| `record_date` | `DATE` | Yes | Book closure record date |
| `value` | `DOUBLE` | No | Cash dividend in PKR, or bonus percentage ratio |
| `usable_from` | `TIMESTAMP`| No | Earliest market session this action could be factored |

---

## 4. Technical Features Schema (`data/features/technical/{symbol}_tech_features.parquet`)

| Feature Category | Column Name | Type | Description |
| :--- | :--- | :--- | :--- |
| **Ident** | `symbol` | `VARCHAR` | PSX Ticker |
| **Ident** | `trade_date` | `DATE` | Trading date |
| **Returns** | `return_1d` | `DOUBLE` | 1-day logarithmic return on `adjusted_close` |
| **Returns** | `return_3d`, `return_5d`, `return_10d`, `return_20d` | `DOUBLE` | Rolling cumulative returns |
| **Trend** | `sma_5`, `sma_10`, `sma_20`, `sma_50`, `sma_200` | `DOUBLE` | Simple Moving Averages |
| **Trend** | `ema_12`, `ema_26` | `DOUBLE` | Exponential Moving Averages |
| **Momentum** | `rsi_14` | `DOUBLE` | Relative Strength Index (14-session) |
| **Momentum** | `macd_line`, `macd_signal`, `macd_hist` | `DOUBLE` | Moving Average Convergence Divergence |
| **Momentum** | `roc_10` | `DOUBLE` | Rate of Change over 10 sessions |
| **Volatility**| `rolling_vol_20` | `DOUBLE` | 20-session annualized rolling standard deviation |
| **Volatility**| `atr_14` | `DOUBLE` | Average True Range |
| **Volatility**| `bb_upper`, `bb_lower`, `bb_width` | `DOUBLE` | Bollinger Bands (20-period, 2-std) |
| **Volume** | `volume_sma_20` | `DOUBLE` | 20-day Volume Moving Average |
| **Volume** | `relative_volume` | `DOUBLE` | `volume / volume_sma_20` |
| **Volume** | `obv` | `DOUBLE` | On-Balance Volume |
| **PSX Micro** | `lock_streak` | `INTEGER` | Count of consecutive limit lock days |

---

## 5. Supervised Prediction Targets Schema

| Target Name | Type | Definition | Look-Ahead Protection |
| :--- | :--- | :--- | :--- |
| `target_next_day_dir` | `INTEGER` | `1` if $Close_{t+1} > Close_t$ else `0` | Exclude last row of available dataset from training |
| `target_next_day_3class` | `VARCHAR` | `UP` (>+0.75%), `DOWN` (<-0.75%), `NEUTRAL` (between) | Exclude last row of available dataset |
| `target_return_5d` | `DOUBLE` | $(Close_{t+5} / Close_t) - 1.0$ | Exclude last 5 rows of available dataset |

---

## 6. Prediction Registry Schema (`data/predictions/predictions.parquet`)

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `prediction_id` | `VARCHAR` | UUIDv4 unique identifier |
| `symbol` | `VARCHAR` | PSX Ticker |
| `generated_at` | `TIMESTAMP` | When model ran inference (UTC) |
| `target_date` | `DATE` | The future session date being forecasted |
| `model_name` | `VARCHAR` | Identifier of model (e.g., `xgboost_classifier`) |
| `model_version` | `VARCHAR` | Version string (e.g., `v1.2.0`) |
| `up_probability` | `DOUBLE` | Forecasted probability of upward session (0.0 to 1.0) |
| `expected_return` | `DOUBLE` | Projected return for target horizon |
| `signal` | `VARCHAR` | Discretionary signal (`STRONG_BUY`, `BUY`, `HOLD`, `SELL`) |
| `confidence` | `DOUBLE` | Uncertainty estimate (0.0 to 1.0) |
| `feature_version` | `VARCHAR` | Version/hash of the feature vector used |
| `drivers_json` | `VARCHAR` | JSON-encoded top feature attributions (SHAP) |
| `realized_outcome` | `DOUBLE` | Null until target date concludes, then updated with actual return |
| `is_correct` | `BOOLEAN` | Null until target date concludes, then evaluated |

---

## 7. News & Macro Schema

### News Table (`data/processed/news/processed_news.parquet`):
- `news_id`: Unique identifier (URL hash or source ID)
- `headline`, `body`, `source`
- `published_at`: Original article publication timestamp
- `collected_at`: Fetch timestamp
- `symbols`: List of extracted PSX tickers mentioned
- `event_type`: Categorized topic (`EARNINGS`, `DIVIDEND`, `REGULATORY`, etc.)
- `sentiment_score`: Normalized score between -1.0 and +1.0
- `usable_from`: Earliest market session this news affects
- `session_date`: Market session date

### Macro Table (`data/processed/macro/macro_indicators.parquet`):
- `observation_date`: Period to which metric applies
- `release_date`: Exact date/time published by SBP/PBS (critical to avoid leakage)
- `indicator_name`: `SBP_POLICY_RATE`, `KIBOR_6M`, `USD_PKR`, `CPI_YOY`, `BRENT_OIL`
- `value`: Numeric reading
