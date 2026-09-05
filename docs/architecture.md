# System Architecture — PSX Stock Market Prediction Platform

## 1. Architectural Overview

The PSX Stock Market Prediction Platform is a local-first, modular research and decision-support system designed specifically for the Pakistan Stock Exchange (PSX). It operates with strict separation between data collection, storage, feature engineering, model training, evaluation, inference, and visualization.

```text
       [ External Data Sources ]
  (PSX DPS Portal, Yahoo Finance .KA, SBP Macro, RSS News)
                     │
                     ▼
         ┌───────────────────────┐
         │   Collectors Layer    │  (Common interface, retries, rate limits)
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │    Raw Data Layer     │  (Immutable Parquet / JSON storage)
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │ Validation & Cleaning │  (OHLC checks, circuit locks, corporate actions)
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │  Processed Data Layer │  (Normalized Parquet + DuckDB Catalog)
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │  Feature Engineering  │  (Point-in-time technical, news & macro features)
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │     Feature Store     │  (Deterministic, versioned feature datasets)
         └─────┬───────────┬─────┘
               │           │
               ▼           ▼
        ┌─────────────┐ ┌──────────────┐
        │ Tabular ML  │ │   NLP News   │
        │   Models    │ │    Models    │
        └──────┬──────┘ └──────┬───────┘
               │               │
               └───────┬───────┘
                       ▼
           ┌───────────────────────┐
           │     Ensemble Layer    │
           └───────────┬───────────┘
                       ▼
           ┌───────────────────────┐
           │ Walk-Forward Backtest │  (PSX friction, circuit locks, rolling splits)
           └───────────┬───────────┘
                       ▼
           ┌───────────────────────┐
           │   Prediction Engine   │  (Permanent forecast logging & explainability)
           └───────────┬───────────┘
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
     ┌───────────────┐   ┌────────────────┐
     │  FastAPI REST │   │   Streamlit    │
     │    Backend    │   │   Dashboard    │
     └───────────────┘   └────────────────┘
```

---

## 2. Core Architectural Principles

1. **Local-First Analytical Engine:**
   - Primary storage is local **Parquet** files partitioned logically.
   - **DuckDB** acts as the high-speed in-process SQL engine querying Parquet directly without requiring a standalone database server.
   
2. **Immutable Raw Data:**
   - Data fetched from external APIs is saved untouched in `data/raw/` before any parsing, transformation, or repair.
   - Enables complete auditability and re-processing if schemas or logic change.

3. **Zero Look-Ahead Bias Guarantee:**
   - Features must only access information available at or before the close of trading session $t$.
   - News and macro data are stamped with `usable_from` session timestamps.
   - Walk-forward chronological splits are strictly enforced across all training pipelines.

4. **PSX Microstructure Fidelity:**
   - **Circuit Breaker Tracking:** Daily limit up / limit down (±7.5% or 1 PKR) prevents unrealistic fill assumptions in backtesting.
   - **Ex-Dividend / Corporate Actions:** High dividend yields in PSX require tracking both unadjusted and dividend/split-adjusted prices.
   - **Friday Session Split:** Explicit calendar accounting for Friday's prayer break (09:15–12:00 and 14:30–16:30 PKT).

---

## 3. Subsystem Breakdown

### 3.1 Data Collectors (`psx_predictor.collectors`)
- Inherits from `BaseCollector`.
- Handles connection pooling, exponential backoff, rate limiting, and timeout management.
- Abstracted from downstream consumers; changes to source APIs only affect this layer.

### 3.2 Storage & Catalog (`psx_predictor.storage`)
- Manages directory paths, atomic file writes (`.tmp` $\to$ rename), and DuckDB connection lifecycles.
- Exposes clean dataset read/write interfaces with schema verification.

### 3.3 Data Processing (`psx_predictor.processing`)
- Ingests raw data, validates against Pydantic schemas, deduplicates, and calculates adjusted close prices based on dividends and splits.
- Writes validated records to `data/processed/`.

### 3.4 Feature Engineering (`psx_predictor.features`)
- Modular feature generators:
  - `price_features.py`: Log returns, SMA/EMA, RSI, MACD, Bollinger Bands, ATR.
  - `volume_features.py`: Relative volume, volume moving averages, OBV.
  - `market_features.py`: Beta to KSE-100, relative strength.
  - `news_features.py`: Sentiment polarity, event flags, news velocity.
  - `macro_features.py`: SBP policy rate, KIBOR, inflation, Brent crude oil.

### 3.5 Modeling & Ensembling (`psx_predictor.models`)
- Baselines: Majority class, Naive persistence, SMA crossover, Logistic Regression.
- Tree Ensembles: Random Forest, XGBoost, LightGBM.
- Model registry tracking hyperparameter configs, feature versions, and serialization (`data/models/`).

### 3.6 Backtesting (`psx_predictor.backtesting`)
- Walk-forward rolling and expanding window simulator.
- PSX transaction costs: 0.15% brokerage + SECP/CDC charges.
- Circuit breaker rejection logic.

### 3.7 Prediction Registry (`psx_predictor.predictions`)
- Appends every generated forecast into `data/predictions/predictions.parquet`.
- Evaluates past predictions against real future outcomes as time moves forward.

### 3.8 Presentation Layer (`psx_predictor.api` & `psx_predictor.dashboard`)
- **FastAPI:** Clean typed REST endpoints for external consumers or frontends.
- **Streamlit:** Interactive visual dashboard for research, charts, predictions, feature explanations, and backtest diagnostics.
