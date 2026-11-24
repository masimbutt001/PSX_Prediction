# Pakistan Stock Exchange (PSX) Prediction Platform

[![Python 3.12](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-179%20passed-brightgreen.svg)]()
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: Mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)

An institutional-grade, end-to-end quantitative analytics, machine learning, and automated predictive modeling platform designed specifically for the **Pakistan Stock Exchange (PSX)**.

The platform explicitly incorporates local market microstructures (circuit breaker locks at $\pm7.5\%$, Friday split prayer sessions, high-dividend corporate actions, and transaction friction), combining historical price action, multi-modal feature engineering, FinBERT financial sentiment NLP, macroeconomic indicator synchronization, and a stacked meta-learner ensemble.

---

## Table of Contents
1. [Key Architecture Highlights](#key-architecture-highlights)
2. [End-to-End Platform Architecture](#end-to-end-platform-architecture)
3. [Quick Start & Installation](#quick-start--installation)
4. [Complete CLI Command Reference](#complete-cli-command-reference)
   - [Configuration & Inspection](#1-configuration--inspection-psx-config)
   - [Data Ingestion & Quality](#2-data-ingestion--quality-psx-data)
   - [Feature Engineering & Multi-Modal Merger](#3-feature-engineering--multi-modal-merger-psx-features)
   - [Machine Learning & Backtesting](#4-machine-learning--backtesting-psx-model)
   - [Live Forecasts & Audit Trail](#5-live-forecasts--audit-trail-psx-predict)
   - [Financial News & NLP](#6-financial-news--nlp-psx-news)
   - [Macroeconomic Indicators](#7-macroeconomic-indicators-psx-macro)
   - [FastAPI REST Backend](#8-fastapi-rest-backend-psx-api)
   - [Streamlit Research Dashboard](#9-streamlit-research-dashboard-psx-dashboard)
   - [Scheduled Automation & Daemon](#10-scheduled-automation--daemon-psx-schedule)
   - [Model Monitoring & Drift Detection](#11-model-monitoring--drift-detection-psx-health)
5. [Interactive Streamlit Dashboard Guide](#interactive-streamlit-dashboard-guide)
6. [FastAPI REST Backend API Guide](#fastapi-rest-backend-api-guide)
7. [Automated Daily Operational Routine](#automated-daily-operational-routine)
8. [Testing & Static Analysis](#testing--static-analysis)
9. [Project Directory Layout](#project-directory-layout)

---

## Key Architecture Highlights

- **PSX Microstructure Fidelity:** Explicit enforcement of $\pm7.5\%$ statutory circuit breaker price bands, high-dividend cash/bonus adjustments, realistic Pakistani broker commission structures (0.15% fee/commission model with regulatory turnover levies), and Friday split session timings.
- **Local-First Analytical Engine:** High-performance columnar storage using Apache Parquet with atomic write swaps and in-process DuckDB analytical SQL execution.
- **Zero Look-Ahead Bias:** Strict point-in-time chronological joins for all technical, news sentiment, and macroeconomic variables.
- **Multi-Modal Feature Fusion:** Dynamic point-in-time fusion of 25+ technical indicators, FinBERT NLP sentiment scores, corporate announcement categories, USD/PKR exchange rates, SBP policy rates, KIBOR, and Brent crude oil.
- **Stacking Meta-Learner Ensemble:** Out-of-fold probability weight optimization blending specialized base classifiers (Random Forest, LightGBM, XGBoost, and Logistic Regression).
- **Automated Operational Daemon:** Session-aligned APScheduler cron triggers configured for Pakistan Standard Time (`Asia/Karachi`):
  - **Monday – Thursday:** Market close at 15:30 PKT $\to$ Automated pipeline runs at 16:00 PKT.
  - **Friday:** Market close at 16:30 PKT $\to$ Automated pipeline runs at 17:00 PKT.
- **Continuous Reliability & Drift Monitoring:** Real-time Population Stability Index (PSI) drift tracking with regulatory thresholds and rolling 30-day / 90-day probability verification Brier scoring.

---

## End-to-End Platform Architecture

```mermaid
flowchart TD
    subgraph Data Layer
        A1[Yahoo Finance / DPS Scraping] --> B1[(Raw Parquet Data)]
        A2[Business Recorder / Dawn RSS] --> B2[(Raw News Parquet)]
        A3[SBP / FRED Macro Data] --> B3[(Raw Macro Parquet)]
    end

    subgraph Feature Engineering
        B1 --> C1[Technical Indicators<br/>RSI, MACD, BB, ATR, MAs]
        B2 --> C2[FinBERT NLP<br/>Sentiment & Entity Extraction]
        B3 --> C3[Macro Ingestion<br/>USD/PKR, Brent, KIBOR]
        C1 & C2 & C3 --> C4[MultiModalFeatureMerger<br/>Point-in-Time Fusion]
    end

    subgraph Machine Learning Core
        C4 --> D1[Walk-Forward Splitter]
        D1 --> D2[Base Models<br/>RF, XGBoost, LightGBM]
        D2 --> D3[Stacking Ensemble<br/>Out-of-Fold Optimizer]
    end

    subgraph Production & Serving
        D3 --> E1[(Prediction Registry<br/>predictions.parquet)]
        E1 --> F1[FastAPI REST Backend<br/>HTTP :8000]
        E1 --> F2[Streamlit Dashboard<br/>HTTP :8501]
        E1 --> F3[System Health & Drift<br/>PSI & Brier Verification]
    end

    subgraph Orchestration
        Sched[APScheduler Cron Daemon] -->|Mon-Thu 16:00 / Fri 17:00 PKT| Pipe[DailyPipelineRunner]
        Pipe --> A1 & A2 & A3 & C4 & D3 & E1
    end
```

---

## Quick Start & Installation

### Prerequisites
- Python $\ge$ 3.12
- [uv](https://github.com/astral-sh/uv) (strongly recommended) or standard `pip`

### Step 1: Clone Repository
```bash
git clone https://github.com/masimbutt001/PSX_Prediction.git
cd PSX_Prediction
```

### Step 2: Set Up Virtual Environment & Dependencies
Using `uv`:
```bash
# Automatically creates .venv and installs all production & development dependencies
uv sync
```

Or using standard `pip`:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -e ".[dev]"
```

### Step 3: Verify Installation
```bash
uv run psx version
uv run psx config show
```

---

## Complete CLI Command Reference

The command-line interface `psx` is built with [Typer](https://typer.tiangolo.com/) and provides rich, colorized terminal outputs across 11 functional subsystems.

### 1. Configuration & Inspection (`psx config`)
```bash
# Display active settings, directories, and configured PSX stock universe
uv run psx config show

# Use a custom configuration directory
uv run psx config show --config-dir /path/to/config
```

### 2. Data Ingestion & Quality (`psx data`)
```bash
# Bootstrap 5 years of historical OHLCV data for all enabled stocks
uv run psx data bootstrap

# Bootstrap specific symbols with custom lookback years
uv run psx data bootstrap --symbols OGDC,HBL,LUCK --years 3

# Execute daily incremental price ingestion
uv run psx data update

# Update a subset of symbols
uv run psx data update --symbols SYS,ENGROH

# Inspect data quality and identify missing sessions or anomalies
uv run psx data quality
```

### 3. Feature Engineering & Multi-Modal Merger (`psx features`)
```bash
# Compute technical indicators (RSI, MACD, Bollinger Bands, ATR, MAs)
uv run psx features build

# Build technical indicators for specific symbols
uv run psx features build --symbols OGDC,PPL

# Fuse technicals, news sentiment, and macro indicators into combined feature matrices
uv run psx features merge

# Merge specific symbols
uv run psx features merge --symbols OGDC,HBL
```

### 4. Machine Learning & Backtesting (`psx model`)
```bash
# Train predictive models (baseline, tree models, or stacking ensemble)
uv run psx model train --symbol OGDC --model-type stacking

# Available model types: baseline, rf, lgbm, xgboost, stacking
uv run psx model train --symbol HBL --model-type lgbm

# Run a walk-forward chronological backtest with 0.15% friction
uv run psx model backtest --symbol OGDC --strategy momentum

# Backtest a moving average crossover strategy
uv run psx model backtest --symbol LUCK --strategy macrossover --short-window 10 --long-window 30
```

### 5. Live Forecasts & Audit Trail (`psx predict`)
```bash
# Generate out-of-sample forward prediction for a stock and log to registry
uv run psx predict generate --symbol OGDC

# Reconcile historical forecasts against realized market closes (hit rate & Brier score)
uv run psx predict audit

# Audit a specific symbol's predictions
uv run psx predict audit --symbol OGDC
```

### 6. Financial News & NLP (`psx news`)
```bash
# Scrape financial RSS news (Business Recorder, Dawn) and DPS corporate announcements
uv run psx news collect

# Collect DPS announcements only
uv run psx news collect --source dps --count 50

# Run FinBERT sentiment analysis and entity extraction on collected news
uv run psx news process

# Run news extraction in dry-run mode
uv run psx news process --dry-run
```

### 7. Macroeconomic Indicators (`psx macro`)
```bash
# Synchronize USD/PKR, Brent crude, KIBOR, and SBP monetary policy rates
uv run psx macro update

# Inspect latest macroeconomic data series
uv run psx macro show
```

SBP policy rates and PBS CPI releases are read from `config/macro_rates.yaml`. Add a line there
whenever SBP announces a decision or PBS publishes CPI. 6-month KIBOR is currently estimated as
the policy rate plus the spread set in that file. USD/PKR, Brent and KSE-100 come from Yahoo
Finance; if Yahoo fails, the previously saved values are kept and the macro step is reported as
failed (no synthetic data is generated).

### 8. FastAPI REST Backend (`psx api`)
```bash
# Start the production REST API server (default: http://127.0.0.1:8000)
uv run psx api start

# Run on custom host/port with hot-reload enabled
uv run psx api start --host 0.0.0.0 --port 8080 --reload

# Validate API server configuration without binding ports
uv run psx api start --dry-run
```

### 9. Streamlit Research Dashboard (`psx dashboard`)
```bash
# Launch interactive Streamlit research dashboard (default: http://127.0.0.1:8501)
uv run psx dashboard

# Launch on custom port
uv run psx dashboard --port 8502

# Dry-run validation check
uv run psx dashboard --dry-run
```

### 10. Scheduled Automation & Daemon (`psx schedule`)
```bash
# Inspect automated cron triggers, PSX market session timings, and run history
uv run psx schedule status

# Run the 6-step EOD operational pipeline immediately in dry-run mode
uv run psx schedule run-daily --dry-run

# Run the full live EOD pipeline immediately (all enabled symbols)
uv run psx schedule run-daily

# Run the live pipeline for specific symbols, continuing if non-critical steps fail
uv run psx schedule run-daily --symbols OGDC,HBL --continue-on-error

# Command for Windows Task Scheduler / cron: does nothing on weekends and PSX holidays
uv run psx schedule run-daily --continue-on-error --skip-non-trading-days

# Launch the long-running APScheduler process in foreground blocking mode
uv run psx schedule start

# Launch scheduler daemon in background non-blocking mode
uv run psx schedule start --daemon
```

### 11. Model Monitoring & Drift Detection (`psx health`)
```bash
# Run comprehensive platform diagnostic audit (data feeds, calibration, and drift)
uv run psx health check

# Check health for a subset of symbols with detailed reporting
uv run psx health check --symbols OGDC,HBL,SYS --detailed

# Inspect Population Stability Index (PSI) and KS-test drift for a specific stock
uv run psx health drift --symbol OGDC

# Configure custom lookback evaluation windows (trailing sessions vs baseline)
uv run psx health drift --symbol OGDC --recent-sessions 60 --reference-sessions 250
```

---

## Interactive Streamlit Dashboard Guide

The dashboard is a Streamlit app with top navigation, a dark and a light theme, and interactive Plotly charts.

### Launching the Dashboard
Run it from the repository root, so Streamlit picks up the theme in `.streamlit/config.toml` and the app finds `data/`:
```bash
uv run psx dashboard
# Alternatively:
uv run streamlit run src/psx_predictor/dashboard/app.py
```
Open your browser at `http://localhost:8501`. Switch between light and dark from the app menu: ⋮ > Settings > Theme.

### Dashboard Pages

```
src/psx_predictor/dashboard/
├── app.py              -> Top navigation, stock search, data freshness, refresh
├── views/overview.py   -> KPIs (KSE-100, signals, hit rate, Brier, USD/PKR), watchlist with
│                          sector and signal filters, 30-session trend, CSV export,
│                          sector moves, last daily run, latest news
├── views/stock.py      -> Stock switcher, candlestick chart (1M-All, SMA 20/50, Bollinger,
│                          volume, next-session marker), prediction card with drivers,
│                          prediction history, news and research tabs
├── views/performance.py-> Live track record (hit rate, Brier, calibration chart, per stock)
│                          and the strategy backtester
├── views/research.py   -> Stock research files, score history, catalysts, risks, sources
└── views/pipeline.py   -> Daily run steps and errors, run history, price freshness per stock
```

---

## Stock Research (Google Antigravity)

Research is written by an AI agent, not by this app. Paste the prompt in
[`docs/research/antigravity_research_prompt.md`](docs/research/antigravity_research_prompt.md)
into Google Antigravity with this folder open. For each stock it writes
`data/research/<SYMBOL>_<YYYY-MM-DD>.json` (strict schema, version 1.0) and a matching `.md`
report. `data/research/` is gitignored and stays on your machine.

```bash
uv run psx research list                 # all research files with key scores
uv run psx research validate             # schema check; exit code 1 if any file is invalid
uv run psx research show --symbol OGDC   # latest research summary
```

Narrated presentation videos (broker research note style) can be made from each report with
[`docs/research/antigravity_video_prompt.md`](docs/research/antigravity_video_prompt.md). They are
saved to `data/research/videos/` (also gitignored), with an `index.json` listing every video.

How research reaches predictions:
- `psx features merge` and `psx predict` add `research_*` features (sentiment, outlook, stance,
  fundamentals, valuation, risk, upside, days to next event).
- Point in time: a session only sees research dated on or before it, so backtests have no look-ahead.
- Recency decay: influence halves every 30 days, scaled by the research's own confidence, and
  stops after 120 days. Without research the features are neutral (0).
- Each prediction stores the research snapshot it used (`research_json`), shown on the Stock
  Analysis and Research pages.
- The model only learns from research once there is research history across many dates, so
  run research regularly (for example weekly) for it to carry weight.

---

## FastAPI REST Backend API Guide

The platform provides a high-performance, asynchronous REST backend built with FastAPI and Pydantic v2.

### Starting the API Server
```bash
uv run psx api start --port 8000
```
Visit the interactive Swagger UI at **`http://127.0.0.1:8000/docs`**.

### Key REST Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Service health status, UTC timestamp, and version. |
| `GET` | `/api/v1/stocks` | List configured universe with sector, enabled status, and Parquet data availability. |
| `GET` | `/api/v1/stocks/{symbol}/history` | Historical OHLCV prices with optional `start_date` and `end_date` filters. |
| `GET` | `/api/v1/predictions/latest` | Latest forward forecasts across the entire universe or filtered by `?symbol=`. |
| `GET` | `/api/v1/predictions/audit` | Historical prediction outcome audit trail, realized returns, and Brier score. |
| `POST` | `/api/v1/backtest/run` | Execute backtest simulation via JSON payload with strategy parameters. |

#### Sample Request: Run Backtest via API
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/backtest/run" \
     -H "Content-Type: application/json" \
     -d '{
       "symbol": "OGDC",
       "strategy": "macrossover",
       "initial_capital": 1000000,
       "parameters": {"short_window": 10, "long_window": 30}
     }'
```

---

## Automated Daily Operational Routine

Everything runs on your own computer. Git holds source code only: all data stays in the local
`data/` folder, which is not committed.

### One-time setup on Windows 11 (Task Scheduler)
From the repository folder in PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_laptop_schedule_windows.ps1
```
This registers a "PSX Daily Pipeline" task that checks every hour and at log-on. Each check runs
`psx schedule run-daily --catch-up`, which does the work once per PSX trading day after market
close (16:00 PKT Mon-Thu, 17:00 PKT Fri), skips holidays, and retries a failed day up to three
times. If the laptop was off or asleep at 16:00, the run happens at the next check.
Output is written to `data\scheduler.out.log`; each run is also recorded in `data\pipeline.log`.
To remove the task, run the same script with `-Uninstall`.

On macOS the equivalent is `bash scripts/install_laptop_schedule_macos.sh` (launchd).

### How a run works

The automated operational routine runs on APScheduler aligned with PSX post-market sessions in the `Asia/Karachi` timezone:

1. **Topological Pipeline Sequence (Step 1 to 6)**:
   - **Step 1: `UPDATE_PRICES`**: Fetches today's closing prices for all enabled stocks.
   - **Step 2: `COLLECT_NEWS`**: Ingests new articles via RSS & corporate announcements via DPS, scoring sentiment with FinBERT.
   - **Step 3: `UPDATE_MACRO`**: Synchronizes USD/PKR, Brent crude, KIBOR, and policy rates.
   - **Step 4: `BUILD_FEATURES`**: Generates updated technical indicators and merged multi-modal feature matrices.
   - **Step 5: `GENERATE_PREDICTIONS`**: Runs stacking ensemble forward inference across active universe and logs to registry.
   - **Step 6: `AUDIT_PREDICTIONS`**: Reconciles yesterday's predictions against today's realized returns and evaluates Brier score.
2. **Audit Logging**:
   - Every execution atomically appends a detailed execution report to `data/pipeline.log` in JSON Lines format.
3. **Resilience & Fault Tolerance**:
   - All external network actions utilize exponential backoff retries.
   - The `--continue-on-error` flag allows downstream audit and monitoring tasks to proceed even if an upstream scraper experiences a transient network error.

---

## Testing & Static Analysis

The project enforces a strict policy of **100% deterministic, offline-capable unit and integration testing** with zero tolerance for lint errors or type violations.

```bash
# Run complete test suite (179 tests across 21 modules)
uv run pytest -v

# Run linting and style checks with Ruff
uv run ruff check .

# Run strict static type checking with Mypy
uv run mypy src
```

---

## Project Directory Layout

```
PSX_Prediction/
├── config/                         # Application & stock configuration
│   ├── settings.yaml               # System parameters, paths, and timezones
│   └── stocks.yaml                 # Configured PSX universe (OGDC, HBL, LUCK, etc.)
├── data/                           # Local-first Parquet analytical storage
│   ├── raw/                        # Raw scraped prices, news, and macro series
│   ├── processed/                  # Cleaned, split-adjusted OHLCV price series
│   ├── features/                   # Technical, news NLP, and merged feature stores
│   ├── predictions/                # Permanent append-only predictions.parquet registry
│   └── pipeline.log                # Scheduled EOD operational execution audit trail
├── docs/                           # Engineering specifications and phase walkthroughs
│   └── phases/                     # Specifications for Phases 00 through 20
│       └── walkthroughs/           # Verification and implementation reports
├── src/psx_predictor/              # Core application package
│   ├── api/                        # FastAPI REST service & Pydantic schemas
│   ├── backtest/                   # Walk-forward backtester & performance analytics
│   ├── cli/                        # Typer CLI subcommands & Rich visual formatting
│   ├── config/                     # YAML configuration loaders & validators
│   ├── dashboard/                  # Streamlit dashboard (app.py, views/, components/, theme.py)
│   ├── data/                       # Market data collectors (DPS, Yahoo Finance)
│   ├── features/                   # Technical indicator builders & multi-modal mergers
│   ├── macro/                      # Macroeconomic indicator ingestors
│   ├── models/                     # Base models, tree models, and stacking ensemble
│   ├── monitoring/                 # Drift detection (PSI, KS-test) & Brier calibration
│   ├── news/                       # RSS collectors, DPS announcements & FinBERT NLP
│   ├── predictions/                # Live prediction engine, registry & auditor
│   ├── processing/                 # Corporate actions, calendar sessions & incremental updater
│   ├── scheduler/                  # APScheduler EOD runner, cron triggers & jobs
│   └── storage/                    # Atomic Parquet I/O & DuckDB catalog
├── tests/                          # Complete automated test suite (179 tests)
│   ├── unit/                       # Unit tests for all individual subsystems
│   └── integration/                # End-to-end multi-step integration tests
├── pyproject.toml                  # Project metadata, dependencies, and tooling configuration
└── README.md                       # Comprehensive platform documentation and run guide
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
