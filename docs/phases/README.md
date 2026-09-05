# PSX Stock Market Prediction Platform — Phase Specifications Index

This directory contains the detailed engineering specifications, test plans, and completion criteria for each of the 21 development phases.

## Phase Execution Rules
1. **Strict Sequential Progression:** Never start Phase $N+1$ until Phase $N$ passes all tests and linting, and produces its completion report.
2. **Build → Test → Validate → Document → Proceed:** Every phase must be independently testable and reviewable.
3. **No Code Without Tests:** Every feature and calculation must have accompanying unit and integration tests.

---

## Phase Matrix

| Phase | Specification Document | Walkthrough Document | Key Objective | Output Artifacts |
| :--- | :--- | :--- | :--- | :--- |
| **00** | [phase_00_bootstrap.md](phase_00_bootstrap.md) | [phase_00_walkthrough.md](walkthroughs/phase_00_walkthrough.md) | Repository & Tooling Bootstrap | `pyproject.toml`, `config/stocks.yaml`, CLI `psx config show` |
| **01** | [phase_01_storage.md](phase_01_storage.md) | [phase_01_walkthrough.md](walkthroughs/phase_01_walkthrough.md) | Local Storage & DuckDB Catalog | Parquet I/O, DuckDB connection, Atomic writes, Schema validation |
| **02** | [phase_02_collector.md](phase_02_collector.md) | *Pending* | Historical Price Collector Interface | `BaseCollector`, Yahoo Finance (`.KA`) & DPS collectors, retry logic |
| **03** | [phase_03_data_bootstrap.md](file:///d:/Development/PSX_Prediction/docs/phases/phase_03_data_bootstrap.md) | 5-Year Historical Data Ingestion | Multi-year Parquet dataset, Data quality report |
| **04** | [phase_04_incremental_updates.md](file:///d:/Development/PSX_Prediction/docs/phases/phase_04_incremental_updates.md) | Incremental Daily Ingestion | Idempotent daily updates, Latest-date detection |
| **05** | [phase_05_price_features.md](file:///d:/Development/PSX_Prediction/docs/phases/phase_05_price_features.md) | Deterministic Price Features | Returns, MAs, RSI, MACD, Bollinger, ATR, Volume indicators |
| **06** | [phase_06_prediction_targets.md](file:///d:/Development/PSX_Prediction/docs/phases/phase_06_prediction_targets.md) | Supervised Prediction Targets | Next-day direction (binary/3-class), 5-day return, Zero-leakage tests |
| **07** | [phase_07_baseline_models.md](file:///d:/Development/PSX_Prediction/docs/phases/phase_07_baseline_models.md) | Baseline Models & Linear Models | Majority, Naive, SMA crossover, Logistic Regression with chrono-split |
| **08** | [phase_08_walk_forward_backtester.md](file:///d:/Development/PSX_Prediction/docs/phases/phase_08_walk_forward_backtester.md) | Walk-Forward Backtesting Engine | Rolling/expanding splits, PSX friction (0.15%), Circuit breaker rules |
| **09** | [phase_09_tree_models.md](file:///d:/Development/PSX_Prediction/docs/phases/phase_09_tree_models.md) | Tree-Based Machine Learning | Random Forest, XGBoost, LightGBM with metric comparisons |
| **10** | [phase_10_prediction_registry.md](file:///d:/Development/PSX_Prediction/docs/phases/phase_10_prediction_registry.md) | Prediction Registry & Audit Log | `predictions.parquet`, calibration & realization tracking |
| **11** | [phase_11_news_collection.md](file:///d:/Development/PSX_Prediction/docs/phases/phase_11_news_collection.md) | News & Announcement Collectors | DPS announcements & financial RSS feeds, raw persistence |
| **12** | [phase_12_news_classification.md](file:///d:/Development/PSX_Prediction/docs/phases/phase_12_news_classification.md) | News NLP & Sentiment Analysis | Symbol matching, sentiment scoring (-1 to +1), event tagging |
| **13** | [phase_13_news_market_alignment.md](file:///d:/Development/PSX_Prediction/docs/phases/phase_13_news_market_alignment.md) | News Market Session Alignment | PSX trading hours & Friday split schedule, `usable_from` alignment |
| **14** | [phase_14_macro_data.md](file:///d:/Development/PSX_Prediction/docs/phases/phase_14_macro_data.md) | Macroeconomic Data Ingestion | USD/PKR, SBP Policy Rate, KIBOR, CPI, Brent crude oil |
| **15** | [phase_15_combined_features.md](file:///d:/Development/PSX_Prediction/docs/phases/phase_15_combined_features.md) | Multi-Modal Feature Store | Point-in-time joins of price + news + macro, ablation benchmarks |
| **16** | [phase_16_ensemble_model.md](file:///d:/Development/PSX_Prediction/docs/phases/phase_16_ensemble_model.md) | Multi-Model Ensemble | Stacking meta-learner combining specialized sub-models |
| **17** | [phase_17_fastapi_backend.md](file:///d:/Development/PSX_Prediction/docs/phases/phase_17_fastapi_backend.md) | FastAPI REST Service | Typed endpoints for stocks, history, features, predictions, backtests |
| **18** | [phase_18_streamlit_dashboard.md](file:///d:/Development/PSX_Prediction/docs/phases/phase_18_streamlit_dashboard.md) | Streamlit Research Dashboard | Interactive charts, forecast visualizer with SHAP, backtest reports |
| **19** | [phase_19_scheduler.md](file:///d:/Development/PSX_Prediction/docs/phases/phase_19_scheduler.md) | Scheduled Automation | Automated daily EOD ingestion, feature updates, and inference |
| **20** | [phase_20_monitoring.md](file:///d:/Development/PSX_Prediction/docs/phases/phase_20_monitoring.md) | Monitoring & Drift Detection | Forecast calibration tracking, Brier scores, feature drift detection |
