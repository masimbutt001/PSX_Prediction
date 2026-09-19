# PSX Stock Market Prediction Platform — Phase Specifications Index

This directory contains the detailed engineering specifications, test plans, and completion criteria for each of the 21 development phases.

## Phase Execution Rules
1. **Strict Sequential Progression:** Never start Phase $N+1$ until Phase $N$ passes all tests and linting, and produces its completion report.
2. **Build → Test → Validate → Document → Proceed:** Every phase must be independently testable and reviewable.
3. **No Code Without Tests:** Every feature and calculation must have accompanying unit and integration tests.

---

## Phase Matrix

| Phase | Specification Document | Walkthrough Document | Key Objective | Status |
| :--- | :--- | :--- | :--- | :--- |
| **00** | [phase_00_bootstrap.md](phase_00_bootstrap.md) | [phase_00_walkthrough.md](walkthroughs/phase_00_walkthrough.md) | Repository & Tooling Bootstrap | Completed |
| **01** | [phase_01_storage.md](phase_01_storage.md) | [phase_01_walkthrough.md](walkthroughs/phase_01_walkthrough.md) | Local Storage & DuckDB Catalog | Completed |
| **02** | [phase_02_collector.md](phase_02_collector.md) | [phase_02_walkthrough.md](walkthroughs/phase_02_walkthrough.md) | Historical Price Collector Interface | Completed |
| **03** | [phase_03_data_bootstrap.md](phase_03_data_bootstrap.md) | [phase_03_walkthrough.md](walkthroughs/phase_03_walkthrough.md) | 5-Year Historical Data Ingestion | Completed |
| **04** | [phase_04_incremental_updates.md](phase_04_incremental_updates.md) | [phase_04_walkthrough.md](walkthroughs/phase_04_walkthrough.md) | Incremental Daily Ingestion | Completed |
| **05** | [phase_05_price_features.md](phase_05_price_features.md) | [phase_05_walkthrough.md](walkthroughs/phase_05_walkthrough.md) | Deterministic Price Features | Completed |
| **06** | [phase_06_prediction_targets.md](phase_06_prediction_targets.md) | [phase_06_walkthrough.md](walkthroughs/phase_06_walkthrough.md) | Supervised Prediction Targets | Completed |
| **07** | [phase_07_baseline_models.md](phase_07_baseline_models.md) | [phase_07_walkthrough.md](walkthroughs/phase_07_walkthrough.md) | Baseline Models & Linear Models | Completed |
| **08** | [phase_08_walk_forward_backtester.md](phase_08_walk_forward_backtester.md) | [phase_08_walk_forward_backtester.md](walkthroughs/phase_08_walkthrough.md) | Walk-Forward Backtesting Engine | Completed |
| **09** | [phase_09_tree_models.md](phase_09_tree_models.md) | [phase_09_walkthrough.md](walkthroughs/phase_09_walkthrough.md) | Tree-Based Machine Learning | Completed |
| **10** | [phase_10_prediction_registry.md](phase_10_prediction_registry.md) | [phase_10_walkthrough.md](walkthroughs/phase_10_walkthrough.md) | Prediction Registry & Audit Log | Completed |
| **11** | [phase_11_news_collection.md](phase_11_news_collection.md) | [phase_11_walkthrough.md](walkthroughs/phase_11_walkthrough.md) | News & Announcement Collectors | Completed |
| **12** | [phase_12_news_classification.md](phase_12_news_classification.md) | [phase_12_walkthrough.md](walkthroughs/phase_12_walkthrough.md) | News NLP & Sentiment Analysis | Completed |
| **13** | [phase_13_news_market_alignment.md](phase_13_news_market_alignment.md) | [phase_13_walkthrough.md](walkthroughs/phase_13_walkthrough.md) | News Market Session Alignment | Completed |
| **14** | [phase_14_macro_data.md](phase_14_macro_data.md) | [phase_14_walkthrough.md](walkthroughs/phase_14_walkthrough.md) | Macroeconomic Data Ingestion | Completed |
| **15** | [phase_15_combined_features.md](phase_15_combined_features.md) | [phase_15_walkthrough.md](walkthroughs/phase_15_walkthrough.md) | Multi-Modal Feature Store | Completed |
| **16** | [phase_16_ensemble_model.md](phase_16_ensemble_model.md) | [phase_16_walkthrough.md](walkthroughs/phase_16_walkthrough.md) | Multi-Model Ensemble | Completed |
| **17** | [phase_17_fastapi_backend.md](phase_17_fastapi_backend.md) | [phase_17_walkthrough.md](walkthroughs/phase_17_walkthrough.md) | FastAPI REST Service | Completed |
| **18** | [phase_18_streamlit_dashboard.md](phase_18_streamlit_dashboard.md) | [phase_18_walkthrough.md](walkthroughs/phase_18_walkthrough.md) | Streamlit Research Dashboard | Completed |
| **19** | [phase_19_scheduler.md](phase_19_scheduler.md) | [phase_19_walkthrough.md](walkthroughs/phase_19_walkthrough.md) | Scheduled Automation | Completed |
| **20** | [phase_20_monitoring.md](phase_20_monitoring.md) | [phase_20_walkthrough.md](walkthroughs/phase_20_walkthrough.md) | Monitoring & Drift Detection | Completed |
