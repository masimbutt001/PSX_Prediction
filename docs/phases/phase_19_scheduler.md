# Phase 19 — Daily Scheduled Automation

## 1. Objective
Add automated execution for daily operational routines: End-of-Day (EOD) market data ingestion, news collection, macro update, feature regeneration, daily inference generation, and prediction registry logging.

---

## 2. Deliverables & Files to Create
```text
src/psx_predictor/scheduler/
├── __init__.py
├── runner.py                          # End-to-end daily pipeline orchestrator
└── jobs.py                            # Task definitions and cron trigger mappings

tests/unit/
└── test_scheduler.py                  # Job sequence and failure tolerance tests
```

---

## 3. Detailed Specifications

### 3.1 Daily EOD Routine Sequence
Triggered Monday–Thursday at 16:00 PKT and Friday at 17:00 PKT:
1. `psx data update` (Fetch today's closing prices)
2. `psx news fetch && psx news process` (Fetch latest announcements and score sentiment)
3. `psx macro update` (Sync USD/PKR, oil prices, policy rates)
4. `psx features build` (Regenerate latest feature vectors)
5. `psx predict --all` (Run model inference and append to prediction registry)
6. `psx predict audit` (Reconcile yesterday's predictions against today's realized returns)

### 3.2 Robustness
- Simple local scheduler (APScheduler or standard crontab wrapper).
- Retries with exponential backoff on transient network failures.
- Logs daily pipeline execution report to `data/pipeline.log`.

---

## 4. Testing Plan
- `test_eod_pipeline_order()`: Mocks sub-commands and asserts execution runs in strict topological sequence.
- `test_step_failure_logs_and_alerts()`: Asserts that if step 2 fails, error is captured without leaving unhandled exceptions.

---

## 5. Validation Commands
```bash
pytest tests/unit/test_scheduler.py -v
psx schedule run-daily --dry-run
ruff check .
```

---

## 6. Phase Completion Exit Criteria
- [ ] End-to-end daily pipeline orchestrator implemented.
- [ ] Automated execution tested in dry-run mode.
- [ ] Phase 19 completion report documented.
