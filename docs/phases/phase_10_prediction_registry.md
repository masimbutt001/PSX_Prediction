# Phase 10 — Prediction Registry & Audit Log

## 1. Objective
Establish a permanent prediction registry where every generated model forecast is appended rather than overwritten. As future trading sessions conclude, the registry matches past predictions against realized market outcomes to audit live calibration and accuracy.

---

## 2. Deliverables & Files to Create
```text
src/psx_predictor/predictions/
├── __init__.py
├── registry.py                        # Parquet prediction logger and retrieval
└── auditor.py                         # Compares past forecasts against realized outcomes

tests/unit/
└── test_predictions.py                # Logging, retrieval, and realization evaluation tests
```

---

## 3. Detailed Specifications

### 3.1 Registry Schema & Storage
- Storage path: `data/predictions/predictions.parquet`.
- Columns:
  - `prediction_id`: UUIDv4
  - `symbol`: Ticker
  - `generated_at`: UTC timestamp
  - `target_date`: Date forecasted
  - `model_name`, `model_version`
  - `up_probability`, `expected_return`, `signal`, `confidence`
  - `feature_version`, `drivers_json`
  - `realized_outcome`: Float (initially Null)
  - `is_correct`: Boolean (initially Null)

### 3.2 Audit Reconciliation
- Running `psx predict audit`:
  1. Finds records where `target_date <= today` and `realized_outcome IS NULL`.
  2. Queries processed price data for actual returns on `target_date`.
  3. Updates `realized_outcome` and `is_correct`.
  4. Calculates rolling Brier calibration score and accuracy.

---

## 4. Testing Plan
- `test_prediction_registration()`: Logs a new forecast and verifies persistence in Parquet.
- `test_auditor_reconciliation()`: Creates forecast with past target date, runs auditor, and verifies `realized_outcome` is correctly populated.

---

## 5. Validation Commands
```bash
pytest tests/unit/test_predictions.py -v
psx predict --symbol OGDC --model xgboost
psx predict audit
ruff check .
```

---

## 6. Phase Completion Exit Criteria
- [ ] Forecasts permanently logged with UUIDs and metadata.
- [ ] Audit reconciliation correctly updates realized returns.
- [ ] Phase 10 completion report documented.
