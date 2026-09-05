# Phase 20 — Model Monitoring & Drift Detection

## 1. Objective
Continuously assess real-world model reliability over time. Track rolling prediction accuracy, probability calibration (Brier score and calibration curves), feature distribution drift (Kolmogorov-Smirnov / PSI), and data quality health.

---

## 2. Deliverables & Files to Create
```text
src/psx_predictor/monitoring/
├── __init__.py
├── calibration.py                     # Brier score and reliability curve tracking
├── drift.py                           # Population Stability Index (PSI) and feature drift
└── health.py                          # Data feed quality and missing session monitoring

tests/unit/
└── test_monitoring.py                 # Drift calculation and calibration alert tests
```

---

## 3. Detailed Specifications

### 3.1 Probability Calibration Tracking
- Computes rolling 30-day and 90-day Brier scores:
  $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (P_i - y_i)^2$$
- Generates calibration bins: If a model predicts $70\%$ probability, does the stock rise $\approx 70\%$ of the time in reality?
- Flags severe miscalibration.

### 3.2 Feature Drift Detection
- Compares rolling feature distributions against baseline training distributions using Population Stability Index (PSI):
  - $\text{PSI} < 0.10$: Minimal drift.
  - $0.10 \le \text{PSI} < 0.25$: Moderate drift (warning).
  - $\text{PSI} \ge 0.25$: Significant drift (triggers recommendation for model retraining).

### 3.3 CLI Health Check
- Running `psx health check` outputs:
  - Data freshness & missing sessions
  - Current model performance vs. historical backtest metrics
  - Active drift warnings

---

## 4. Testing Plan
- `test_brier_score_computation()`: Validates Brier score math against known test distribution.
- `test_psi_detects_distributional_shift()`: Feeds shifted Gaussian series and asserts PSI triggers drift alert.

---

## 5. Validation Commands
```bash
pytest tests/unit/test_monitoring.py -v
psx health check
ruff check .
```

---

## 6. Phase Completion Exit Criteria
- [ ] Calibration and drift metrics operational.
- [ ] Health check command summarizes system status.
- [ ] Complete platform Phase 20 milestone achieved.
