# Phase 07 — Baseline Models & Linear Classifiers

## 1. Objective
Implement standard baseline benchmarks and a regularized linear model evaluated on strict chronological splits. Determine whether the engineered features contain genuine predictive signal above naive heuristics.

---

## 2. Deliverables & Files to Create
```text
src/psx_predictor/models/
├── __init__.py
├── base.py                            # BaseModel abstract class
├── baselines.py                       # Majority, Naive, and SMA crossover baselines
└── linear.py                          # Logistic Regression / Ridge classifier

tests/unit/
└── test_baselines.py                  # Baseline performance and split validation
```

---

## 3. Detailed Specifications

### 3.1 Model Implementations
1. **Majority Class Classifier:**
   - Always predicts the dominant class from the training set.
2. **Naive Persistence Baseline:**
   - Predicts tomorrow will move in the same direction as today: $\hat{y}_{t+1} = \text{sign}(r_{1d, t})$.
3. **SMA Crossover Strategy Baseline:**
   - Predicts Up when SMA(20) > SMA(50), else Down.
4. **Logistic Regression (L2 Regularized):**
   - Fits on $Z$-score standardized training features.
   - Standardizer is fit **only on training data** and used to transform test data.

### 3.2 Evaluation Metrics
- Accuracy, Precision, Recall, Macro F1-score
- ROC-AUC & Brier Score (probability calibration)
- Confusion Matrix

---

## 4. Testing Plan
- `test_majority_classifier_predicts_mode()`: Verifies majority class matches training mode.
- `test_scaler_no_data_leakage()`: Ensures feature scaler mean/std are calculated exclusively from the training partition.
- `test_model_produces_valid_probabilities()`: Ensures output probabilities sum to 1.0 and lie in $[0, 1]$.

---

## 5. Validation Commands
```bash
pytest tests/unit/test_baselines.py -v
psx model train --model baselines --symbol OGDC
ruff check .
```

---

## 6. Phase Completion Exit Criteria
- [ ] All baselines and logistic regression models operational.
- [ ] Chronological split evaluation table printed to console.
- [ ] Phase 7 completion report documented.
