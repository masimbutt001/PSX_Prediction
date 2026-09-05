# Phase 09 — Tree-Based Machine Learning Models

## 1. Objective
Implement non-linear tabular machine learning models (Random Forest and XGBoost) within the walk-forward evaluation framework. Generate an automated benchmark comparison table against baselines and buy-and-hold.

---

## 2. Deliverables & Files to Create
```text
src/psx_predictor/models/
├── trees.py                           # Random Forest and XGBoost model implementations
└── comparator.py                      # Multi-model benchmarking table generator

tests/unit/
└── test_tree_models.py                # Model training, hyperparameter parsing, and evaluation tests
```

---

## 3. Detailed Specifications

### 3.1 Model Implementations
1. **Random Forest Classifier / Regressor (`sklearn.ensemble.RandomForestClassifier`):**
   - Controlled complexity: `n_estimators=100`, `max_depth=5`, `min_samples_leaf=20`.
   - Feature importances extraction.
2. **XGBoost Classifier / Regressor (`xgboost.XGBClassifier`):**
   - Binary objective: `binary:logistic`.
   - 3-class objective: `multi:softprob`.
   - Regularized hyperparameters: `learning_rate=0.03`, `max_depth=4`, `subsample=0.8`, `colsample_bytree=0.8`.
   - Early stopping on internal chronological validation split.

### 3.2 Benchmark Comparison Table
- Compares on the identical walk-forward test folds:
  ```text
  Model               Accuracy   F1-Score   ROC-AUC    Sharpe   Total Return   Max DD
  Majority Baseline   ...        ...        ...        ...      ...            ...
  SMA Crossover       ...        ...        ...        ...      ...            ...
  Logistic Regression ...        ...        ...        ...      ...            ...
  Random Forest       ...        ...        ...        ...      ...            ...
  XGBoost             ...        ...        ...        ...      ...            ...
  ```

---

## 4. Testing Plan
- `test_xgboost_fit_predict()`: Verifies training, probability prediction, and reproducibility with fixed seed.
- `test_early_stopping_respects_chronology()`: Validates internal validation split is strictly posterior to training split.
- `test_benchmark_comparator_output()`: Asserts comparator outputs all expected metric columns.

---

## 5. Validation Commands
```bash
pytest tests/unit/test_tree_models.py -v
psx model train --model xgboost --symbol OGDC
psx model compare --symbol OGDC
ruff check .
```

---

## 6. Phase Completion Exit Criteria
- [ ] Random Forest and XGBoost models trained and evaluated.
- [ ] Comparison table generated showing performance vs. baselines.
- [ ] Serialized models saved to `data/models/`.
- [ ] Phase 9 completion report documented.
