# Machine Learning Modeling Methodology — PSX Stock Market Prediction Platform

## 1. Modeling Strategy

The modeling philosophy strictly adheres to an incremental, baseline-first approach:

```text
Simple Baselines ──► Linear Models ──► Tree Ensembles ──► Multi-Modal Ensembles ──► Deep Learning
(Majority / Naive)   (Logistic Reg)   (RF / XGBoost)    (Price + News + Macro)   (LSTM/Transformers)
```

We do not advance to complex architectures without proving that simpler models produce statistically meaningful edge over market baselines.

---

## 2. Strict Chronological Evaluation (No Random Splits)

Financial time-series data is non-stationary, autocorrelated, and subject to regime shifts. **Random k-fold cross-validation or random train-test splitting is strictly forbidden.**

### Validation Regimes:
1. **Expanding Window (Anchored Walk-Forward):**
   - Window 1: Train $[T_0, T_{1}]$, Test $(T_{1}, T_{2}]$
   - Window 2: Train $[T_0, T_{2}]$, Test $(T_{2}, T_{3}]$
   - Window 3: Train $[T_0, T_{3}]$, Test $(T_{3}, T_{4}]$
2. **Rolling Window (Unanchored Walk-Forward):**
   - Keeps fixed training lookback length (e.g., 3 years) and steps forward by 3 or 6 months.

```text
Fold 1: [--- Train: 3 Years ---][ Test: 3 Months ]
Fold 2:      [--- Train: 3 Years ---][ Test: 3 Months ]
Fold 3:           [--- Train: 3 Years ---][ Test: 3 Months ]
```

---

## 3. Model Progression & Specifications

### Phase 7: Baselines
1. **Majority Class Classifier:**
   - Always predicts the dominant historical direction (Up or Down). Establishes the floor metric.
2. **Naive Previous-Day Direction:**
   - $\hat{y}_t = y_{t-1}$. Tests momentum persistence.
3. **Simple Moving Average Strategy:**
   - Golden / Death Cross baseline (SMA-20 vs SMA-50).
4. **Logistic Regression (L2 Regularized):**
   - Baseline linear probabilistic classifier. Requires normalized features ($Z$-score calculated purely on training set, applied forward).

### Phase 9: Tree-Based Models
1. **Random Forest:**
   - Robust against overfitting, handles non-linear interactions without feature scaling.
   - Constrained `max_depth` (3 to 6) and `min_samples_leaf` to avoid memorization.
2. **XGBoost:**
   - Gradient boosted trees optimizing log-loss (binary) or multi-class log-loss (3-class).
   - Early stopping on chronological validation split.
   - Regularization: `colsample_bytree`, `subsample`, `reg_alpha`, `reg_lambda`.
3. **LightGBM:**
   - Fast histogram-based gradient boosting for larger feature spaces.

### Phase 16: Multi-Modal Ensemble
- Combines individual specialized learners:
  - Technical Price Model $\to$ $P_{price}$
  - News/Sentiment Model $\to$ $P_{news}$
  - Macroeconomic Model $\to$ $P_{macro}$
- Meta-learner (Logistic Regression or Stacking Classifier) with weights learned via out-of-fold walk-forward validation.

---

## 4. Targets & Feature Hygiene

### Target Horizons:
- **1-Day Horizon:** Next session direction. Sensitive to noise, tight stops.
- **5-Day Horizon:** Weekly expected return. Captures institutional swing flows.

### Look-Ahead Bias Defenses:
- **Feature Computation:** Any rolling indicator computed on date $t$ must only use data up to close of date $t$.
- **Target Exclusion:** The final row in any training partition cannot have its target computed without future data; therefore, rows where future targets overlap with the test set are strictly excluded (purging and embargoing).
- **Leakage Unit Tests:** Dedicated tests in `tests/unit/test_leakage.py` that perturb future data and assert that historical feature vectors remain bit-for-bit identical.

---

## 5. Model Explainability & Interpretability

Every production forecast must include explainable drivers:
- **SHAP (SHapley Additive exPlanations):** Extract top 3–5 positive and negative contributing features for each inference row.
- **Feature Importance Tracking:** Monitor whether models rely disproportionately on single features over time.
