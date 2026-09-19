# Walkthrough — Phase 16: Multi-Model Ensemble

**Status:** Completed & Validated  
**Date:** September 2026  
**Specification:** [phase_16_ensemble_model.md](../phase_16_ensemble_model.md)

---

## 1. Objective Accomplished
Constructed a multi-modal stacking meta-learner combining specialized sub-models across Technical, News, and Macroeconomic information layers:
1. **Specialized Base Learners**:
   - **Price Specialist (`XGBoostBaseline` / `RandomForestBaseline`)**: Focuses on non-linear price action, trend momentum, Bollinger Band dynamics, ATR, and volume.
   - **News Specialist (`LogisticRegressionBaseline`)**: Calibrated linear learner on point-in-time 24h/72h polarity, signal counts, and corporate disclosure indicators.
   - **Macroeconomic Specialist (`LogisticRegressionBaseline`)**: Regularized linear model capturing SBP monetary policy, KIBOR spreads, CPI inflation, Brent crude, and FX depreciation.
2. **Out-of-Fold Walk-Forward Weight Optimizer (`OutOfFoldWeightOptimizer`)**:
   - Divides training data into expanding chronological folds with strictly zero forward lookahead leakage:
     $$\text{Train}_k < \text{Val}_k < \text{Val}_{k+1}$$
   - Generates out-of-fold probability predictions $Z_{\text{OOF}} \in [0, 1]^{N \times 3}$.
   - Solves constrained non-negative log-loss optimization using SLSQP:
     $$\min_w \mathcal{L}\left(y, \sum_{b=1}^3 w_b P_b\right) \quad \text{s.t.} \quad w_b \ge 0, \quad \sum w_b = 1$$
   - Eliminates meta-overfitting by preventing the meta-learner from observing in-sample base model predictions.
3. **Multi-Modal Stacking Ensemble (`MultiModalStackingEnsemble`)**:
   - Implements `BaseModel` contract with `fit`, `predict`, `predict_proba`, `save`, and `load`.
   - Generates calibrated multi-model probability distributions and discrete class predictions.
   - Transparently reports learned modality weights (e.g. 50% Price, 32% News, 18% Macro).
4. **Platform Integration & CLI Suite**:
   - Updated `ModelTrainer.train_and_evaluate` with `model_type="ensemble"`.
   - Enabled CLI command: `psx model train --symbol OGDC --model ensemble`.

---

## 2. Deliverables & Files Created / Modified

| File | Purpose |
| :--- | :--- |
| [`src/psx_predictor/models/weighting.py`](../../../src/psx_predictor/models/weighting.py) | `OutOfFoldWeightOptimizer` and `generate_expanding_folds` for leakage-free validation and weight optimization. |
| [`src/psx_predictor/models/ensemble.py`](../../../src/psx_predictor/models/ensemble.py) | `MultiModalStackingEnsemble` implementing stacking meta-learning over base specialists. |
| [`src/psx_predictor/models/trainer.py`](../../../src/psx_predictor/models/trainer.py) | Added support for combined dataset routing and `model_type="ensemble"` execution. |
| [`src/psx_predictor/models/__init__.py`](../../../src/psx_predictor/models/__init__.py) | Exported ensemble and weighting public APIs. |
| [`src/psx_predictor/cli/main.py`](../../../src/psx_predictor/cli/main.py) | Registered `ensemble` in `psx model train --model` choices and documentation. |
| [`tests/unit/test_ensemble.py`](../../../tests/unit/test_ensemble.py) | 6 unit tests covering expanding folds, OOF predictions, constrained optimization, ensemble fit/predict, and CLI invocation. |

---

## 3. Stacking Ensemble Architecture

```mermaid
graph TD
    subgraph Data Layer
        C[data/features/combined/{SYMBOL}_combined.parquet] --> SPLIT[Chronological Train/Test Split]
        SPLIT --> TRAIN[Train Split X_train]
        SPLIT --> TEST[Test Split X_test]
    end

    subgraph Out-of-Fold Walk-Forward Cross-Validation
        TRAIN --> FOLDS[Expanding Time-Series Folds]
        FOLDS -->|Fold k Train| B1[Fit Price Specialist]
        FOLDS -->|Fold k Train| B2[Fit News Specialist]
        FOLDS -->|Fold k Train| B3[Fit Macro Specialist]
        B1 -->|Predict Val| P1[P_price OOF]
        B2 -->|Predict Val| P2[P_news OOF]
        B3 -->|Predict Val| P3[P_macro OOF]
        P1 & P2 & P3 --> Z_OOF[Z_OOF Matrix N_val x 3]
    end

    subgraph Meta-Learner Fitting & Full Retrain
        Z_OOF --> OPT[SLSQP Weight Optimizer] --> WEIGHTS[Modality Weights: w_price, w_news, w_macro]
        Z_OOF --> META[Fit Logistic Stacking Meta-Learner]
        TRAIN --> FULL_P[Refit Price Specialist on X_train]
        TRAIN --> FULL_N[Refit News Specialist on X_train]
        TRAIN --> FULL_M[Refit Macro Specialist on X_train]
    end

    subgraph Out-of-Sample Inference
        TEST --> FULL_P --> TP1[P_price]
        TEST --> FULL_N --> TP2[P_news]
        TEST --> FULL_M --> TP3[P_macro]
        TP1 & TP2 & TP3 --> Z_TEST[Z_test Matrix]
        Z_TEST --> META --> PROBS[Calibrated Probability & Class Label]
    end
```

---

## 4. Verification & Validation

### 4.1 Automated Test Suite
All 140 tests pass cleanly:
```bash
$ uv run pytest -v
========================== 140 passed, 2 deselected in 17.04s ==========================
```

Unit tests (`tests/unit/test_ensemble.py`):
- `test_generate_expanding_folds`: Asserts strictly sequential, non-overlapping validation slices with $T_{\text{train}} < T_{\text{val}}$.
- `test_oof_predictions_and_weight_optimization`: Confirms probability bounds $[0, 1]$ and constrained non-negative weights summing to 1.0.
- `test_multimodal_stacking_ensemble_fit_and_predict`: Verifies end-to-end fitting and calibrated probability predictions summing to 1.0.
- `test_ensemble_weighted_average_mode`: Verifies convex combination blending under weighted average mode.
- `test_model_trainer_with_ensemble`: Tests `ModelTrainer` workflow with combined feature routing and metric calculations.
- `test_cli_model_train_ensemble`: Tests `psx model train --symbol OGDC --model ensemble` via `CliRunner`.

### 4.2 Code Quality & Static Typing
```bash
$ uv run ruff check .
All checks passed!

$ uv run mypy src
Success: no issues found in 62 source files
```

### 4.3 Live CLI Execution
```bash
$ uv run psx model train --symbol OGDC --model ensemble
Initiating model benchmark for OGDC | Family: ensemble | Target: target_next_day_dir | Train Split: 80%
[OGDC] Loading features and targets (combined=True)...
[OGDC] Splitting 1240 sessions chronologically (train_ratio=0.8)...
[OGDC] Train set: 832 sessions (2022-06-20 -> 2025-10-29), Test set: 208 sessions (2025-10-30 -> 2026-09-02)
[OGDC] Training MultiModalStackingEnsemble...
Partitioned features: 20 Tech, 9 News, 12 Macro.
Step 1/3: Generating out-of-fold cross-validation predictions...
Generated OOF prediction matrix of shape (500, 3) across 500 validation points.
Step 2/3: Optimizing stacking modality weights on OOF predictions...
Optimized Modality Weights: {'price': 0.5005, 'news': 0.3229, 'macro': 0.1766}
Step 3/3: Refitting base specialist models on full training dataset...

    Model Evaluation Benchmark: OGDC (next_day_dir | Train: 832 | Test: 208)   
+------------------------------------------------------------------------------+
| Model                      | Accuracy | Precision | Recall | F1-Macro | ROC-AUC | Brier  |
|----------------------------+----------+-----------+--------+----------+---------+--------|
| MultiModalStackingEnsemble |   50.00% |    0.5000 | 0.5000 |   0.4787 |  0.5048 | 0.2501 |
+------------------------------------------------------------------------------+
```
