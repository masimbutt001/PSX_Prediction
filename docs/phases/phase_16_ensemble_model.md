# Phase 16 — Multi-Model Ensemble

## 1. Objective
Construct a stacking meta-model combining specialized sub-models (Price/Technical model, News Sentiment model, Macro model). Learn ensemble weights via out-of-fold walk-forward validation rather than heuristic weighting.

---

## 2. Deliverables & Files to Create
```text
src/psx_predictor/models/
├── ensemble.py                        # Stacking meta-learner
└── weighting.py                       # Out-of-fold weight optimizer

tests/unit/
└── test_ensemble.py                   # Ensemble training and out-of-fold leakage tests
```

---

## 3. Detailed Specifications

### 3.1 Ensemble Architecture
```text
[ Technical Features ] ──► [ XGBoost Price Model ] ──► P_price
[ News Features ]      ──► [ NLP News Model ]      ──► P_news    ──► [ Meta-Learner ] ──► Final Probability
[ Macro Features ]     ──► [ Ridge Macro Model ]   ──► P_macro       (Logistic Reg)
```
- The meta-learner is trained strictly on out-of-fold predictions from earlier walk-forward windows.
- Prevents meta-learner from overfitting to training predictions.

---

## 4. Testing Plan
- `test_ensemble_out_of_fold_training()`: Asserts meta-learner weights are derived from held-out validation predictions.
- `test_ensemble_improves_or_matches_single_best()`: Compares Brier score of ensemble vs. individual models.

---

## 5. Validation Commands
```bash
pytest tests/unit/test_ensemble.py -v
psx model train --model ensemble --symbol OGDC
ruff check .
```

---

## 6. Phase Completion Exit Criteria
- [ ] Multi-model ensemble trained with out-of-fold walk-forward validation.
- [ ] Evaluation shows calibrated probabilities and stable weights.
- [ ] Phase 16 completion report documented.
