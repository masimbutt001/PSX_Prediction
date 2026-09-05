# Phase 15 — Combined Multi-Modal Feature Set & Ablations

## 1. Objective
Merge price technical indicators, market session-aligned news sentiment, and macroeconomic series into a unified point-in-time dataset. Enable modular feature group toggles to run empirical ablation studies (testing whether news and macro actually improve predictive edge over price alone).

---

## 2. Deliverables & Files to Create
```text
src/psx_predictor/features/
├── merger.py                          # Point-in-time multi-modal merger
└── ablations.py                       # Automated ablation study runner

tests/unit/
└── test_merger.py                     # Multi-table join integrity and point-in-time leakage tests
```

---

## 3. Detailed Specifications

### 3.1 Point-in-Time Join Logic
- Performs strictly retrospective joins:
  $$\text{FeatureVector}(S, t) = \left[ \text{Tech}(S, t), \text{News}(S, t), \text{Macro}(t) \right]$$
- Uses DuckDB `ASOF JOIN` or Pandas merge-as-of where appropriate.

### 3.2 Ablation Study Configurations
- Experiment 1: `Price Only`
- Experiment 2: `Price + Relative Market (KSE-100)`
- Experiment 3: `Price + News`
- Experiment 4: `Price + Macro`
- Experiment 5: `Price + News + Macro (Full)`
- Outputs performance comparison table across identical walk-forward validation folds.

---

## 4. Testing Plan
- `test_asof_join_does_not_peek_future()`: Verifies macro release after date $t$ is not present in row $t$.
- `test_ablation_configurations_load()`: Verifies feature group subsets toggle on/off cleanly without schema errors.

---

## 5. Validation Commands
```bash
pytest tests/unit/test_merger.py -v
psx features merge --symbol OGDC
psx model ablation --symbol OGDC
ruff check .
```

---

## 6. Phase Completion Exit Criteria
- [ ] Combined datasets generated in `data/features/combined/`.
- [ ] Ablation report quantifies the exact contribution of news and macro signals.
- [ ] Phase 15 completion report documented.
