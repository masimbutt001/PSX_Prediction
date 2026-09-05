# Phase 06 — Supervised Prediction Targets

## 1. Objective
Generate mathematically rigorous, future-aligned supervised learning targets for classification and regression tasks. Ensure rigorous alignment tests guarantee that input features at time $t$ do not contain target information from time $t+1$ or beyond.

---

## 2. Deliverables & Files to Create
```text
src/psx_predictor/features/
└── targets.py                         # Target generation and alignment validation

tests/unit/
└── test_targets.py                    # Alignment, off-by-one, and leakage tests
```

---

## 3. Detailed Specifications

### 3.1 Target Definitions
1. **Target A1 — Binary Direction (`target_next_day_dir`):**
   $$y_t = \begin{cases} 1 & \text{if } \text{adj\_close}_{t+1} > \text{adj\_close}_t \\ 0 & \text{if } \text{adj\_close}_{t+1} \le \text{adj\_close}_t \end{cases}$$
   *Training Rule:* The last row of any dataset must have target `NaN` and be dropped from training.

2. **Target A2 — Thresholded 3-Class (`target_next_day_3class`):**
   Filters bid-ask spread noise and transaction costs:
   $$y_t = \begin{cases} 
   \text{UP} & \text{if } (\text{adj\_close}_{t+1} / \text{adj\_close}_t - 1) > +0.0075 \\
   \text{DOWN} & \text{if } (\text{adj\_close}_{t+1} / \text{adj\_close}_t - 1) < -0.0075 \\
   \text{NEUTRAL} & \text{otherwise}
   \end{cases}$$

3. **Target B — 5-Session Future Return (`target_return_5d`):**
   $$R_{t, t+5} = \frac{\text{adj\_close}_{t+5}}{\text{adj\_close}_t} - 1.0$$
   *Training Rule:* The last 5 rows of any dataset must have target `NaN` and be excluded from training.

---

## 4. Testing Plan
- `test_target_alignment_hand_calculated()`: Verifies target labels on a 10-row synthetic sequence against manual calculation.
- `test_last_row_target_is_nan()`: Asserts target on the final session is strictly `NaN`.
- `test_target_no_leakage_into_features()`: Verifies that shifting future prices does not alter feature vectors at date $t$.

---

## 5. Validation Commands
```bash
pytest tests/unit/test_targets.py -v
psx features build --with-targets
ruff check .
```

---

## 6. Phase Completion Exit Criteria
- [ ] Targets generated cleanly and appended to feature datasets.
- [ ] Explicit exclusion of incomplete future rows validated.
- [ ] Phase 6 completion report documented.
