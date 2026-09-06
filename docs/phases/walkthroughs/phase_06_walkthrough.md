# Walkthrough — Phase 06: Supervised Prediction Targets

**Status:** Completed & Validated  
**Date:** September 2026  
**Specification:** [phase_06_prediction_targets.md](../phase_06_prediction_targets.md)

---

## 1. Objective Accomplished
Engineered mathematically verified, future-aligned supervised prediction targets for ML classification and multi-horizon regression tasks. Enforced rigorous forward-alignment and masking rules ensuring zero target leakage into contemporaneous feature vectors ($t$). The pipeline serializes 48-column augmented feature datasets to `data/features/technical/{symbol}_tech_features.parquet`.

---

## 2. Deliverables & Files Created / Modified

| File | Purpose |
| :--- | :--- |
| [`src/psx_predictor/features/targets.py`](../../../src/psx_predictor/features/targets.py) | Mathematical implementations for `target_next_day_dir` (binary direction $y_t \in \{0.0, 1.0\}$), `target_next_day_3class` ($\pm 0.75\%$ noise-filtered 3-class $y_t \in \{-1.0, 0.0, 1.0\}$), and `target_return_5d` (5-session cumulative forward return). Accompanied by `compute_all_prediction_targets`. |
| [`src/psx_predictor/features/builder.py`](../../../src/psx_predictor/features/builder.py) | Updated `TechnicalFeatureBuilder` with `with_targets: bool = True`, `threshold: float = 0.0075`, and `target_horizon: int = 5` flags to automatically append forward labels. |
| [`src/psx_predictor/features/__init__.py`](../../../src/psx_predictor/features/__init__.py) | Exported prediction target functions and types. |
| [`src/psx_predictor/cli/main.py`](../../../src/psx_predictor/cli/main.py) | Added `--with-targets / --without-targets` CLI flag to `psx features build`. |
| [`tests/unit/test_targets.py`](../../../tests/unit/test_targets.py) | 6 comprehensive unit tests validating exact hand-calculated targets, tail NaN invariants, threshold boundary conditions, feature leakage invariance, builder integration, and CLI execution. |

---

## 3. Mathematical Specifications & Tail Invariants

### 3.1 Targets Overview
1. **Target A1 — Next-Day Direction (`target_next_day_dir`):**
   $$y_t = \begin{cases} 1.0 & \text{if } P_{t+1} > P_t \\ 0.0 & \text{if } P_{t+1} \le P_t \end{cases}$$
   *Tail Invariant:* Session $T$ (last session) is strictly `NaN`.

2. **Target A2 — Thresholded 3-Class (`target_next_day_3class`):**
   Noise filter mitigating transaction friction, commissions, and bid-ask spreads ($\theta = 0.0075$ or $0.75\%$):
   $$y_t = \begin{cases} 
   1.0 & \text{if } (P_{t+1} - P_t) / P_t > +\theta \\
   -1.0 & \text{if } (P_{t+1} - P_t) / P_t < -\theta \\
   0.0 & \text{if } |(P_{t+1} - P_t) / P_t| \le \theta
   \end{cases}$$
   *Tail Invariant:* Session $T$ is strictly `NaN`.

3. **Target B — 5-Session Forward Cumulative Return (`target_return_5d`):**
   $$R_{t, t+5} = \frac{P_{t+5}}{P_t} - 1.0$$
   *Tail Invariant:* Sessions $T-4, T-3, T-2, T-1, T$ (last 5 sessions) are strictly `NaN`.

---

## 4. Verification & Test Results

### 4.1 Code Quality & Static Typing
```powershell
uv run ruff format --check .
# Output: 37 files left unchanged

uv run ruff check .
# Output: All checks passed!

uv run mypy src
# Output: Success: no issues found in 25 source files
```

### 4.2 Automated Unit Test Execution
```powershell
uv run pytest -v
```
**Test Results:**
```text
============================= test session starts =============================
platform win32 -- Python 3.12.11, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\Development\PSX_Prediction
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.15.1, respx-0.23.1
collected 64 items / 2 deselected / 62 selected

tests\unit\test_collectors.py ............                               [ 19%]
tests\unit\test_config.py ..........                                     [ 35%]
tests\unit\test_features.py ......                                       [ 45%]
tests\unit\test_pipeline.py ..........                                   [ 61%]
tests\unit\test_storage.py ............                                  [ 80%]
tests\unit\test_targets.py ......                                        [ 90%]
tests\unit\test_updater.py ......                                        [100%]

====================== 62 passed, 2 deselected in 2.75s =======================
```

### 4.3 Live Production Execution
```powershell
uv run psx features build --with-targets --symbols OGDC,PPL,GGL
```
**Output:**
```text
                     Technical Feature Generation Results                      
+-----------------------------------------------------------------------------+
| Symbol | Status  | Sessions | Features |        Date Span        | Duration |
|--------+---------+----------+----------+-------------------------+----------|
| GGL    | SUCCESS |     1240 |       48 |      2021-09-05 ->      |    0.04s |
|        |         |          |          |       2026-09-03        |          |
| OGDC   | SUCCESS |     1240 |       48 |      2021-09-05 ->      |    0.02s |
|        |         |          |          |       2026-09-03        |          |
| PPL    | SUCCESS |     1240 |       48 |      2021-09-05 ->      |    0.02s |
|        |         |          |          |       2026-09-03        |          |
+-----------------------------------------------------------------------------+
+----------------------------- Features Summary ------------------------------+
| Total Symbols: 3  |  Successful: 3  |  Failed: 0                            |
| Total Feature Rows Generated: 3,720                                         |
+-----------------------------------------------------------------------------+
```

### 4.4 Live Parquet Tail Inspection
```powershell
uv run python -c "import pandas as pd; df = pd.read_parquet('data/features/technical/OGDC_tech_features.parquet'); print(df[['trade_date', 'close', 'target_next_day_dir', 'target_next_day_3class', 'target_return_5d']].tail(7))"
```
```text
      trade_date   close  target_next_day_dir  target_next_day_3class  target_return_5d
1233  2026-08-26  323.88                  1.0                     0.0          0.009695
1234  2026-08-27  325.20                  1.0                     1.0          0.011070
1235  2026-08-30  328.70                  1.0                     1.0               NaN
1236  2026-08-31  331.69                  0.0                    -1.0               NaN
1237  2026-09-01  328.44                  0.0                     0.0               NaN
1238  2026-09-02  327.02                  1.0                     0.0               NaN
1239  2026-09-03  328.80                  NaN                     NaN               NaN
```
The exact forward NaN invariants are verified: 1-day targets are `NaN` only on row 1239; 5-day return is `NaN` across the last 5 sessions (1235–1239).

---

## 5. Phase 06 Sign-off
Phase 06 is fully validated and ready for Phase 07 (Baseline Prediction Models).
