# Walkthrough — Phase 05: Price Feature Engineering

**Status:** Completed & Validated  
**Date:** September 2026  
**Specification:** [phase_05_price_features.md](../phase_05_price_features.md)

---

## 1. Objective Accomplished
Constructed a deterministic, vectorized technical feature engineering engine. All indicators are calculated strictly from contemporaneous and historical pricing sessions with zero look-ahead bias (mathematically validated via price perturbation testing). The pipeline serializes 45-column feature datasets to `data/features/technical/{symbol}_tech_features.parquet`.

---

## 2. Deliverables & Files Created

| File | Purpose |
| :--- | :--- |
| [`src/psx_predictor/features/technical.py`](../../../src/psx_predictor/features/technical.py) | Vectorized indicator functions: log returns ($1d, 3d, 5d, 10d, 20d$), moving averages ($SMA_{5, 10, 20, 50, 200}$, $EMA_{12, 26}$, normalized price distances), Wilder's RSI-14, MACD (12, 26, 9), ROC-10, 20-day annualized volatility, 14-day ATR & ATR%, Bollinger Bands ($\pm 2\sigma$, bandwidth, %B), relative volume, OBV, and volume ROC. |
| [`src/psx_predictor/features/builder.py`](../../../src/psx_predictor/features/builder.py) | `TechnicalFeatureBuilder` orchestrating batch feature generation across tickers with failure isolation, atomic Parquet storage, and manifest tracking (`feature_manifest.json`). |
| [`src/psx_predictor/features/__init__.py`](../../../src/psx_predictor/features/__init__.py) | Package initialization exporting feature engineering functions and builder classes. |
| [`src/psx_predictor/cli/main.py`](../../../src/psx_predictor/cli/main.py) | Registered `features` CLI group with `psx features build` command. |
| [`tests/unit/test_features.py`](../../../tests/unit/test_features.py) | 6 unit tests validating exact mathematical values, zero look-ahead perturbation invariance, infinite/NaN bounds checking, atomic Parquet write, and CLI execution. |

---

## 3. Verification & Test Results

### 3.1 Code Quality & Static Typing
```powershell
uv run ruff format --check .
# Result: 35 files already formatted

uv run ruff check .
# Result: All checks passed!

uv run mypy src
# Result: Success: no issues found in 24 source files
```

### 3.2 Automated Test Suite Execution
```powershell
uv run pytest -v
```
**Output:**
```text
============================= test session starts =============================
platform win32 -- Python 3.12.11, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\Development\PSX_Prediction
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.15.1, respx-0.23.1
collected 58 items / 2 deselected / 56 selected

tests\unit\test_collectors.py ............                               [ 21%]
tests\unit\test_config.py ..........                                     [ 39%]
tests\unit\test_features.py ......                                       [ 50%]
tests\unit\test_pipeline.py ..........                                   [ 67%]
tests\unit\test_storage.py ............                                  [ 89%]
tests\unit\test_updater.py ......                                        [100%]

====================== 56 passed, 2 deselected in 2.24s =======================
```

### 3.3 Zero Look-Ahead Bias Mathematical Proof
Validated via `test_zero_lookahead_perturbation`:
- Generated a synthetic multi-day price series and calculated feature matrix $F_1$.
- Altered solely the final session $T$ by $+250\%$ and recomputed feature matrix $F_2$.
- Asserted that for all prior sessions $0 \dots T-1$, across all 33 derived feature columns, $F_1[0 \dots T-1] \equiv F_2[0 \dots T-1]$ with zero floating-point deviation.

### 3.4 Live CLI Feature Generation
Executed feature generation across the 5-year bootstrapped datasets:
```powershell
uv run psx features build --symbols OGDC,PPL,GGL
```
**Output:**
```text
Starting technical feature generation | Mode: PERSIST
[GGL] Saved 1240 records to data\features\technical\GGL_tech_features.parquet
[OGDC] Saved 1240 records to data\features\technical\OGDC_tech_features.parquet
[PPL] Saved 1240 records to data\features\technical\PPL_tech_features.parquet
Saved feature manifest to data\features\technical\feature_manifest.json
                     Technical Feature Generation Results                      
+-----------------------------------------------------------------------------+
| Symbol | Status  | Sessions | Features |        Date Span        | Duration |
|--------+---------+----------+----------+-------------------------+----------|
| GGL    | SUCCESS |     1240 |       45 |  2021-09-05 -> 2026-09-03|    0.05s |
| OGDC   | SUCCESS |     1240 |       45 |  2021-09-05 -> 2026-09-03|    0.02s |
| PPL    | SUCCESS |     1240 |       45 |  2021-09-05 -> 2026-09-03|    0.02s |
+-----------------------------------------------------------------------------+
Features Summary: Total: 3 | Successful: 3 | Failed: 0 | Total Records: 3,720
```
Generated 45 feature dimensions across 3,720 sessions in **0.09s total execution time**.
