# Phase 05 — Price Feature Engineering

## 1. Objective
Build a deterministic, vectorized technical feature engineering engine. All indicators must compute strictly from historical and contemporaneous data without look-ahead bias. Features are written to `data/features/technical/{symbol}_tech_features.parquet`.

---

## 2. Deliverables & Files to Create
```text
src/psx_predictor/features/
├── __init__.py
├── technical.py                       # Vectorized technical indicator computations
└── builder.py                         # Feature dataset pipeline

tests/unit/
└── test_features.py                   # Numerical correctness and look-ahead verification
```

---

## 3. Detailed Specifications

### 3.1 Implemented Indicators
1. **Log Returns:**
   - $r_{1d} = \ln(\text{adj\_close}_t / \text{adj\_close}_{t-1})$
   - $r_{3d}, r_{5d}, r_{10d}, r_{20d}$
2. **Moving Averages:**
   - Simple Moving Averages: SMA(5), SMA(10), SMA(20), SMA(50), SMA(200)
   - Exponential Moving Averages: EMA(12), EMA(26)
   - Normalized distance to SMA: $(\text{Close} - \text{SMA}_N) / \text{SMA}_N$
3. **Momentum:**
   - 14-period RSI (Wilder's smoothing method)
   - MACD (12, 26, 9): Line, Signal line, and Histogram
   - 10-period Rate of Change (ROC)
4. **Volatility:**
   - Rolling standard deviation of 1-day returns (20-day window, annualized)
   - 14-period Average True Range (ATR)
   - Bollinger Bands (20-period, $\pm 2$ std dev): Upper, Lower, Bandwidth
5. **Volume Dynamics:**
   - Relative Volume: $\text{Volume}_t / \text{SMA}(\text{Volume}, 20)$
   - On-Balance Volume (OBV)
   - Volume Rate of Change (5-day)

---

## 4. Testing Plan
- `test_indicator_exact_values()`: Compares calculated RSI, MACD, and SMA against hand-verified fixture values.
- `test_zero_lookahead_perturbation()`: Changes the final row's close price and verifies all prior rows' features remain 100% unchanged.
- `test_no_inf_or_unhandled_nan()`: Verifies feature output has NaN only in initial warmup periods (e.g. first 200 rows for SMA-200) and zero `inf` values.

---

## 5. Validation Commands
```bash
pytest tests/unit/test_features.py -v
psx features build --symbol OGDC
ruff check .
```

---

## 6. Phase Completion Exit Criteria
- [ ] Technical features generated deterministically.
- [ ] Leakage tests pass with zero deviation.
- [ ] Parquet feature dataset saved to `data/features/technical/`.
- [ ] Phase 5 completion report documented.
