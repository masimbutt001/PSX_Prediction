# Phase 14 — Macroeconomic Data Ingestion

## 1. Objective
Ingest high-impact Pakistani macroeconomic indicators (State Bank of Pakistan policy rate, KIBOR, USD/PKR, CPI inflation, and Brent crude oil). Maintain strict separation between observation dates and public release timestamps to prevent historical backfill leakage.

---

## 2. Deliverables & Files to Create
```text
src/psx_predictor/macro/
├── __init__.py
├── sbp_collector.py                   # SBP policy rate, FX reserves, KIBOR collector
├── market_collector.py                # USD/PKR exchange rate, Brent oil, Gold
└── storage.py                         # Macro Parquet persistence

tests/unit/
└── test_macro.py                      # Macro collector parsing and release-date validation
```

---

## 3. Detailed Specifications

### 3.1 Indicators Tracked
1. **USD/PKR Spot Rate:** Daily closing interbank exchange rate.
2. **SBP Monetary Policy Rate:** Announced via Monetary Policy Statements (MPS).
3. **6-Month KIBOR:** Interbank benchmark lending rate.
4. **CPI Inflation (YoY / MoM):** Released monthly by Pakistan Bureau of Statistics (PBS).
5. **Brent Crude Oil:** Crucial global macro driver for PSX E&P (OGDC, PPL, MARI) and Fertilizer stocks.
6. **KSE-100 Index Level:** Broad market benchmark.

### 3.2 Look-Ahead Bias Prevention Rule
- Monthly CPI for January is released around February 1st.
- A model predicting on January 15th **cannot use January CPI**. It can only use the most recent reading publicly available on January 15th (December CPI).
- Macro dataset records both `observation_date` (e.g. Jan 2024) and `public_release_date` (e.g. 2024-02-01).

---

## 4. Testing Plan
- `test_macro_forward_fill_only_after_release()`: Verifies that forward-fill begins at `public_release_date`, not `observation_date`.
- `test_oil_price_merging()`: Verifies Brent crude joins on contemporaneous trade dates.

---

## 5. Validation Commands
```bash
pytest tests/unit/test_macro.py -v
psx macro update
ruff check .
```

---

## 6. Phase Completion Exit Criteria
- [ ] Macro indicators ingested and stored in `data/processed/macro/`.
- [ ] Release-date temporal integrity strictly enforced.
- [ ] Phase 14 completion report documented.
