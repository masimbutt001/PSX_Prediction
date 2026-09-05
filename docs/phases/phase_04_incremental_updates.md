# Phase 04 — Incremental Daily Updates

## 1. Objective
Enable idempotent, lightweight daily updates without re-downloading entire multi-year histories. The update mechanism must determine the latest existing local trading date per symbol, query only newer data from the provider, merge records, recompute adjustments, and atomically persist updates.

---

## 2. Deliverables & Files to Create
```text
src/psx_predictor/processing/
├── updater.py                         # Incremental merge logic and state manager

tests/unit/
└── test_updater.py                    # Incremental update & idempotency tests
```

---

## 3. Detailed Specifications

### 3.1 Latest-Date Resolution
- Query DuckDB or Parquet metadata to extract:
  $$T_{max} = \max(\text{trade\_date}) \quad \text{for symbol } S$$
- Request provider data starting from $T_{max} + 1 \text{ day}$.
- If $T_{max}$ is already the latest trading session (e.g. today's market close), exit cleanly with message: `Already up to date`.

### 3.2 Idempotent Merge & Re-Adjustment
- Append new records to existing processed dataframe.
- Deduplicate on `(symbol, trade_date)` retaining latest record.
- Sort chronologically by `trade_date ASC`.
- Recalculate `adjusted_close` retroactively if a dividend or bonus action was recorded in the new delta.
- Atomically write updated Parquet file.

---

## 4. Testing Plan
- `test_incremental_fetch_determines_correct_start_date()`: Validates provider query begins at $T_{max} + 1$.
- `test_idempotency_double_run()`: Running `psx data update` twice in succession results in zero duplicate rows and identical row counts.
- `test_dividend_in_delta_updates_adjustments()`: Ensures an ex-dividend date in the new daily update adjusts backward prices properly.

---

## 5. Validation Commands
```bash
pytest tests/unit/test_updater.py -v
psx data update
psx data update   # Must be idempotent
ruff check .
```

---

## 6. Phase Completion Exit Criteria
- [ ] `psx data update` executes incrementally and updates existing Parquet files.
- [ ] Running update multiple times is 100% idempotent.
- [ ] Phase 4 completion report documented.
